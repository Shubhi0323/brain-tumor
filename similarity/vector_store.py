"""
Vector Similarity Engine
==========================
Stores patient embeddings and retrieves similar tumor cases.
Primary: Weaviate vector database.
Fallback: NumPy brute-force cosine similarity on local .npy files.

Production features:
  - L2 embedding normalization before similarity computation
  - Minimum similarity threshold filtering (configurable, default 0.5)
  - Embedding diagnostics (norm, sparsity, degeneracy detection)
"""
import os
import json
import numpy as np

from utils.pipeline_logger import get_logger

logger = get_logger("Similarity")

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

COLLECTION_NAME = "TumorCase"


def _get_sim_config():
    try:
        from config.config_loader import get_config
        return get_config().similarity
    except Exception:
        from dataclasses import dataclass
        @dataclass
        class _D:
            top_k: int = 5
            min_similarity_threshold: float = 0.5
            normalize_embeddings: bool = True
            embedding_dim: int = 768
        return _D()


# ─── Embedding Utilities ─────────────────────────────────────────────

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector."""
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding


def compute_embedding_diagnostics(embedding: np.ndarray) -> dict:
    """
    Compute embedding quality diagnostics.
    Returns norm, sparsity, effective dimensions, degeneracy flag.
    """
    norm = float(np.linalg.norm(embedding))
    total = len(embedding)
    nonzero = int(np.count_nonzero(embedding))
    sparsity = 1.0 - (nonzero / max(total, 1))

    has_nan = bool(np.any(np.isnan(embedding)))
    has_inf = bool(np.any(np.isinf(embedding)))
    is_degenerate = has_nan or has_inf or norm < 1e-8 or sparsity > 0.99

    # Effective dimensions via participation ratio
    effective_dims = 0
    if norm > 0 and not is_degenerate:
        normalized = embedding / norm
        p = normalized ** 2
        p = p[p > 0]
        if len(p) > 0:
            effective_dims = int(1.0 / np.sum(p ** 2)) if np.sum(p ** 2) > 0 else 0

    return {
        "embedding_norm": round(norm, 4),
        "sparsity": round(sparsity, 4),
        "nonzero_dims": nonzero,
        "total_dims": total,
        "effective_dims": effective_dims,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_degenerate": is_degenerate,
    }


# ─── Weaviate Backend ────────────────────────────────────────────────

def _get_weaviate_client(url: str = None):
    if url:
        client = weaviate.connect_to_custom(
            http_host=url.split("://")[-1].split(":")[0],
            http_port=int(url.split(":")[-1]) if ":" in url.split("://")[-1] else 8080,
            http_secure=url.startswith("https"),
            grpc_host=url.split("://")[-1].split(":")[0],
            grpc_port=50051, grpc_secure=False,
        )
    else:
        client = weaviate.connect_to_embedded()
    return client


def ensure_collection(client):
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="patient_id", data_type=DataType.TEXT),
                Property(name="tumor_location", data_type=DataType.TEXT_ARRAY),
                Property(name="volume_severity", data_type=DataType.TEXT),
                Property(name="inferred_symptoms", data_type=DataType.TEXT_ARRAY),
                Property(name="tumor_volume", data_type=DataType.NUMBER),
                Property(name="max_diameter", data_type=DataType.NUMBER),
                Property(name="sphericity", data_type=DataType.NUMBER),
                Property(name="primary_location", data_type=DataType.TEXT),
            ],
        )
    return client.collections.get(COLLECTION_NAME)


def upsert_patient_weaviate(client, patient_id, embedding, clinical_profile):
    collection = ensure_collection(client)
    morph = clinical_profile.get("morphology", {})
    properties = {
        "patient_id": patient_id,
        "tumor_location": clinical_profile.get("tumor_location", []),
        "volume_severity": clinical_profile.get("volume_severity", "unknown"),
        "inferred_symptoms": clinical_profile.get("inferred_symptoms", []),
        "tumor_volume": float(morph.get("tumor_volume", 0)),
        "max_diameter": float(morph.get("max_diameter", 0)),
        "sphericity": float(morph.get("sphericity", 0)),
        "primary_location": clinical_profile.get("primary_location", "unknown"),
    }

    existing = collection.query.fetch_objects(
        filters=weaviate.classes.query.Filter.by_property("patient_id").equal(patient_id),
        limit=1,
    )
    for obj in existing.objects:
        collection.data.delete_by_id(obj.uuid)
    collection.data.insert(properties=properties, vector=embedding)


def query_similar_weaviate(client, embedding, top_k=5, exclude_patient=None):
    collection = ensure_collection(client)
    results = collection.query.near_vector(
        near_vector=embedding,
        limit=top_k + (1 if exclude_patient else 0),
        return_metadata=MetadataQuery(distance=True),
    )

    similar_cases = []
    for obj in results.objects:
        pid = obj.properties.get("patient_id", "")
        if exclude_patient and pid == exclude_patient:
            continue
        distance = obj.metadata.distance if obj.metadata else None
        similarity = 1.0 - distance if distance is not None else None
        similar_cases.append({
            "patient_id": pid,
            "tumor_location": obj.properties.get("tumor_location", []),
            "volume_severity": obj.properties.get("volume_severity", ""),
            "inferred_symptoms": obj.properties.get("inferred_symptoms", []),
            "tumor_volume": obj.properties.get("tumor_volume", 0),
            "primary_location": obj.properties.get("primary_location", ""),
            "distance": distance,
            "similarity": similarity,
        })
    return similar_cases[:top_k]


# ─── NumPy Fallback Backend ──────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    cfg = _get_sim_config()

    if cfg.normalize_embeddings:
        a = normalize_embedding(a)
        b = normalize_embedding(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def query_similar_numpy(query_embedding, embeddings_dir, clinical_dir,
                        top_k=5, exclude_patient=None):
    cfg = _get_sim_config()
    similarities = []

    if not os.path.isdir(embeddings_dir):
        return []

    query_emb = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding
    if cfg.normalize_embeddings:
        query_emb = normalize_embedding(query_emb)

    for fname in os.listdir(embeddings_dir):
        if not fname.endswith("_embedding.npy"):
            continue
        pid = fname.replace("_embedding.npy", "")
        if exclude_patient and pid == exclude_patient:
            continue

        emb_path = os.path.join(embeddings_dir, fname)
        try:
            stored_emb = np.load(emb_path)
            if cfg.normalize_embeddings:
                stored_emb = normalize_embedding(stored_emb)
        except Exception:
            continue

        sim = _cosine_similarity(query_emb, stored_emb)

        # Apply minimum threshold
        if sim < cfg.min_similarity_threshold:
            continue

        clinical_path = os.path.join(clinical_dir, f"{pid}_clinical.json")
        profile = {}
        if os.path.exists(clinical_path):
            try:
                with open(clinical_path, "r") as f:
                    profile = json.load(f)
            except Exception:
                pass

        similarities.append({
            "patient_id": pid,
            "similarity": sim,
            "tumor_location": profile.get("tumor_location", []),
            "volume_severity": profile.get("volume_severity", ""),
            "inferred_symptoms": profile.get("inferred_symptoms", []),
            "tumor_volume": profile.get("morphology", {}).get("tumor_volume", 0),
            "primary_location": profile.get("primary_location", ""),
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# ─── Unified Interface ───────────────────────────────────────────────

def upsert_patient(patient_id, embedding, clinical_profile, weaviate_url=None):
    cfg = _get_sim_config()

    emb = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
    if cfg.normalize_embeddings:
        emb = normalize_embedding(emb)

    if WEAVIATE_AVAILABLE:
        try:
            client = _get_weaviate_client(weaviate_url)
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            upsert_patient_weaviate(client, patient_id, emb_list, clinical_profile)
            client.close()
            logger.info(f"Stored in Weaviate: {patient_id}")
            return True
        except Exception as e:
            logger.warning(f"Weaviate upsert failed: {e}")
    return False


def query_similar(embedding, output_dir, top_k=5, exclude_patient=None,
                  weaviate_url=None):
    cfg = _get_sim_config()
    effective_k = top_k or cfg.top_k

    emb = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
    if cfg.normalize_embeddings:
        emb = normalize_embedding(emb)

    if WEAVIATE_AVAILABLE:
        try:
            client = _get_weaviate_client(weaviate_url)
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            results = query_similar_weaviate(client, emb_list, effective_k, exclude_patient)
            client.close()
            if results:
                # Apply threshold filtering for Weaviate results too
                results = [
                    r for r in results
                    if r.get("similarity", 1.0) >= cfg.min_similarity_threshold
                ]
                return results
        except Exception as e:
            logger.warning(f"Weaviate query failed, using numpy fallback: {e}")

    emb_dir = os.path.join(output_dir, "embeddings")
    clinical_dir = os.path.join(output_dir, "clinical_features")
    return query_similar_numpy(emb, emb_dir, clinical_dir, effective_k, exclude_patient)


def run_similarity_retrieval(state: dict) -> dict:
    """LangGraph node: Store embedding and retrieve similar tumor cases."""
    patient_id = state["patient_id"]
    output_dir = state["output_dir"]
    embedding = state.get("embedding")
    clinical_profile = state.get("clinical_profile", {})
    errors = list(state.get("errors", []))

    logger.log_stage_start(patient_id)

    if embedding is None:
        msg = f"No embedding for {patient_id}, skipping similarity retrieval."
        logger.warning(msg, patient_id=patient_id)
        errors.append(msg)
        return {**state, "similar_cases": [], "errors": errors}

    try:
        # Embedding diagnostics
        emb_arr = np.array(embedding)
        diagnostics = compute_embedding_diagnostics(emb_arr)
        if diagnostics["is_degenerate"]:
            logger.warning(
                f"Degenerate embedding for {patient_id}: {diagnostics}",
                patient_id=patient_id,
            )
            errors.append(f"Degenerate embedding detected for {patient_id}")

        logger.info(
            f"Embedding diagnostics: norm={diagnostics['embedding_norm']}, "
            f"sparsity={diagnostics['sparsity']:.2f}, "
            f"effective_dims={diagnostics['effective_dims']}",
            patient_id=patient_id,
        )

        # Store this patient
        upsert_patient(patient_id, embedding, clinical_profile)

        # Query similar
        similar = query_similar(
            embedding, output_dir, top_k=5, exclude_patient=patient_id
        )

        if not similar:
            logger.info("No similar cases above threshold", patient_id=patient_id)
        else:
            logger.info(f"Found {len(similar)} similar cases.", patient_id=patient_id)
            for i, case in enumerate(similar):
                score = case.get("similarity", case.get("distance", "N/A"))
                logger.info(f"  {i+1}. {case['patient_id']} (score: {score})")

        return {**state, "similar_cases": similar, "errors": errors}

    except Exception as e:
        msg = f"Error in similarity retrieval for {patient_id}: {e}"
        logger.error(msg, patient_id=patient_id)
        errors.append(msg)
        return {**state, "similar_cases": [], "errors": errors}
