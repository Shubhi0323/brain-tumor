# Brain Tumor Clinical Decision Support Pipeline

An end-to-end agentic pipeline for brain tumor MRI analysis, clinical reasoning, and structured report generation. Built with LangGraph, MONAI DynUNet, BioClinicalBERT, and Llama 3.

Supported dataset formats: DICOM (default) and NIfTI.

## Overview

The system processes brain MRI scans through three phases:

| Phase | Purpose | Components |
|-------|---------|------------|
| **Phase 1** — Imaging | MRI preprocessing, tumor segmentation, radiomics feature extraction, brain region mapping | MONAI DynUNet, PyRadiomics, Harvard-Oxford Atlas |
| **Phase 2** — Intelligence | Tumor classification, similar case retrieval, AI clinical reasoning, report generation | WHO CNS5, RANO criteria, BioClinicalBERT, Llama 3 |
| **Phase 3** — Clinical | Patient memory, physician review, structured CAP report, visualization | ChromaDB, HITL validation, matplotlib |

## Project Structure

```
├── main.py                    # Entry point & CLI
├── pipeline/graph.py          # LangGraph DAG for Phase 1
├── agents/orchestrator.py     # LangGraph DAG for Phase 2 & 3
├── preprocessing/mri_prep.py  # N4 bias correction, skull stripping, normalization
├── segmentation/segresnet_infer.py  # MONAI SegResNet inference + GT fallback
├── radiomics/feature_extractor.py  # PyRadiomics + manual fallback
├── clinical_features/
│   ├── location_mapper.py     # Atlas-based brain region mapping
│   └── symptom_builder.py     # Clinical profile & symptom inference
├── agents/
│   ├── tumor_analysis.py      # WHO/RANO classification via MCP tools
│   ├── similarity_agent.py    # Embedding generation + case retrieval
│   ├── clinical_reasoning.py  # Llama 3 reasoning (or rule-based fallback)
│   └── report_agent.py        # JSON report consolidation
├── embeddings/generator.py    # BioClinicalBERT embeddings
├── similarity/vector_store.py # Weaviate + NumPy cosine fallback
├── mcp_servers/               # MCP tool servers
│   ├── who_classification.py  # WHO CNS5 tumor grading
│   ├── rano_criteria.py       # RANO response assessment
│   └── cap_report.py          # CAP structured pathology report
├── memory/patient_memory.py   # ChromaDB patient history
├── validation/hitl.py         # Human-in-the-loop review
├── visualization/viewer.py    # 5-panel MRI visualization
├── evaluation/                # Dice, Hausdorff, ICC, Precision@K metrics
├── utils/dicom_adapter.py     # DICOM series discovery + NIfTI conversion
├── ui/app.py                  # Streamlit dashboard
├── Dockerfile
├── Dockerfile.ui
└── docker-compose.yml
```

## Quick Start with Docker Compose

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **8 GB RAM** allocated to Docker (Llama 3 + pipeline)
- Input MRI dataset mounted at `data/` (DICOM folders or NIfTI patient folders)

### Run the Pipeline

```bash
# Start all services (Ollama, Weaviate, pipeline) and process 1 patient
docker compose up --build

# Process more patients (edit max_patients in docker-compose.yml or override)
docker compose run --rm pipeline \
  --data_dir /app/data \
  --format dicom --phase all --output_dir /app/outputs \
  --skip_hitl --max_patients 5

# Run only Phase 1 (imaging — no Ollama/Weaviate needed)
docker compose run --rm pipeline \
  --data_dir /app/data \
  --format dicom --phase 1 --output_dir /app/outputs \
  --max_patients 1

# Shut down all services
docker compose down
```

### Run with Evaluation

```bash
docker compose run --rm pipeline \
  --data_dir /app/data \
  --format dicom --phase all --output_dir /app/outputs \
  --skip_hitl --max_patients 1 --evaluate
```

### Launch the Dashboard UI

```bash
# Start the Streamlit dashboard (accessible at http://localhost:8501)
docker compose up ui --build

# Or run it alongside the pipeline
docker compose up --build
```

## Local Setup (without Docker)

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start External Services

```bash
# Ollama (for Llama 3 clinical reasoning)
brew install ollama
ollama serve &
ollama pull llama3

# Weaviate (optional — falls back to NumPy cosine similarity)
docker run -d -p 8080:8080 semitechnologies/weaviate:1.28.2
```

### Run the Pipeline

```bash
# Full pipeline on 1 patient
python main.py \
  --data_dir ./data \
  --format dicom --phase all --output_dir ./outputs \
  --skip_hitl --max_patients 1

# Single patient by ID
python main.py \
  --data_dir ./data \
  --format dicom --phase all --output_dir ./outputs \
  --skip_hitl --patient_id patient_001

# NIfTI format
python main.py \
  --data_dir /path/to/nifti_patients \
  --format nifti --phase all --output_dir ./outputs

# DICOM format (patient folders with DICOM series)
python main.py \
  --data_dir /path/to/dicom_patients \
  --format dicom --phase all --output_dir ./outputs

# Custom Ollama URL
python main.py \
  --data_dir ./data \
  --format dicom --phase all --output_dir ./outputs \
  --ollama_url http://192.168.1.5:11434
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Path to dataset (DICOM studies or NIfTI folders) |
| `--output_dir` | `./outputs` | Output directory for all results |
| `--format` | `dicom` | Dataset format: `dicom` or `nifti` |
| `--phase` | `all` | Pipeline phase: `1`, `2`, `3`, `23`, or `all` |
| `--patient_id` | `None` | Process a single patient (folder name for NIfTI/DICOM) |
| `--max_patients` | `None` | Limit number of patients (DICOM mode) |
| `--skip_hitl` | `False` | Auto-approve physician validation |
| `--ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `--evaluate` | `False` | Run evaluation metrics after pipeline |

## DICOM Input Contract

Expected structure:

```text
data_dir/
  patient_001/
    <series folders with DICOM files>
  patient_002/
    <series folders with DICOM files>
```

The pipeline auto-maps series to modalities using folder names and DICOM metadata
(`SeriesDescription`, `ProtocolName`, and `SequenceName`). Required modalities:
`t1`, `t1ce`, `t2`, `flair`.

## Output Structure

```
outputs/
├── preprocessed/          # Bias-corrected, normalized .npy volumes
├── segmentation/          # Binary tumor masks (.nii.gz)
├── radiomics/             # Shape, intensity, texture features (.json)
├── clinical_features/     # Clinical profiles with symptoms (.json)
├── embeddings/            # BioClinicalBERT 768-dim vectors (.npy)
├── reports/
│   ├── *_report.json      # Full analysis reports
│   └── cap/               # CAP structured pathology reports
├── memory/                # Patient scan history
├── visualizations/        # MRI slices, overlays, 3D renders (.png)
├── chromadb/              # Persistent patient memory DB
└── evaluation/            # Dice, Hausdorff, ICC metrics (.json)
```

## Models Used

| Model | Purpose | Source | Fallback |
|-------|---------|--------|----------|
| DynUNet (MONAI) | Tumor segmentation | Pretrained weights (user-provided) | Ground-truth mask |
| BioClinicalBERT | Clinical text embeddings | HuggingFace | Manual feature vector |
| Llama 3 | Clinical reasoning | Ollama (local) | Rule-based text generation |
| Harvard-Oxford Atlas | Brain region mapping | nilearn | Heuristic geometry |

## Notes

- **No training involved** — all models are used for inference only.
- **Graceful degradation** — every component has a fallback. The pipeline runs without GPU, Ollama, or Weaviate, but output quality decreases.
- **macOS compatible** — runs natively or via Docker Desktop. No CUDA required (CPU inference throughout).
- **HITL validation** — use `--skip_hitl` for batch/automated runs. Without it, the pipeline prompts for physician review in the terminal.
