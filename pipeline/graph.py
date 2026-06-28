from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Import node functions (we will define these next)
from preprocessing.mri_prep import run_preprocessing
from segmentation.segresnet_infer import run_segmentation
from radiomics_pipeline.feature_extractor import extract_radiomics
from clinical_features.location_mapper import map_tumor_location
from clinical_features.symptom_builder import build_clinical_profile

# Define the State schema
class PatientState(TypedDict):
    patient_id: str
    base_dir: str
    output_dir: str
    preprocessed_path: str | None
    segmentation_path: str | None
    radiomics_features: Dict[str, Any]
    tumor_location: List[str]
    clinical_profile: Dict[str, Any]
    errors: List[str]

def build_pipeline():
    """Builds the LangGraph computational pipeline."""
    workflow = StateGraph(PatientState)

    # Add Nodes
    workflow.add_node("preprocess", run_preprocessing)
    workflow.add_node("segment", run_segmentation)
    workflow.add_node("extract_features", extract_radiomics)
    workflow.add_node("map_location", map_tumor_location)
    workflow.add_node("build_clinical", build_clinical_profile)

    # Sequential flow: preprocess → segment → extract_features → map_location → build_clinical
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "segment")
    workflow.add_edge("segment", "extract_features")
    workflow.add_edge("extract_features", "map_location")
    workflow.add_edge("map_location", "build_clinical")
    workflow.add_edge("build_clinical", END)

    # Compile the graph
    app = workflow.compile()
    return app
