from glob import glob
import os

from utils.constants import APP_CACHE
from utils.utils import execute_command

def infer(entrypoint: str = "grounding_dino",
          model_path: str = None,
          detection_classes: list = None, 
          results_dir: str = None,
          inference_input_path: str = None):
    
    """Function to perform grounding dino inference."""
    assert execute_command(f"{entrypoint} --help"), f"Inference entrypoint [{entrypoint}] isn't supported."
    overrides = []
    model_file = glob(f'{model_path}/*.pth')
    if isinstance(model_file, list):
        model_file = model_file[0]
    if not isinstance(inference_input_path, list):
        inference_input_path = [inference_input_path]
        overrides.append(f"dataset.infer_data_sources.image_dir={inference_input_path}")
    if results_dir:
        overrides.append(f"results_dir={results_dir}")
    command = f"{entrypoint} inference -e /workspace/tao_mm_workflows/config/gdino_spec.yaml"
    if model_path:
        overrides.append(f"inference.checkpoint={model_file}")
    if isinstance(detection_classes, list):
        detection_classes_str = f"\\\"dataset.infer_data_sources.captions={detection_classes}\\\""
        overrides.append(detection_classes_str)
    compiled_command = f"{command} {' '.join(overrides)}"
    print(compiled_command)
    execute_command(
        compiled_command
    )
