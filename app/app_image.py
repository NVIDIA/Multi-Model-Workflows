import glob
import gradio as gr
import json
import logging
import os
import shutil
from pathlib import Path
import tempfile
from tqdm import tqdm
import pandas as pd

from llm_nim.openai_nim import InstructionalNIM, NounChunkNIM
from llm_nim.executor import Executor
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
# from primary_cv.model_handler import ModelInstance
# from primary_cv.gdino_infer import infer as model_inference
from cv_nim.ocd_nim import OCDNIM
from cv_nim.gdino_nim import GDINONIM
from schema.default_config import GradioApp
from utils.constants import NVCF_API, URL
from utils import kitti_util
from utils.utils import execute_command

SAMPLING_FPS = 3

logging.basicConfig(
    format='[%(asctime)s] [TAO Toolkit] [MM] [%(name)s] [%(levelname)s]: %(message)s',
    level='INFO'
)
logger = logging.getLogger(__name__)

config_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def exit_cleanup(intermediate_paths):
    """Clean tmp paths."""
    print("Running cleanup")
    assert isinstance(intermediate_paths, list), (
        "Paths to be removed must be a list."
    )
    for path in intermediate_paths:
        print(f"Removing: {path}")
        shutil.rmtree(path)


def extract_noun_chunks(prompt):
    """Extract noun chunks from user prompt."""
    noun_chunk_extractor = NounChunkNIM(
        URL, NVCF_API
    )
    base_prompt = noun_chunk_extractor.get_base_prompt()
    compiled_prompt = f"{base_prompt} Given text: {prompt}"
    noun_chunk_extractor.assign_model("meta/llama3-70b-instruct")
    data = noun_chunk_extractor.infer(compiled_prompt)
    return data["noun_chunks"]


def generate_analytics(metadata: dict, question: str, code_executor: Executor):
    """Run and cache codellama code for each frame."""
    instructional_nim = InstructionalNIM(
        URL, NVCF_API
    )
    if not code_executor.postprocessor:
        instructional_nim.assign_model(
           "mistralai/codestral-22b-instruct-v0.1"
        )
        base_prompt = instructional_nim.get_base_prompt()
        compiled_prompt = base_prompt.format(
            bbox_prompt=json.dumps(metadata),
            codellama_prompt=question
        )
        function_string = instructional_nim.infer(compiled_prompt)
        code_executor.load_function_from_string(
            function_string
        )
    return code_executor.execute(metadata), code_executor


def run_demo(input_image, question):
    """Run the gradio demo."""
    result = None
    model_output_path = tempfile.mkdtemp()
    output_video_path = tempfile.mkdtemp()
    inference_output_path = tempfile.mkdtemp()
    overlayn_image_path = tempfile.mkdtemp()
    
    # fed into Grounding DINO.
    noun_chunks = ','.join(extract_noun_chunks(question))
    logging.info(f"Noun Chunks: {noun_chunks}")

    try:
        input_image_path = os.path.dirname(input_image)
        #Inference Grounding Dino
        gdino_nim = GDINONIM(NVCF_API)
        gdino_output_path = Path(model_output_path) / "gdino_inference"
        gdino_nim.batch_infer(input_image_path, noun_chunks, gdino_output_path)

        #Inference OCD
        ocd_nim = OCDNIM(NVCF_API)
        ocd_output_path = Path(model_output_path) / "ocd_inference"
        ocd_nim.batch_infer(input_image_path, ocd_output_path)
        analytics_path = os.path.join(model_output_path, "analytics")
        os.makedirs(analytics_path, exist_ok=True)
        annotations_path = os.path.join(model_output_path, "inference/labels")
        frame_files = glob.iglob(os.path.join(annotations_path, "*.txt"))
        code_executor = Executor()
        output_frame_responses = {"Frame ID": [], "LLM Output": []}

        #Read OCD/OCR metadata 
        ocd_metadata = ocd_nim.parse_output(ocd_output_path/Path(input_image).stem)

        #Grounding dino metadata
        gdino_nim.parse_output(gdino_output_path/Path(input_image).stem)
        metadata = kitti_util.read_kitti(
            os.path.join(gdino_output_path/Path(input_image).stem, "labels.txt"), ocd_data=ocd_metadata
        )
        print(f"Object Level Metadata: \n{metadata}")

        analytic_label = Path(analytics_path)/ (Path(input_image).stem + ".txt")
        print(f"Analytic label: {analytic_label}")
        with open(analytic_label, "w+") as fo:
            result, code_executor = generate_analytics(metadata, question, code_executor=code_executor)
            fo.write(str(result))

            #create table output for llm responses. 
            output_frame_responses["Frame ID"].append(Path(input_image).stem)
            output_frame_responses["LLM Output"].append(str(result))
    
        return result  # Return the path to the generated video file & llm response table
    
    except Exception as e:
        raise e
    finally:
        intermediate_paths = [
            model_output_path,
            inference_output_path,
            overlayn_image_path,
        ]
        exit_cleanup(intermediate_paths=intermediate_paths)

@hydra_runner(
    config_path=config_root,
    config_name="config.yaml",
    schema=GradioApp
)
def main(cfg: GradioApp):
    """Wrapper to instantiate a gradio app."""
    model_config = cfg.model
    global model_instances
    model_instances = {}
    global demo_configuration
    demo_configuration = cfg

    inputs = [
        gr.Image(label="Input Image", type="filepath"),
        gr.Textbox(label="Query")
    ]
    app_config = cfg.app

    outputs = [gr.Textbox(label="Output")]
    gr.Interface(
        fn=run_demo,
        inputs=inputs,
        outputs=outputs,
        title="Multi-Model Workflows").launch(
            server_port=app_config.server_port,
            server_name=app_config.server_name,
            debug=app_config.debug
        )


if __name__ == "__main__":
    main()
