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


def run_demo(input_video_path, question):
    """Run the gradio demo."""
    frames_dir = tempfile.mkdtemp()
    model_output_path = tempfile.mkdtemp()
    output_video_path = tempfile.mkdtemp()
    inference_output_path = tempfile.mkdtemp()
    overlayn_image_path = tempfile.mkdtemp()
    
    # fed into Grounding DINO.
    noun_chunks = extract_noun_chunks(question)
    logging.info(f"Noun Chunks: {noun_chunks}")

    try:
        ffmpeg_command = f"ffmpeg -i {input_video_path} -vf \"fps={SAMPLING_FPS}\" {frames_dir}/frame_%05d.png"
        assert execute_command(ffmpeg_command), (
            "Video wasn't serialized to run inference."
        )

        #Inference Grounding Dino
        gdino_nim = GDINONIM(NVCF_API)
        gdino_output_path = Path(model_output_path) / "gdino_inference"
        gdino_nim.batch_infer(frames_dir, noun_chunks, gdino_output_path)

        #Inference OCD
        ocd_nim = OCDNIM(NVCF_API)
        ocd_output_path = Path(model_output_path) / "ocd_inference"
        ocd_nim.batch_infer(frames_dir, ocd_output_path, workers=16)
        analytics_path = os.path.join(model_output_path, "analytics")
        os.makedirs(analytics_path, exist_ok=True)
        annotations_path = os.path.join(model_output_path, "inference/labels")
        frame_files = glob.iglob(os.path.join(annotations_path, "*.txt"))
        code_executor = Executor()
        output_frame_responses = {"Frame ID": [], "LLM Output": []}
        for frame in tqdm(frame_files):
            #Read OCD/OCR metadata 
            ocd_metadata = ocd_nim.parse_output(ocd_output_path/Path(frame).stem)

            #Grounding dino metadata
            metadata = kitti_util.read_kitti(
                os.path.join(annotations_path, frame), ocd_data=ocd_metadata
            )
            logging.debug(f"Object Level Metadata: \n{metadata}")

            analytic_label = Path(analytics_path)/ (Path(frame).stem + ".txt")
            print(f"Analytic label: {analytic_label}")
            with open(analytic_label, "w+") as fo:
                result, code_executor = generate_analytics(metadata, question, code_executor=code_executor)
                fo.write(str(result))

                #create table output for llm responses. 
                output_frame_responses["Frame ID"].append(Path(frame).stem)
                output_frame_responses["LLM Output"].append(str(result))
        
        # Overlay the annotation on the image
        kitti_util.overlay_labels_on_images(frames_dir, analytics_path, overlayn_image_path, detection_dir=Path(model_output_path)/"inference/labels")

        # Concatenate command for ffmpeg
        output_video_file = f"{output_video_path}/gradio_output_video.mp4"
        ffmpeg_command = (
            f"ffmpeg -y -framerate {SAMPLING_FPS} -i {overlayn_image_path}/frame_%05d.png "  # Input frames
            f"-c:v libx264 -pix_fmt yuv420p {output_video_file}"  # Output video codec and format
        )
        
        # Execute ffmpeg command
        execute_command(ffmpeg_command)
        return output_video_file, pd.DataFrame(output_frame_responses)  # Return the path to the generated video file & llm response table
    
    except Exception as e:
        raise e
    finally:
        intermediate_paths = [
            frames_dir,
            model_output_path,
            inference_output_path,
            overlayn_image_path,
        ]
        exit_cleanup(intermediate_paths=intermediate_paths)


# def pull_and_cache_models(model_instance_config):
#     """Create a model instance config."""
#     model_instance = ModelInstance(
#         **dict(model_instance_config)
#     )
#     if not model_instance.local_model_path: #If local model, then skip NGC pull 
#         model_metadata = model_instance.retrieve_model_metadata()
#         logger.info(model_metadata)
#         model_instance.download_model()
#     return model_instance


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
    # for instance_config in model_config:
    #     model_instances[instance_config.name] = pull_and_cache_models(instance_config)

    inputs = [
        gr.Video(label="Input Video"),
        gr.Textbox(label="Query")
    ]
    app_config = cfg.app

    outputs = [
        gr.Video(), 
        gr.Dataframe(headers=["Frame ID", "LLM Output"])
    ]
    gr.Interface(
        fn=run_demo,
        inputs=inputs,
        outputs=outputs,
        title="Frame Annotation Video Renderer").launch(
            server_port=app_config.server_port,
            server_name=app_config.server_name,
            debug=app_config.debug
        )


if __name__ == "__main__":
    main()
