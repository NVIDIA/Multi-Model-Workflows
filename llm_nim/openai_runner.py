import ast
import csv
import importlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from config.config_util import load_config

from openai import OpenAI
from utils.constants import NVCF_API, URL


def get_open_api_output(prompt, model_name):
    """Simple function to setup an OpenAPI client and get inferences."""
    client = OpenAI(
        base_url=URL,
        api_key=NVCF_API
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=0.1,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    return completion


def read_kitti(kitti_file: str, delimiter: str = " "):
    """Simple function to read a kitti file."""
    if not os.path.exists(kitti_file):
        raise FileNotFoundError(f"Kitti file not found at {kitti_file}.")
    object_list = []
    with open(kitti_file, "r") as kfile:
        csv_reader = csv.reader(kfile, delimiter=delimiter)
        for row in csv_reader:
            assert len(row) >= 15, "Atleast 15 elements are needed in the KITTI file."
            metadata = row[-15:]
            object_list.append(
                {
                    "class_name": " ".join(row[:-15]),
                    "bbox": [ast.literal_eval(coordinate) for coordinate in metadata[3:7]],
                    "confidence": float(metadata[-1])
                }
            )
    return object_list

def postprocessor(detections):
    # Initialize variables
    robot_bbox = None
    pallets_bbox = None

    # Iterate through each detection
    for detection in detections:
        if detection["class_name"] == "the robot":
            robot_bbox = detection["bbox"]
        elif detection["class_name"] == "pile of pallets":
            pallets_bbox = detection["bbox"]

    # Check if both robot and pallets are detected
    if robot_bbox is not None and pallets_bbox is not None:
        # Calculate the distance between the robot and pallets
        robot_center_x = (robot_bbox[0] + robot_bbox[2]) / 2
        robot_center_y = (robot_bbox[1] + robot_bbox[3]) / 2
        pallets_center_x = (pallets_bbox[0] + pallets_bbox[2]) / 2
        pallets_center_y = (pallets_bbox[1] + pallets_bbox[3]) / 2
        distance = ((robot_center_x - pallets_center_x) ** 2 + (robot_center_y - pallets_center_y) ** 2) ** 0.5

        # Check if the robot is close to the pallets
        if distance < 100:  # Adjust the threshold as needed
            return True

    return False


def extract_function(input_string):
    # Split the input string by newline characters
    lines = input_string.split("\n")

    # Find the line number where the function definition starts
    function_start_index = next(i for i, line in enumerate(lines) if line.strip().startswith("def"))

    # Get the initial indentation level of the function
    initial_indent = len(lines[function_start_index]) - len(lines[function_start_index].lstrip())

    # Iterate through lines to find the end of the function definition
    function_end_index = function_start_index + 1
    for line in lines[function_start_index + 1:]:
        current_indent = len(line) - len(line.lstrip())
        
        # Check if the line is outside the function definition
        if current_indent <= initial_indent and line.strip():
            break

        function_end_index += 1

    # Extract the function definition with docstring
    function_definition = "\n".join(lines[function_start_index:function_end_index])

    return function_definition


def infer(image, metadata, postprocessor_prompt, question):
    prompt = postprocessor_prompt.format(bbox_prompt=json.dumps(metadata), codellama_prompt=question)
    completion = get_open_api_output(prompt, "meta/codellama-70b")

    func = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            func = func + f"{chunk.choices[0].delta.content}"
    func = extract_function(func)

    _, temp_py_file = tempfile.mkstemp(suffix=".py")
    with open(temp_py_file, "w") as tmpfile:
        tmpfile.write(func)
    sys.path.append("/tmp")

    output_data = None
    traceback_str = None 
    try:
        module = importlib.import_module(os.path.basename(temp_py_file).strip(".py"))
        custom_function = getattr(module, "postprocessor")
        output_data = custom_function(json.loads(metadata))
    except Exception as e:
        traceback_str = traceback.format_exc()

    # output_data = postprocessor(metadata)
    sys.path.pop()
    return output_data, temp_py_file

def get_py_ouput(temp_py_file, metadata):
    sys.path.append("/tmp")

    output_data = None
    traceback_str = None 
    try:
        module = importlib.import_module(os.path.basename(temp_py_file).strip(".py"))
        custom_function = getattr(module, "postprocessor")
        output_data = custom_function(json.loads(metadata))
    except Exception as e:
        traceback_str = traceback.format_exc()

    sys.path.pop()
    return output_data

def infer_with_string_inp(image, metadata, postprocessor_prompt, question):
    prompt = """Write a single Python function called postprocessor that takes an string input representing objects in a video. Each line of the input contains the following information: frame_index, object_index, object_type, truncation, occlusion, alpha, x_min, y_min, x_max, y_max, length, width, height, x, y, z, rotation_y. The function should track objects across frames to avoid duplication. It should also answer a given question based on the tracked objects. 

Sample Input:
1 1 car 0.1 0.2 0.3 10.0 20.0 100.0 200.0 4.0 2.0 1.0 5.0 5.0 10.0 30.0
1 2 Pedestrian 0.2 0.3 0.4 50.0 50.0 150.0 250.0 1.0 1.0 1.0 2.0 3.0 5.0 15.0
2 1 Cyclist 0.1 0.2 0.3 12.0 22.0 102.0 202.0 4.0 2.0 1.0 5.0 5.0 10.0 30.0
2 2 Pedestrian 0.2 0.3 0.4 55.0 55.0 155.0 255.0 1.0 1.0 1.0 2.0 3.0 5.0 15.0

Question:
"Give me the count of the cars in the video?"

Function Signature:
def postprocessor(sample_input):
    # Function implementation goes here
"""
    completion = get_open_api_output(prompt, "meta/codellama-70b")

    func = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            func = func + f"{chunk.choices[0].delta.content}"
    func = extract_function(func)

    _, temp_py_file = tempfile.mkstemp(suffix=".py")
    with open(temp_py_file, "w") as tmpfile:
        tmpfile.write(func)
    sys.path.append("/tmp")

    output_data = None
    traceback_str = None 
    try:
        module = importlib.import_module(os.path.basename(temp_py_file).strip(".py"))
        custom_function = getattr(module, "postprocessor")
        output_data = custom_function(metadata)
    except Exception as e:
        traceback_str = traceback.format_exc()

    sys.path.pop()
    return output_data, temp_py_file
