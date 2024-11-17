import os
import io 
import sys
import uuid
import json 
import zipfile
import logging 
from pathlib import Path
from PIL import Image 
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed 

import requests

class OCDNIM:

    def __init__(self, api_key, url="https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"):

        self.api_key = api_key 
        self.url = url 
        self.header_auth = f"Bearer {self.api_key}"

    def _upload_asset(self, image_path, description):
        """
        Uploads an asset to the NVCF API.
        :param image_path: The image path 
        :param description: A description of the asset

        """

        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

        headers = {
            "Authorization": self.header_auth,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        s3_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": f"image/jpeg",
        }

        payload = {"contentType": f"image/jpeg", "description": description}

        response = requests.post(assets_url, headers=headers, json=payload, timeout=30)

        response.raise_for_status()

        asset_url = response.json()["uploadUrl"]
        asset_id = response.json()["assetId"]

        #Convert image to jpeg before uploading 
        image = Image.open(image_path).convert("RGB")
        buf = io.BytesIO() #temporary buffer to save image
        image.save(buf, format="JPEG")

        #upload image 
        response = requests.put(
            asset_url,
            data=buf.getvalue(),
            headers=s3_headers,
            timeout=300,
        )

        response.raise_for_status()
        return uuid.UUID(asset_id)

    def infer(self, image_path, output_folder=None):
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        asset_id = self._upload_asset(image_path, "Input Image")

        inputs = {"image": f"{asset_id}", "render_label": False}
        asset_list = f"{asset_id}"

        headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_list,
        "NVCF-FUNCTION-ASSET-IDS": asset_list,
        "Authorization": self.header_auth,
        }

        response = requests.post(self.url, headers=headers, json=inputs)

        if output_folder:
            zip_path = Path(output_folder) / (Path(image_path).stem + ".zip")
        else:
            zip_path = Path(image_path).with_suffix(".zip")

        with open(zip_path, "wb") as out:
            out.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(zip_path.parent/zip_path.stem)
        
        zip_path.unlink() #delete temp zip 
   
    def batch_infer(self, input_folder, output_folder, workers=16):
        input_folder = Path(input_folder)

        image_files = []
        for image_path in input_folder.iterdir():
            if image_path.suffix  in ['.png', '.jpeg', '.jpg']:
                image_files.append(image_path)

        with ThreadPoolExecutor(max_workers = workers) as executor: 
            futures = [executor.submit(self.infer, image_path.resolve(), output_folder) for image_path in image_files]
            #Wait for all jobs to complete. 
            logging.info("OCD Inference")
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass

    # Function to calculate centroid
    def _calculate_centroid(self, polygon):
        x_coords = [polygon[key] for key in polygon.keys() if key.startswith('x')]
        y_coords = [polygon[key] for key in polygon.keys() if key.startswith('y')]
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        return centroid_x, centroid_y


    def parse_output(self, results_path, sort=True):
        ocd_response_file = list(Path(results_path).glob('*.response'))[0] #get response file 
        with ocd_response_file.open('r') as file:
            data = json.load(file)

        if sort:
            """Sort the results based on the polygons from topleft to bottoms right."""
            for entry in data["metadata"]: #calculate centroid for each polygon 
                polygon = entry["polygon"]
                centroid = self._calculate_centroid(polygon)
                entry['centroid'] = centroid

            data['metadata'].sort(key=lambda x: (x['centroid'][1], x['centroid'][0])) #sort based on centroid 
            for entry in data['metadata']: #remove centroid 
                del entry['centroid']

        return data 
