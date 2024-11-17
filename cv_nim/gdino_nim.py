import io
import json
import os
import sys
import uuid
import zipfile
import time
import requests
from pathlib import Path
from PIL import Image 
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed 
import traceback
nvai_polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
MAX_RETRIES = 5 # Max num of retries while polling
DELAY_BTW_RETRIES = 1 # adding 1s delay between each polls

class GDINONIM:

    def __init__(self, api_key, url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"):

        self.api_key = api_key 
        self.url = url 
        self.header_auth = f"Bearer {self.api_key}"

    def _upload_asset(self, input, description):
        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

        headers = {
            "Authorization": self.header_auth,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        s3_headers = {
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        }

        payload = {"contentType": "image/jpeg", "description": description}

        response = requests.post(assets_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()

        asset_url = response.json()["uploadUrl"]
        asset_id = response.json()["assetId"]

        # Convert image to jpeg before uploading 
        try:
            image = Image.open(str(input)).convert("RGB")
            buf = io.BytesIO() #temporary buffer to save image
            image.save(buf, format="JPEG")
        except Exception as e:
            print("An error occurred:")
            print(e)
            print(traceback.format_exc())

        try:
            response = requests.put(
                asset_url,
                data=buf.getvalue(),
                headers=s3_headers,
                timeout=300,
            )
        except Exception as e:
            print("An error occurred:")
            print(e)
            print(traceback.format_exc())

        response.raise_for_status()
        return uuid.UUID(asset_id)

    def infer(self, image_path, prompt, output_folder=None):
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        asset_id = self._upload_asset(image_path, "Input Image")

        inputs = {"image": f"{asset_id}", "render_label": False}
        inputs = { "model": "Grounding-Dino",
                    "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                            "type": "text",
                            "text": f"{prompt}"
                            },
                            {
                            "type": "media_url",
                            "media_url": {
                            "url": f"data:image/jpeg;asset_id,{asset_id}"
                            }
                        }
                        ]
                    }
                    ],
                    "threshold": 0.3
                }
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

        if response.status_code == 200: # evaluation complete, output video ready
            with open(zip_path, "wb") as out:
                out.write(response.content)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_path.parent/zip_path.stem)

        elif response.status_code == 202: # pending evaluation
            print("Pending evaluation ...")
            nvcf_reqid = response.headers['NVCF-REQID']
            nvai_polling_url = nvai_polling_url + nvcf_reqid

            # Polling to check if the response is ready
            while( MAX_RETRIES ):
                print(f'Polling ...')
                headers_polling = { "accept": "application/json", "Authorization": header_auth }
                response_polling = requests.get(nvai_polling_url, headers=headers_polling)
                if response_polling.status_code == 202: # evaluation pending
                    print('Result is not yet ready.')
                    MAX_RETRIES -= 1
                    time.sleep(DELAY_BTW_RETRIES)
                    continue
                elif response_polling.status_code == 200: # evaluation complete, output video ready
                    print('Result ready!')
                    with open(zip_path, "wb") as out:
                        out.write(response_polling.content)
                    break
                else:
                    print(f"Unexpected response status: {response_polling.status_code}")

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_path.parent/zip_path.stem)

        zip_path.unlink() #delete temp zip 

    def batch_infer(self, input_folder, prompt, output_folder, workers=16):
        input_folder = Path(input_folder)

        image_files = []
        for image_path in input_folder.iterdir():
            if image_path.suffix  in ['.png', '.jpeg', '.jpg']:
                image_files.append(image_path)

        with ThreadPoolExecutor(max_workers = workers) as executor: 
            futures = [executor.submit(self.infer, image_path.resolve(), prompt, output_folder) for image_path in image_files]
            #Wait for all jobs to complete. 
            for future in tqdm(as_completed(futures), total=len(futures)):
                pass

    def write_output_as_kitti_file(self, data, output_file_path):
        # Process the bounding boxes and write to KITTI format
        with open(output_file_path, 'w') as file:
            for choice in data["choices"]:
                message_content = choice["message"]["content"]
                frame_no = message_content["frameNo"]
                
                # Iterate through each bounding box
                for box in message_content["boundingBoxes"]:
                    phrase = box["phrase"].strip("[]").replace("'", "").strip()  # Clean up phrase
                    for bbox, confidence in zip(box["bboxes"], box["confidence"]):
                        # KITTI format: Class, 0, 0, 0, xmin, ymin, xmax, ymax, 0, 0, 0, 0, 0, 0, 0, confidence
                        xmin, ymin, xmax, ymax = bbox
                        kitti_line = f"{phrase} 0 0 0 {xmin} {ymin} {xmax} {ymax} 0 0 0 0 0 0 0 {confidence}\n"
                        file.write(kitti_line)

        print(f"Bounding boxes have been written to {output_file_path} in KITTI format.")


    def parse_output(self, results_path, sort=True):
        ocd_response_file = list(Path(results_path).glob('*.response'))[0] #get response file 
        with ocd_response_file.open('r') as file:
            data = json.load(file)

            self.write_output_as_kitti_file(data, f"{results_path}/labels.txt")
