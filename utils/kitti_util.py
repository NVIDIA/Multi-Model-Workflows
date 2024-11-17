import ast
import os
import csv
import cv2
from shapely.geometry import Polygon, box 
from pathlib import Path 

def _polygon_intersection(polygon_dict, bbox):
    polygon_points = []
    for key in polygon_dict.keys():
        if "x" in key:
            x = polygon_dict[key]
            y = polygon_dict[key.replace("x", "y")]
        point = (x, y)
        polygon_points.append(point)
    
    polygon = Polygon(polygon_points)
    bbox = box(*bbox)

    return polygon.intersects(bbox)

def read_kitti(kitti_file, ocd_data=None):
    """Function to read the KITTI dataset."""
    if not os.path.exists(kitti_file):
        raise FileNotFoundError(f"Kitti file not found at {kitti_file}.")
    object_list = []
    with open(kitti_file, "r") as kfile:
        csv_reader = csv.reader(kfile, delimiter=" ")
        for row in csv_reader:
            assert len(row) >= 15, "Atleast 15 elements are needed in the KITTI file."
            metadata = row[-15:]
            object_bbox = [ast.literal_eval(coordinate) for coordinate in metadata[3:7]]
            object_list.append(
                {
                    "class_name": " ".join(row[:-15]),
                    "bbox": object_bbox,
                    "confidence": float(metadata[-1])
                }
            )

            #if ocd data then correlate it with the object and add to metadata 
            if ocd_data:
                object_str = ""
                """Add ocd field to metadata"""
                for ocd in ocd_data["metadata"]:
                    if _polygon_intersection(ocd["polygon"], object_bbox):
                        object_str = object_str + " " + (ocd["label"])
                object_list[-1]["object_text"] = object_str

    return object_list

def overlay_labels_on_images(images_dir: str, labels_dir: str, output_dir: str, detection_dir:str=None):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image files
    image_files = sorted(os.listdir(images_dir))
    
    # Iterate over image files
    for image_file in image_files:
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)

        #load detections
        if detection_dir:
            kitti_file = os.path.join(detection_dir, Path(image_file).with_suffix(".txt"))
            detection_metadata = read_kitti(kitti_file, ocd_data=None)
            for object in detection_metadata:
                bbox = object["bbox"]
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)

        
        # Load label file
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            label = f.read().strip()
        
        # Overlay label onto image
        cv2.putText(image, label, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        
        # Save annotated image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)
