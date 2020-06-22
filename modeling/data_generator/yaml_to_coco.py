import json
import os
import yaml
import numpy as np
from modeling.data_generator import pycococreatortools
from modeling.data_generator.parser import read_image_rgb

# go through each image
output_path = "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb2/annotation.json"
image_dir = "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb2/images"
annotation_file = "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb2/sim_data_annotations.yaml"
coco_output = {"images": [], "annotations": []}
coco_output["categories"] = [
    {
        "supercategory": "none",
        "id": 4,
        "name": "Unknown"
    },
    {
        "supercategory": "none",
        "id": 3,
        "name": "Green"
    },
    {
        "supercategory": "none",
        "id": 2,
        "name": "Yellow"
    },
    {
        "supercategory": "none",
        "id": 1,
        "name": "Red"
    }
]

with open(annotation_file, "r") as file:
    annotations = yaml.load(file)
    annotation_id = 0
    for annot in annotations:
        image_id = os.path.basename(annot["filename"])
        # if image_id != "left0021.jpg":
        #     continue
        image_filename = os.path.join(image_dir, image_id)

        image = read_image_rgb(image_filename)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.shape
        )

        has_annotation = False
        # go through each associated annotation
        for bbox_meta in annot["annotations"]:
            print(bbox_meta)

            has_annotation = True
            if bbox_meta["class"] == "Red":
                class_id = 1
            elif bbox_meta["class"] == "Yellow":
                class_id = 2
            elif bbox_meta["class"] == "Green":
                class_id = 3
            else:
                class_id = 4

            category_info = {"id": class_id, "is_crowd": 0}

            x1, y1, w, h = (
                int(bbox_meta["xmin"]),
                int(bbox_meta["ymin"]),
                int(bbox_meta["x_width"]),
                int(bbox_meta["y_height"]),
            )

            annotation_info = pycococreatortools.create_annotation_info(
                annotation_id,
                image_id,
                category_info,
                binary_mask=None,
                image_size=image.shape[0:-1],
                tolerance=2,
                bounding_box=[x1, y1, w, h],
            )
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            annotation_id += 1

        if has_annotation:
            coco_output["images"].append(image_info)
        # print(annotation_info)
        #
        # break

print("annotation_info")
print(coco_output)
json_obj = json.dumps(coco_output, indent=4)
with open(output_path, "w") as outfile:
    outfile.write(json_obj)
