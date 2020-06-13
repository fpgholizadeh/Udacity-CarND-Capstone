import os
import cv2
import numpy as np
import pandas as pd
import json
from pprint import pprint
from collections import defaultdict

low_res_path = "/Users/sardhendu/workspace/udacity-nd/ImageDataset/dataset_lowres"


file_paths = [os.path.join(low_res_path, i) for i in os.listdir(low_res_path)]
color_dict = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Yellow": (128, 128, 128),
    "Unknown": (255, 255, 255),
}
label_id_map = {0: "Red", 1: "Yellow", 2: "Green", 3: "Unknown"}


def read_image_rgb(img_path):
    image = cv2.imread(img_path)
    image = np.array(image, dtype=np.uint8)[::, :-1]
    return image


def write_image_rgb(img, img_path):
    cv2.imwrite(img_path, img)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_bbox_plots(image, list_of_object_bounds, category_ids):
    assert len(list_of_object_bounds) == len(category_ids)
    for bbox, cat_id in zip(list_of_object_bounds, category_ids):
        cv2.polylines(
            image,
            [np.array(bbox, dtype=np.int32)],
            True,
            color_dict[label_id_map[cat_id]],
            1,
        )
    return image


def validate_data(annotation_dict, how_many):
    out_path = "./data/validate_annotation"
    make_dir(out_path)
    for num, (id, annot_dict) in enumerate(annotation_dict.items()):
        image = read_image_rgb(annot_dict["path"])
        annotated_image = make_bbox_plots(
            image.copy(), annot_dict["bbox"], annot_dict["category_id"]
        )
        write_image_rgb(annotated_image, os.path.join(
            out_path, "{}.jpg".format(id)))

        if how_many == num:
            break


def parse_get_bbox(coco_like_annotation_path, img_dir):
    """
    Args:
        coco_like_annotation_path:  .json path to coco dataset
        img_dir:                    path to_image_storage

    Returns:        list of dictionaries
             [
                {
                    img_id: {
                        path: path_to_the_image
                        bbox: list_of_object_vertices
                        label: list_of_object_label
                    }
                }
            ]
    """
    with open(coco_like_annotation_path, "r") as file:
        annotation_dict = json.load(file)

    annot_dict_per_image = defaultdict(lambda: defaultdict(list))
    for annotation in annotation_dict["annotations"]:
        x1, y1, w, h = annotation["bbox"]
        x2 = x1 + w
        y2 = y1 + h
        label = annotation["category_id"] - 1  # Substract 1 to make it 0, n-1
        annot_dict_per_image[annotation["image_id"]]["bbox"].append(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
        )
        annot_dict_per_image[annotation["image_id"]
                             ]["category_id"].append(label)

    for image_meta in annotation_dict["images"]:
        path = os.path.join(img_dir, image_meta["file_name"])
        annot_dict_per_image[image_meta["id"]]["path"] = path

    print("Total length of metadata = ", len(annot_dict_per_image))
    validate_data(annot_dict_per_image, 10)


def run(data_json_path):
    img_num = 0
    img_path = file_paths[img_num]
    out_path = "./data/check_images"
    make_dir(out_path)
    out_path = os.path.join(out_path, str(img_num) + ".jpg")
    image = read_image_rgb(img_path)
    print(image.shape)
    image = image[0:200, 120:500]
    print(image.shape)
    write_image_rgb(image, out_path)


if __name__ == "__main__":
    prefix_path = "/Users/sardhendu/workspace/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb/red"
    data_json_path = os.path.join(prefix_path, "annotation.json")
    img_path = os.path.join(prefix_path, "images")
    # run(data_json_path)
    parse_get_bbox(data_json_path, img_path)
