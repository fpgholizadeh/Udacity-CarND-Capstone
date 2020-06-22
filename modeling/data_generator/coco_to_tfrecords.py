"""
Usage:

# Create train data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/train_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/train.record

# Create test data:
python generate_tfrecord.py --label=<LABEL> --csv_input=<PATH_TO_ANNOTATIONS_FOLDER>/test_labels.csv  --output_path=<PATH_TO_ANNOTATIONS_FOLDER>/test.record
"""

from __future__ import division
from collections import namedtuple
from PIL import Image
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow as tf
import sys
from collections import defaultdict
sys.path.append("../../models/research")


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# TO-DO replace this with label map
# for multiple labels add more else if statements
# def class_text_to_int(row_label):
#     if row_label == FLAGS.label:  # 'ship':
#         return 1
#     # comment upper if statement and uncomment these statements for multiple labelling
#     # if row_label == FLAGS.label0:
#     #   return 1
#     # elif row_label == FLAGS.label1:
#     #   return 0
#     else:
#         None


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]


def create_tf_example(image_meta, annot_list, img_dir, category_index):
    with tf.gfile.GFile(os.path.join(img_dir, "{}".format(image_meta["file_name"])), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    if width < 800 or height < 600:
        print(width, height)

    filename = image_meta["file_name"].encode("utf8")

    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in enumerate(annot_list):
        xmin, ymin, w, h = row["bbox"]
        xmax = xmin+w
        ymax = ymin+h
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes.append(row["category_id"])
        classes_text.append(
            category_index[row["category_id"]]['name'].encode('utf8'))
    #
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": int64_feature(height),
                "image/width": int64_feature(width),
                "image/filename": bytes_feature(filename),
                "image/source_id": bytes_feature(filename),
                "image/encoded": bytes_feature(encoded_jpg),
                "image/format": bytes_feature("jpeg".encode("utf8")),
                "image/object/bbox/xmin": float_list_feature(xmins),
                "image/object/bbox/xmax": float_list_feature(xmaxs),
                "image/object/bbox/ymin": float_list_feature(ymins),
                "image/object/bbox/ymax": float_list_feature(ymaxs),
                "image/object/class/text": bytes_list_feature(classes_text),
                "image/object/class/label": int64_list_feature(classes),
            }
        )
    )
    return tf_example


def group_annotation_per_image(annotations):
    annot_dict_per_image = defaultdict(list)
    for annot in annotations:
        annot_dict_per_image[annot["image_id"]].append(annot)
    return annot_dict_per_image


def image_meta_dict(image_meta):
    image_dict = defaultdict()
    for img_meta in image_meta:
        image_dict[img_meta["id"]] = img_meta
    return image_dict


if __name__ == "__main__":
    import numpy as np
    from modeling.data_generator import parser
    train_output_path = "path_to/udacity-nd/ImageDataset/annotated_dataset/tf-record-train-0000-0000.record"
    eval_output_path = "path_to/udacity-nd/ImageDataset/annotated_dataset/tf-record-eval-0000-0000.record"
    image_dirs = [
        "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb2/images",
        "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb/images",
    ]

    annotation_dir = [
        "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb2/annotation.json",
        "path_to/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb/annotation.json",
    ]

    train_writer = tf.python_io.TFRecordWriter(train_output_path)
    eval_writer = tf.python_io.TFRecordWriter(eval_output_path)
    counter = np.arange(0, 1000)
    np.random.shuffle(counter)
    test_idx = set(counter[0:152])

    running_idx = 0
    for im_dirs, annot_path in zip(image_dirs, annotation_dir):
        file_names = os.listdir(im_dirs)
        a = parser.read_json(annot_path)

        images = a["images"]
        annotation = a["annotations"]
        category_index = a["categories"]

        annotation_group = group_annotation_per_image(annotation)
        image_meta = image_meta_dict(images)
        assert(len(annotation_group) == len(image_meta))

        for img_id, img_meta in image_meta.items():
            annot_list = annotation_group[img_id]
            tf_example = create_tf_example(
                img_meta, annot_list, im_dirs, category_index)

            if running_idx in test_idx:
                eval_writer.write(tf_example.SerializeToString())
            else:
                train_writer.write(tf_example.SerializeToString())

            running_idx += 1

    train_writer.close()
    eval_writer.close()
    print("Successfully created the TFRecords: {}".format(train_output_path))
    print("Successfully created the TFRecords: {}".format(eval_output_path))

    # print("all_paths: ", all_paths)
    # tf.app.run()
