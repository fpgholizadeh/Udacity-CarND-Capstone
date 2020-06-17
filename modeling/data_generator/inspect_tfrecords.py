import os
import numpy as np
import tensorflow as tf
from modeling.data_generator import parser


def extract_fn(data_record):
    features = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
    }
    sample = tf.parse_single_example(data_record, features)

    for k in sample:
        if isinstance(sample[k], tf.SparseTensor):
            if sample[k].dtype == tf.string:
                sample[k] = tf.sparse.to_dense(
                    sample[k], default_value='')
            else:
                sample[k] = tf.sparse.to_dense(
                    sample[k], default_value=0)

    # image = tf.decode_raw(sample["image/encoded"], tf.uint8)
    image = tf.io.decode_image(sample["image/encoded"], channels=3)
    file_name = tf.cast(sample["image/filename"], tf.string)
    height = tf.cast(sample["image/height"], tf.int32)
    width = tf.cast(sample["image/width"], tf.int32)
    image = tf.reshape(image, tf.stack([height, width, 3]))
    x1 = tf.cast(sample["image/object/bbox/xmin"], tf.float32)
    x2 = tf.cast(sample["image/object/bbox/xmax"], tf.float32)
    y1 = tf.cast(sample["image/object/bbox/ymin"], tf.float32)
    y2 = tf.cast(sample["image/object/bbox/ymax"], tf.float32)
    labels = tf.cast(sample["image/object/class/label"], tf.int64)

    boxes = tf.stack([x1, y1, x2, y2], axis=-1)
    return file_name, image, height, width, boxes, labels


# Initialize all tfrecord paths
dataset = tf.data.TFRecordDataset(
    "/Users/sardhendu/workspace/udacity-nd/ImageDataset/annotated_dataset/tf-record-train-0000-0000.record"
    # "/Users/sardhendu/workspace/udacity-nd/ImageDataset/annotated_dataset/simulator_dataset_rgb/tfrecords-00000-of-00001"
)
dataset = dataset.map(extract_fn)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
# print(next_element)


with tf.Session() as sess:
    # try:
    for i in range(0, 1152):
        print("Running iteration = ", i)
        out_path = "./data/validate_annotation"
        parser.make_dir(out_path)
        file_name, image, height, width, boxes, labels = sess.run(
            next_element
        )
        boxes = np.int32(boxes*[width, height, width, height])

        # labels = np.array([2]*len(boxes))
        labels -= 1
        annotated_image = parser.make_bbox_plots(
            image.copy(), boxes, labels, bounds_type="bbox")
        parser.write_image_rgb(annotated_image, os.path.join(
            out_path, "{}.jpg".format(file_name)))

    #
    # print("file_name: ", file_name)
    # print(image.shape, height, width, labels)
    #
    # print(image.shape)
    # print(x1)

# def _parse_image_function(example_proto):
#   # Parse the input tf.Example proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, image_feature_description)
#
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
# print(parsed_image_dataset)
