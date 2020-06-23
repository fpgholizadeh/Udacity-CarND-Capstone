import tensorflow as tf
import numpy as np
from collections import Counter
from modeling.data_generator import parser
from modeling.data_generator.parser import label_id_map


class TrafficLightClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = "./ros/model_weights.pb"
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
            self.image_tensor = self.detection_graph.get_tensor_by_name(
                "image_tensor:0"
            )
            self.d_boxes = self.detection_graph.get_tensor_by_name(
                "detection_boxes:0")
            self.d_scores = self.detection_graph.get_tensor_by_name(
                "detection_scores:0"
            )
            self.d_classes = self.detection_graph.get_tensor_by_name(
                "detection_classes:0"
            )
            self.num_d = self.detection_graph.get_tensor_by_name(
                "num_detections:0")
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img, threshold=0.5, debug=False):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            # img = cv2.resize(img, (300, 300))
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded},
            )

            # TODO: A hack
            orig_img_shape = img.shape
            resized_shape = 300.0
            scale_y = orig_img_shape[0] / resized_shape
            scale_x = orig_img_shape[1] / resized_shape

            boxes = np.int32(boxes * resized_shape)[0]
            scores = np.float32(scores[0])
            classes = np.int32(classes)[0] - 1
            boxes = np.column_stack(
                (
                    boxes[:, 1] * scale_x,
                    boxes[:, 0] * scale_y,
                    boxes[:, 3] * scale_x,
                    boxes[:, 2] * scale_y,
                )
            )
        return self.get_traffic_light_colors(
            img, boxes, scores, classes, threshold, debug
        )

    def get_traffic_light_colors(
        self, img, boxes, class_scores, classes, threshold=0.5, debug=False
    ):
        scores_idx = np.where(class_scores > threshold)
        # class_scores = class_scores[scores_idx]
        classes = classes[scores_idx]
        boxes = boxes[scores_idx]
        class_counter = Counter(classes)

        traffic_light_color = []
        max_cnt = 0
        for class_, cnt in class_counter.items():
            if cnt >= max_cnt:
                traffic_light_color += [label_id_map[class_]]

        if debug:
            img = self.make_plots(img, boxes, classes)

        if "Red" in label_id_map:
            return "Red", img
        else:
            return traffic_light_color[0] if len(traffic_light_color) > 0 else "pass", img

    def make_plots(self, img, boxes, classes):
        img = parser.make_bbox_plots(
            img.copy(), boxes, classes, bounds_type="bbox")
        return img


def center_crop(image, size=(300, 300)):
    h, w = image.shape[0:-1]
    center = [h // 2, w // 2]
    spread_h = size[0] // 2
    spread_w = size[1] // 2
    box = [
        center[0] - spread_h,
        center[0] + spread_h,
        center[1] - spread_w,
        center[1] + spread_w,
    ]
    cropped_img = image[box[0]: box[1], box[2]: box[3]]
    print("cropped_img: ", cropped_img.shape)
    return cropped_img


if __name__ == "__main__":
    import os

    debug = True
    obj_classifier = TrafficLightClassifier()
    prefix = "path_to_data/images_real"
    out_path = "path_to_output/000_real_world"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for num_, path_ in enumerate(os.listdir(prefix)):
        if path_.endswith("jpg") or path_.endswith("png"):
            full_path = os.path.join(prefix, path_)
            image = parser.read_image_rgb(full_path)

            color, img = obj_classifier.get_classification(image, debug=debug)
            print("image_name", path_, "color: ", color)

            parser.write_image_rgb(
                img, os.path.join(out_path,  path_))
            if num_ == 20:
                break
