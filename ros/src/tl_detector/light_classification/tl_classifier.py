from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
from collections import Counter

label_id_map = {0: "Unknown", 1: "Red", 2: "Yellow", 3: "Green", }


class TLClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = "/home/student/CarND-Capstone/ros/model_weights.pb"
        self.threshold = 0.5
        self.debug = False

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

    def get_classification(self, image):
        image = image[..., ::-1]
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            # img = cv2.resize(img, (300, 300))
            img_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded},
            )

            # TODO: A hack
            orig_img_shape = image.shape
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
        return self.get_traffic_light_colors(image, boxes, scores, classes)

    def get_traffic_light_colors(self, img, boxes, class_scores, classes):
        scores_idx = np.where(class_scores > self.threshold)
        # class_scores = class_scores[scores_idx]
        classes = classes[scores_idx]
        boxes = boxes[scores_idx]
        class_counter = Counter(classes)

        traffic_light_color = []
        max_cnt = 0
        for class_, cnt in class_counter.items():
            if cnt >= max_cnt:
                traffic_light_color += [label_id_map[class_]]

        if self.debug:
            img = self.make_plots(img, boxes, classes)

        if len(traffic_light_color) == 0:
            return TrafficLight.UNKNOWN
        elif "Red" in traffic_light_color:
            return TrafficLight.RED
        else:
            if traffic_light_color[0] == "Green":
                return TrafficLight.GREEN
            else:
                return TrafficLight.YELLOW
