import cv2
import keras
import numpy as np
from six import raise_from
from collections import defaultdict
from rough import anchors as anchor_utils, augment


def check_line(x1, y1, x2, y2, class_names):
    if (x1, y1, x2, y2) == ("", "", "", ""):
        return 0
    elif len(class_names) == 0:
        return 1
    else:
        return 2


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


class AnnotationLoader:
    def __init__(self, class_name_to_label, num_classes):
        self.class_name_to_label = class_name_to_label
        self.num_classes = num_classes

    def load_annotations(self, list_of_classes, list_of_polygons):
        """ Load annotations for an image_index.
        """
        assert len(list_of_classes) == len(
            list_of_polygons
        ), "len(list_of_classes) = {} should equal len(list_of_polygons = {}".format(
            len(list_of_classes), len(list_of_polygons)
        )
        annotations = defaultdict(list)

        is_annotation = 0
        for idx, (class_arr, poly_arr) in enumerate(
            zip(list_of_classes, list_of_polygons)
        ):
            is_annotation = 1
            labels = [self.class_name_to_label[cls] for cls in class_arr]
            label_arr = np.zeros(self.num_classes)
            label_arr[labels] = 1

            poly_arr = np.array(poly_arr, dtype=np.int32)
            x1 = max(0, min(poly_arr[:, 1]))
            y1 = max(0, min(poly_arr[:, 0]))
            x2 = max(max(poly_arr[:, 1]), 0)
            y2 = max(max(poly_arr[:, 0]), 0)

            annotations["labels"].append(label_arr)
            annotations["bboxes"].append(
                [float(x1), float(y1), float(x2), float(y2), ]
            )

        if is_annotation == 0:
            # print(annotations)
            annotations["labels"] = []
            annotations["bboxes"] = []

        annotations["labels"] = np.array(
            annotations["labels"], dtype=np.float64)
        annotations["bboxes"] = np.array(
            annotations["bboxes"], dtype=np.float64)

        return annotations

    def filter_annotations(self, annotations, img_shape, image_name):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        if len(np.ravel(annotations["labels"])) == 0:
            return annotations, set()
        # print('annotations[----boxes]', annotations['bboxes'])
        # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
        idxx = np.where(annotations["bboxes"] < 0)
        annotations["bboxes"][idxx] = 0

        idxx_max = np.where(annotations["bboxes"][:, 2] > img_shape[1])
        annotations["bboxes"][list(idxx_max), np.tile(
            2, len(idxx_max))] = img_shape[1]

        idxx_max = np.where(annotations["bboxes"][:, 3] > img_shape[0])[0]
        annotations["bboxes"][list(idxx_max), np.tile(
            3, len(idxx_max))] = img_shape[0]

        # print('annotations[bboxes]: ', annotations['bboxes'])
        invalid_indices = np.where(
            (annotations["bboxes"][:, 2] <= annotations["bboxes"][:, 0])
            | (annotations["bboxes"][:, 3] <= annotations["bboxes"][:, 1])
            | (annotations["bboxes"][:, 2] > img_shape[1])
            | (annotations["bboxes"][:, 3] > img_shape[0])
        )[0]
        # delete invalid indices
        if len(invalid_indices):
            # warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
            #     image_name,
            #     img_shape,
            #     annotations['bboxes'][invalid_indices, :]
            # ))
            for k in annotations.keys():
                annotations[k] = np.delete(
                    annotations[k], invalid_indices, axis=0)

        return annotations, set(invalid_indices)


class GroundTruthAnnotation:
    def __init__(
        self,
        image_shape,
        pyramid_levels,
        anchor_params,
        num_classes,
        hard_positive_threshold,
        hard_negative_threshold,
        get_transform_bboxes=True,
    ):

        self.image_shape = image_shape
        self.get_transform_bboxes = get_transform_bboxes
        self.num_classes = num_classes
        self.hard_positive_threshold = hard_positive_threshold
        self.hard_negative_threshold = hard_negative_threshold

        self.anchors = anchor_utils.anchors_for_shape(
            image_shape, pyramid_levels, anchor_params
        )
        self.anchors = anchor_utils.bound_boxes(
            self.anchors,
            min_xy=(0, 0),
            max_xy=(self.image_shape[0], self.image_shape[1]),
        )

    def anchor_targets_bbox(self, image_batch, annotations_batch):
        assert len(image_batch) == len(
            annotations_batch
        ), "len(image_batch) = {} should equal len(annotations_batch) = {}".format(
            len(image_batch), len(annotations_batch)
        )
        assert (
            len(annotations_batch) > 0
        ), "No data received to compute anchor targets for."
        for annotations in annotations_batch:
            assert "bboxes" in annotations, "Annotations should contain bboxes."
            assert "labels" in annotations, "Annotations should contain labels."

        batch_size = len(image_batch)

        regression_batch_transformed = np.zeros(
            (batch_size, self.anchors.shape[0], 4 + 1), dtype=keras.backend.floatx()
        )
        labels_batch = np.zeros(
            (batch_size, self.anchors.shape[0], self.num_classes + 1),
            dtype=keras.backend.floatx(),
        )

        # compute labels and regression targets
        for index, (image, annotations) in enumerate(
            zip(image_batch, annotations_batch)
        ):

            if annotations["bboxes"].shape[0] > 0:
                (
                    positive_indices,
                    ignore_indices,
                    argmax_overlaps_inds,
                ) = compute_gt_annotations(
                    self.anchors,
                    annotations["bboxes"],
                    self.hard_negative_threshold,
                    self.hard_positive_threshold,
                )
                # print('positive_indices total : ', positive_indices)
                annotation_label_id = np.argmax(annotations["labels"], axis=1)
                overlap_label_ids = annotation_label_id[argmax_overlaps_inds]

                positive_anchor_indices = positive_indices.reshape(-1, 1)
                positive_label_indices = overlap_label_ids[positive_indices]

                # Mark the ignore index with value -1 to be avoided in training
                labels_batch[index, ignore_indices, -1] = -1
                # We add 1 because 0 class here means background, and our labels go from (0, n-1)
                labels_batch[index, positive_indices, -
                             1] = positive_label_indices + 1
                # So we add 1 to avoid confusion between 1st class and the background

                # Regression
                regression_batch_transformed[index, ignore_indices, -1] = -1
                regression_batch_transformed[index, positive_indices, -1] = 1

                pp = np.array(annotations["labels"])[
                    argmax_overlaps_inds[positive_indices]
                ]
                pp = np.expand_dims(pp, axis=1)
                labels_batch[index, positive_anchor_indices, :-1] = pp

                regression_batch_transformed[
                    index, :, :-1
                ] = anchor_utils.bbox_transform(
                    self.anchors, annotations["bboxes"][argmax_overlaps_inds, :]
                )

            if image.shape:
                anchors_centers = np.vstack(
                    [
                        (self.anchors[:, 0] + self.anchors[:, 2]) / 2,
                        (self.anchors[:, 1] + self.anchors[:, 3]) / 2,
                    ]
                ).T
                indices = np.logical_or(
                    anchors_centers[:, 0] >= image.shape[1],
                    anchors_centers[:, 1] >= image.shape[0],
                )

                labels_batch[index, indices, -1] = -1
                regression_batch_transformed[index, indices, -1] = -1

        return regression_batch_transformed, labels_batch

    def compute_targets(self, image_batch, annotations_batch):
        """ Compute target outputs for the network using image and their annotations.
        """
        # get the max image shape
        # print('[Generator] compute_targets ........')
        regression_batch_transformed, labels_batch = self.anchor_targets_bbox(
            image_batch, annotations_batch
        )

        return list([regression_batch_transformed, labels_batch])

    # def plot_anchors(self, image, plot_name, annotations, dump_dir):
    #     """ Plot positive anchors on top of annotations
    #
    #     If the annotation is colored in red, this means that no anchor could have a IOU of > hard_positive_threshold
    #     If the annotation is colored in green, this means one or multiple anchors have IOU > hard_positive_threshold with the annotation box
    #
    #     :param image:         (list) [h, w, 3]
    #     :param plot_name:    (list) image_names
    #     :param annotations:   (dict) with keys bboxes and labels
    #     :param dump_dir:      (str) the dump path
    #     :return:
    #     """
    #     if not os.path.exists(dump_dir):
    #         os.makedirs(dump_dir)
    #
    #     raw_image = image.copy()
    #
    #     if annotations['bboxes'].shape[0] > 0:
    #         positive_indices, _, _, max_indices = self.get_overlap_indices(annotations['bboxes'])
    #         anchors_boxes = self.anchors[positive_indices]
    #
    #         dump_path = os.path.join(dump_dir, plot_name)
    #         for anchor_box in anchors_boxes:
    #             raw_image = plots.draw_bbox(
    #                 raw_image, bbox=list(anchor_box), format='NWSE', axis='xy', show=False, width=1,
    #                 color=tuple((255, 255, 0)), caption=None, caption_color=None, draw_corner_circles=False, dump_path=None
    #             )
    #
    #         # Draw all annotation in Red
    #         for annot in annotations['bboxes']:
    #             raw_image = plots.draw_bbox(
    #                 raw_image, bbox=list(annot), format='NWSE', axis='xy', show=False, width=5,
    #                 color=tuple((255, 0, 0)), caption=None, caption_color=None, draw_corner_circles=False,
    #                 dump_path=None
    #             )
    #
    #         # Override annotation box that have anchors with dark green color
    #         annotaion_good = annotations['bboxes'][max_indices[positive_indices], :]
    #         for annot in annotaion_good:
    #             # Draw all annotation in Red
    #             raw_image = plots.draw_bbox(
    #                 raw_image, bbox=list(annot), format='NWSE', axis='xy', show=False, width=5,
    #                 color=tuple((0, 255, 0)), caption=None, caption_color=None, draw_corner_circles=False,
    #                 dump_path=None
    #             )
    #         data_io.dump_image(dump_path, raw_image)


class TransformImage:
    def __init__(self, image_resize, random_transform=False):
        self.image_resize = image_resize
        # self.transform_generator = None
        if random_transform:
            print("[Transformation] Performing all sorts of Transformation ......")
            self.transform_generator = augment.random_transform_generator(
                min_rotation=-0.1,
                max_rotation=0.1,
                min_translation=(-0.1, -0.1),
                max_translation=(0.1, 0.1),
                min_shear=-0.1,
                max_shear=0.1,
                min_scaling=(0.9, 0.9),
                max_scaling=(1.1, 1.1),
                flip_x_chance=0.5,
                flip_y_chance=0.5,
            )
        else:
            print("[Transformation] Performing Flip Transformation ......")
            self.transform_generator = augment.random_transform_generator(
                flip_x_chance=0.5, flip_y_chance=0.5
            )

        self.transform_methods = augment.TransformMethods()
        # print('transform_parameterstransform_parameters', self.transform_methods)

    def adjust_transform_for_image(self, transform, image, relative_translation):
        """ Adjust a transformation for a specific image.

        The translation of the matrix will be scaled with the size of the image.
        The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
        """
        # print('adjust_transform_for_imageadjust_transform_for_imageadjust_transform_for_image')
        # print('adjust_transform_for_image ........')
        height, width, channels = image.shape

        result = transform

        # Scale the translation with the image size if specified.
        if relative_translation:
            result[0:2, 2] *= [width, height]

        # Move the origin of transformation.
        result = augment.change_transform_origin(
            transform, (0.5 * width, 0.5 * height))

        return result

    def apply_transform(self, matrix, image, params):
        """
        Apply a transformation to an image.

        The origin of transformation is at the top left corner of the image.

        The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
        Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

        Args
          matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
          image:  The image to transform.
          params: The transform parameters (see TransformParameters)
        """
        # print('apply_transform ........')
        # print('apply_transformapply_transformapply_transformapply_transformapply_transform')
        output = cv2.warpAffine(
            image,
            matrix[:2, :],
            dsize=(image.shape[1], image.shape[0]),
            flags=params.cvInterpolation(),
            borderMode=params.cvBorderMode(),
            borderValue=params.cval,
        )
        return output

    def random_transform(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # print('random_transform ........')
        # print('928374892378239 ', image.shape)
        # print('random_transformrandom_transformrandom_transformrandom_transformrandom_transform')
        # randomly transform both image and annotations
        # print(image.dtype)
        # print(image.sum(axis=0).sum(0).sum(0))
        # self.transform_generator = None
        if transform is not None or self.transform_generator:
            if transform is None:
                # print('transformtransform : 8809080 ', self.transform_generator)
                transform = self.adjust_transform_for_image(
                    next(self.transform_generator),
                    image,
                    self.transform_methods.relative_translation,
                )
                # print('Transform: ', transform)
            # apply transformation to image
            image = self.apply_transform(
                transform, image, self.transform_methods)
            # print('image: ', image.dtype)
            # Transform the bounding boxes in the annotations.
            annotations["bboxes"] = annotations["bboxes"].copy()
            for index in range(annotations["bboxes"].shape[0]):
                annotations["bboxes"][index, :] = augment.transform_aabb(
                    transform, annotations["bboxes"][index, :]
                )

        return image, annotations

    def preprocess(self, image, annotations, mode):
        """ Preprocess image and its annotations.
        """
        # print('preprocess_image + resize_image ........')
        image = augment.preprocess_image(
            np.expand_dims(np.array(image, dtype=np.float32), axis=0), mode=mode
        )

        image, image_scale = augment.resize_image(
            image[0], self.image_resize[0], self.image_resize[1]
        )
        # print('Resized Img: ', image)

        # preprocess the image
        # image = trans.preprocess_image(np.expand_dims(image, 0))

        # apply resizing to annotations too
        annotations["bboxes"] *= image_scale

        return image, annotations, image_scale
