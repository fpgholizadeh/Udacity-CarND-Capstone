import random
import keras
import json
import numpy as np
import warnings

from threading import Thread

from rough import utils


def read_json(file_path):
    with open(file_path, "r") as f:
        input_meta = json.load(f)
    return input_meta


def read_crop_image(image_path, img_crop_yx=None, prepend_all_paths_with=None, trim_all_paths_with=None):
    img = data_io.read_image(
        image_path, prepend_all_paths_with, trim_all_paths_with)

    if img_crop_yx is not None:
        img = img[img_crop_yx[0]:img_crop_yx[2], img_crop_yx[1]:img_crop_yx[3]]

    return img


class GenerateShuffle:
    @staticmethod
    def random_shuffle_generate(meta_list, batch_size, shuffle_seed=None):
        image_paths = np.array(meta_list)
        idx_ = np.arange(len(meta_list))
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(idx_)
        image_paths = image_paths[idx_]

        batch_count = len(image_paths) - int(len(image_paths) % batch_size)
        image_paths = image_paths[0:batch_count]
        batch_image_paths = np.array(image_paths).reshape(-1, batch_size)
        return batch_image_paths


class BoundingBoxGenerator(keras.utils.Sequence):
    def __init__(
        self,
        anchor_param,
        data_prep_dict,
        batch_size,
        inp_json_path_list,
        label_tuple,
        mode,
        preprocess_mode,
        loss_on_submodels,
        epochs,
        steps_per_epoch,
        debug_args,
        limit_size,
        prepend_all_paths_with,
        trim_all_paths_with,
        **kwargs
    ):
        self.prepend_all_paths_with = prepend_all_paths_with
        self.trim_all_paths_with = trim_all_paths_with

        if mode == "debug":
            self.debug_plot_dir = debug_args["debug_plot_dir"]
            # self.plot_anchor_dir = debug_args['plot_anchor_dir']
            self.personalized_img_names = debug_args["personalized_img_names"]
            if len(self.personalized_img_names) > 0:
                limit_size = len(self.personalized_img_names)

            if not os.path.exists(self.debug_plot_dir):
                os.makedirs(self.debug_plot_dir)
        else:
            # self.plot_annotation_dir = None
            # self.plot_anchor_dir = None
            self.debug_plot_dir = None
            self.personalized_img_names = []

        print(
            "[{} Generator] Debug: debug_plot_dir: ".format(
                mode), self.debug_plot_dir
        )
        # print('[{} Generator] Debug: plot_anchor_dir: '.format(mode), self.plot_anchor_dir)
        print(
            "[{} Generator] Debug: personalized_img_names: ".format(mode),
            self.personalized_img_names,
        )

        self.batch_input_meta = []
        print("[{} Generator] Preprocessing Mode: ".format(mode), preprocess_mode)
        print("[{} Generator] epochs: ".format(mode), epochs)
        print("[{} Generator] steps_per_epoch: ".format(mode), steps_per_epoch)
        print(
            "[{} Generator] inp_csv_path_or_path_arr: ".format(
                mode), inp_json_path_list
        )
        print("[{} Generator] limit_size: ".format(mode), limit_size)

        self.loss_on_submodels = loss_on_submodels
        self.preprocess_mode = preprocess_mode
        self.shuffle_after_epoch = True
        (
            self.class_name_to_label,
            self.class_name_to_short,
            self.label_to_short,
        ) = parse_label_tuple(label_tuple)

        if type(inp_json_path_list) == str:
            self.input_meta_list = read_json(inp_json_path_list)
        elif type(inp_json_path_list) == list:
            self.input_meta_list = []
            for path_ in inp_json_path_list:
                if path_ is not None:
                    self.input_meta_list += read_json(path_)
        else:
            raise ValueError("Only str and list type accepted")

        if mode == "debug":
            self.input_meta_list = GenerateShuffle.random_shuffle_generate(
                self.input_meta_list,
                batch_size=1,
                shuffle_seed=debug_args["debug_shuffle_seed"],
            )

        # ---------------------------------------------------------------------------------
        # Create input meta if provided personalized paths
        # ---------------------------------------------------------------------------------
        if len(self.personalized_img_names) > 0:
            assert (
                batch_size == 1
            ), "When using personalized images the batch size should be 1"
            personalized_img_names = set(self.personalized_img_names)
            out_meta = []
            for meta_dict in self.input_meta_list.flatten():
                if meta_dict["img_name"] in personalized_img_names:
                    out_meta.append(meta_dict)
            self.input_meta_list = out_meta

        print(
            "[{} Generator] Total Image Paths = {}".format(
                mode, len(self.input_meta_list)
            )
        )

        # ---------------------------------------------------------------------------------
        # Create Objects for Methods
        # ---------------------------------------------------------------------------------
        self.batch_size = int(batch_size) if mode != "debug" else 1
        self.image_height = data_prep_dict["image_height"]
        self.image_width = data_prep_dict["image_width"]
        self.num_classes = data_prep_dict["num_classes"]
        self.random_transform = data_prep_dict["random_transform"]
        self.mode = mode

        self.obj_annot_loader = utils.AnnotationLoader(
            class_name_to_label=self.class_name_to_label, num_classes=self.num_classes
        )

        self.obj_transform = utils.TransformImage(
            image_resize=[self.image_height, self.image_width],
            random_transform=self.random_transform,
        )

        self.obj_gt_annot = utils.GroundTruthAnnotation(
            image_shape=[self.image_height, self.image_width, 3],
            pyramid_levels=data_prep_dict["pyramid_levels"],
            anchor_params=anchor_param,
            num_classes=self.num_classes,
            hard_positive_threshold=data_prep_dict["hard_positive_threshold"],
            hard_negative_threshold=data_prep_dict["hard_negative_threshold"],
            get_transform_bboxes=True,
        )

        # ---------------------------------------------------------------------------------
        # Steps per Epoch and Limit
        # ---------------------------------------------------------------------------------
        image_count = len(self.input_meta_list)
        if limit_size is not None:
            self.input_meta_list = self.input_meta_list[0:limit_size]

        if steps_per_epoch and mode == "train":
            if steps_per_epoch < int(len(self.input_meta_list) // self.batch_size):
                warnings.warn(
                    "Ideal StepsPerEpoch = {}//{} = {}, Provided StepsPerEpoch = {}".format(
                        len(self.input_meta_list),
                        self.batch_size,
                        (len(self.input_meta_list) // self.batch_size),
                        steps_per_epoch,
                    )
                )

            assert steps_per_epoch <= (
                len(self.input_meta_list) // self.batch_size
            ), "Steps per epoch should not exceed: len(self.input_meta_list)={} // self.batch_size={} == {}".format(
                len(self.input_meta_list),
                self.batch_size,
                len(self.input_meta_list) // self.batch_size,
            )

            self.steps_per_epoch = steps_per_epoch
        else:
            self.steps_per_epoch = int(
                len(self.input_meta_list) // self.batch_size)

        print(
            "[{} Generator] Total ImageCount = {}, BatchSize = {}, LimitSize = {}, StepsPerEpoch = {}, ".format(
                self.mode,
                image_count,
                self.batch_size,
                limit_size,
                self.steps_per_epoch,
            )
        )
        # Run the on_epoch_end function for the first time
        # from pprint import pprint
        self.func_sampling_after_every_epoch = None
        self.on_epoch_end()

        self.kwargs = kwargs

    def on_epoch_end(self):
        """
        Handle consistency across shuffle for different epochs
        'Updates The path order given a seed.
        If seed set to None then random batches are generated every,
        which should be the idle case for train_generator'
        A Very Important Note
            on_epoch_end is called only when all the batches are finished processing and the number of batches
            are defined by __len__ function.
            So if you want custom epoch enter a specified value inside the function
        :return:
        """
        # idx = np.arange(len(self.paths))
        if self.func_sampling_after_every_epoch is not None:
            self.input_meta_list = self.func_sampling_after_every_epoch()
            assert self.steps_per_epoch <= (
                len(self.input_meta_list) // self.batch_size
            ), (
                "[USING - func_sampling_after_every_epoch] "
                "Steps per epoch should not exceed: len(self.input_meta_list)={} // self.batch_size={} == {}".format(
                    len(self.input_meta_list),
                    self.batch_size,
                    len(self.input_meta_list) // self.batch_size,
                )
            )

        if self.mode == "train":
            self.batch_input_meta = GenerateShuffle.random_shuffle_generate(
                self.input_meta_list, batch_size=self.batch_size, shuffle_seed=None
            )

        else:
            self.batch_input_meta = GenerateShuffle.random_shuffle_generate(
                self.input_meta_list, batch_size=self.batch_size, shuffle_seed=None
            )
        print(
            "\n[Generator] Epoch Start ----->  Total Count of Batch Image Paths = {}, Reassigned Steps Per Epoch = {} \n".format(
                len(self.batch_input_meta), self.steps_per_epoch
            )
        )

    def __len__(self):
        """
        Denotes the number of batches per epoch
        This function should be implemented, not implementing it would
        produce not-implemented error
        This functions decides the number of batches per epoch, and only when all the batches are run
        function on_epoch_end is invoked
        :return:
        """

        #         print('[%s Generator] Number of Batches per epoch is set to be %s' % (str(self.mode), str(self.size)))
        return self.steps_per_epoch

    def __getitem__(self, index):
        """Generate one batch of data'
        :param index:
        :return:
        """
        #         print('RUNNING FOR INDEX: ', index)
        # Generate indexes of the batch
        image_meta_list = self.batch_input_meta[index]

        if self.mode == "debug":
            inputs, targets = self.__training_data_generator(
                image_meta_list, debug=True
            )
        elif self.mode == "train":
            inputs, targets = self.__training_data_generator(
                image_meta_list, debug=False
            )
        elif self.mode in ["valid", "val", "test"]:
            inputs, targets = self.__validation_data_generator(image_meta_list)
        else:
            raise ValueError("Accepted: train, valid and val for now")

        return inputs, targets

    def get_train_xy(self, img_meta_list, index_, debug):
        img_meta = img_meta_list[index_]
        image_name = img_meta["img_name"] + ".png"
        print(image_name) if self.mode == "debug" else None

        building_meta = img_meta["building_meta"]
        annotations = self.obj_annot_loader.load_annotations(
            list_of_classes=building_meta["building_class_list"],
            list_of_polygons=building_meta["building_poly_list"],
        )

        image = read_crop_image(
            img_meta["img_path"],
            img_meta["img_crop_coords"],
            self.prepend_all_paths_with,
            self.trim_all_paths_with,
        )
        # The model weights were trained with images using BGR channel, hence flip them here.
        image = image[:, :, ::-1]

        if self.func_custom_operation_on_img_and_boxes is not None:
            image, bboxes = self.func_custom_operation_on_img_and_boxes(
                image, annotations["bboxes"], mode="train"
            )
            annotations["bboxes"] = bboxes

        annotations, _ = self.obj_annot_loader.filter_annotations(
            annotations, img_shape=image.shape, image_name=image_name
        )

        if debug:
            self.debug_plot(img_meta, image, annotations, image_name)

        image, annotations = self.obj_transform.random_transform(
            image, annotations)
        image, annotation, image_scale = self.obj_transform.preprocess(
            image, annotations, mode=self.preprocess_mode
        )

        return image_name, image, annotation

    def __training_data_generator(self, image_meta_list, debug):
        """Generates data containing batch_size samples
        """
        # Initialization
        image_batch = np.empty(
            (self.batch_size, self.image_height, self.image_width, 3)
        )
        annotation_batch = [None] * self.batch_size
        img_name_arr = [None] * self.batch_size
        # Generate data
        for index_ in range(self.batch_size):
            image_name, image, annotation = self.get_train_xy(
                image_meta_list, index_, debug
            )
            img_name_arr[index_] = image_name
            image_batch[index_] = np.expand_dims(image, axis=0)
            annotation_batch[index_] = annotation

        (
            regression_batch_normal,
            regression_batch_transformed,
            labels_batch,
        ) = self.obj_gt_annot.compute_targets(image_batch, annotation_batch)

        # Add the ground truth generation for classification loss
        output = [labels_batch]

        # Add the ground truth generation for Regression loss
        if "l1_smooth" in self.loss_on_submodels:
            output += [regression_batch_transformed]
        elif "giou" in self.loss_on_submodels:
            output += [regression_batch_normal]
        else:
            pass

        # Add the ground truth generation for IOU loss
        if "iou_concat" in self.loss_on_submodels:
            output += [regression_batch_normal]

        return image_batch, output

    def get_val_xy(
        self,
        img_meta,
        img_name_arr,
        raw_image_batch,
        processed_image_batch,
        annotation_batch,
        img_scale_arr,
        index_,
        debug,
    ):
        image_name = img_meta["img_name"] + ".png"
        print(image_name) if debug else None
        img_name_arr[index_] = image_name
        building_meta = img_meta["building_meta"]
        annotations = self.obj_annot_loader.load_annotations(
            list_of_classes=building_meta["building_class_list"],
            list_of_polygons=building_meta["building_poly_list"],
        )

        image = read_crop_image(
            img_meta["img_path"],
            img_meta["img_crop_coords"],
            self.prepend_all_paths_with,
            self.trim_all_paths_with,
        )

        if self.func_custom_operation_on_img_and_boxes is not None:
            image, bboxes = self.func_custom_operation_on_img_and_boxes(
                image, annotations["bboxes"], mode="val"
            )
            annotations["bboxes"] = bboxes

        annotations, _ = self.obj_annot_loader.filter_annotations(
            annotations, img_shape=image.shape, image_name=image_name
        )

        raw_image_batch[index_] = image
        image = image[:, :, ::-1]

        image, annotations, image_scale = self.obj_transform.preprocess(
            image, annotations, mode=self.preprocess_mode
        )
        #             print('Validation after preprocess shape: ', image.shape)
        print("annotations: ", annotations)
        processed_image_batch[index_, ] = image
        annotation_batch[index_] = annotations
        img_scale_arr[index_] = image_scale

    def __validation_data_generator(self, image_meta_list):
        """Generates data containing batch_size samples
        """
        raw_image_batch = np.empty(
            (self.batch_size, self.image_height, self.image_width, 3)
        ).astype(np.float32)
        processed_image_batch = np.empty(
            (self.batch_size, self.image_height, self.image_width, 3)
        ).astype(np.float32)
        annotation_batch = [None] * self.batch_size
        img_name_arr = [None] * self.batch_size
        img_scale_arr = [None] * self.batch_size

        # Generate data
        # print('self.batch_size: ', self.batch_size)
        threads = [None] * self.batch_size
        for index_ in range(self.batch_size):
            threads[index_] = Thread(
                target=self.get_val_xy,
                args=(
                    image_meta_list[index_],
                    img_name_arr,
                    raw_image_batch,
                    processed_image_batch,
                    annotation_batch,
                    img_scale_arr,
                    index_,
                    False,
                ),
            )
            threads[index_].start()

        for i in range(len(threads)):
            threads[i].join()

        return (
            [raw_image_batch, processed_image_batch, img_name_arr, img_scale_arr],
            annotation_batch,
        )

    def debug_plot(self, img_meta, image, annotations, image_name):
        plot_dir_ = os.path.join(self.debug_plot_dir, img_meta["img_name"])
        if not os.path.exists(plot_dir_):
            os.makedirs(plot_dir_)

        func.PlotAnnotation().plot_img(
            raw_image=image[:, :, ::-1],
            annotation_dict=annotations,
            class_dict=self.label_to_short,
            plot_name="annotations.png",
            dump_dir=plot_dir_,
        )
        print("[{} Plotted] Annotations for image: ".format(
            self.mode), image_name)

        self.obj_gt_annot.plot_anchors(
            image[:, :, ::-1].astype(np.uint8),
            plot_name="anchors.png",
            annotations=annotations,
            dump_dir=plot_dir_,
        )
        print("[{} Plotted] Anchors for image: ".format(self.mode), image_name)
