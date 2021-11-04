"""Parse SAILVOS data into detectron2 format
"""
import yaml
import torch
import logging
import numpy as np
import detectron2.data.detection_utils as utils
import torch.nn.functional as F

from .augmentation import apply_augmentations
from .transforms import ResizeTransform


class Amodal3DMapper:
    """
    Loads data objects (e.g. images, camera matrices, etc) into memory and
    returns them in a format
    """

    def __init__(self, cfg, is_train=True):
        self.augmentations = [ResizeTransform(800, 1280, 400, 640)]
        self.augmentations.pop()

        self.is_train = is_train
        self.cfg = cfg
        logger = logging.getLogger(__name__)
        logger.info(f"[Amodal3DMapper] Augmentations used: {self.augmentations}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): metadata of an image and its corresponding annotations

        Returns:
            dict: Dict to be consumed by a model
        """
        images = np.array([utils.read_image(img) for img in dataset_dict["image_filenames"]])
        depth_maps = np.array([np.load(depth) for depth in dataset_dict["depth_filenames"]])/6 - 4e-5
        range_matrices = np.array([self._range_proj_matrix(rng) for rng in dataset_dict["range_filenames"]])
        Ks, Rts = [np.array(l) for l in zip(*[
            self._camera_matrices(cams) 
            for cams in dataset_dict["camera_filenames"]
        ])]

        # pop filenames
        dataset_dict.pop("image_filenames", None)
        dataset_dict.pop("camera_filenames", None)
        dataset_dict.pop("depth_filenames", None)
        dataset_dict.pop("visible_filenames", None)
        dataset_dict.pop("range_filenames", None)

        # augmentation stuff
        images, transforms = apply_augmentations(self.augmentations, images)
        image_shape = dataset_dict["height"], dataset_dict["width"]

        dataset_dict["images"] = torch.as_tensor(images.transpose(0, 3, 1, 2)).float()
        dataset_dict["depth_maps"] = torch.as_tensor(depth_maps).float()
        dataset_dict["gproj"] = torch.as_tensor(range_matrices).float()
        dataset_dict["K"] = torch.as_tensor(Ks).float()
        dataset_dict["Rt"] = torch.as_tensor(Rts).float()

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(
                    obj, 
                    transforms, 
                    image_shape
                )
                for obj in dataset_dict.pop("annotations")
            ]
            instances = utils.annotations_to_instances(annos, image_shape, mask_format=self.cfg.INPUT.MASK_FORMAT)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def _camera_matrices(self, cam_file):
        """Extracts camera intrinsic and extrinsic matrix
        """
        # read data from yaml file
        with open(cam_file, 'r') as f:
            cam = yaml.load(f, Loader=yaml.FullLoader)

        K = np.eye(4)
        K[:3, :3] = cam["K"]
        Rt = np.vstack([cam["Rt"], [0, 0, 0, 1]])
        return K, Rt
    
    def _range_proj_matrix(self, range_file):
        """Computes the range matrix for projection
        """
        rangemat = np.fromfile(range_file, dtype='float32').reshape((4, 4, 4))
        return np.linalg.inv(rangemat[2, :, :]) @ rangemat[1, :, :]
