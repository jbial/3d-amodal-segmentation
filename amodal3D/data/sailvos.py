"""Parse SAILVOS data into detectron2 format
"""
import yaml
import copy
import logging
import torch
import pycocotools
import numpy as np
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

from .registry import SceneRegister

"""
Directive: INCOMPLETE
    
    [+] most heavy lifting done here
    [+] Using the registered dataset, load images and other data for input to model
    [+] Can put any key u want in dataset_dict object
"""
# ---------------------------------------------------------------------------------
# hand-register the train and test sets
scene = SceneRegister("data")
for name in ["sailvos_train", "sailvos_test"]:
    scene.register(name)
# ---------------------------------------------------------------------------------


class Amodal3DMapper:
    """
    Loads data objects (e.g. images, camera matrices, etc) into memory and
    returns them in a format
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): metadata of an image and its corresponding annotations

        Returns:
            dict: Dict to be consumed by a model
        """
        # TODO: incorporate data augmentation and transforms

        image = utils.read_image(dataset_dict["filename"])
        depth_map = np.load(dataset_dict["depth_filename"])
        visibles = np.load(dataset_dict["visible_filename"])
        range_matrix = self._range_proj_matrix(dataset_dict["range_filename"])
        K, Rt = self._camera_matrices(dataset_dict["camera_filename"])

        # store image in CHW format and depth map
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
        dataset_dict["depth_map"] = torch.as_tensor(depth_map)
        dataset_dict["K"] = torch.tensor(K)
        dataset_dict["Rt"] = torch.tensor(Rt)
        dataset_dict["gproj"] = torch.tensor(range_matrix)
        dataset_dict["visibles"] = torch.tensor(visibles)

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        H, W = dataset_dict["height"], dataset_dict["width"]
        annos = [self.transform_annotation(anno) for anno in dataset_dict.pop("annotations")]
        instances = utils.annotations_to_instances(annos, (H, W), mask_format='bitmask')
        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def transform_annotation(self, annotation):
        """
        Apply tranforms on the annotations (TODO) and read in the binary masks
        """
        # read and encode instance mask
        bitmask = utils.read_image(annotation["mask_filename"])
        encoded_mask = pycocotools.mask.encode(
            np.array((bitmask/255).astype('bool'), order='F')
        )
        annotation["segmentation"] = encoded_mask

        return annotation

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
        return np.linalg.inv(rangemat[1, :, :]) @ rangemat[2, :, :]


