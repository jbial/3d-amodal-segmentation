"""Parse SAILVOS data into detectron2 format
"""
import yaml
import torch
import pycocotools
import numpy as np
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

from detectron2.data import MetadataCatalog, DatasetCatalog


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

        images = np.array([utils.read_image(img) for img in dataset_dict["image_filenames"]])
        depth_maps = np.array([np.load(depth) for depth in dataset_dict["depth_filenames"]])
        visibles = np.array([np.load(vis) for vis in dataset_dict["visible_filename"]])
        range_matrices = np.array([self._range_proj_matrix(rng) for rng in dataset_dict["range_filename"]])
        Ks, Rts = [np.array(l) for l in zip(*[self._camera_matrices(cams) for cams in dataset_dict["camera_filename"]])]

        dataset_dict["images"] = torch.as_tensor(images.transpose(0, 3, 1, 2)).float()
        dataset_dict["depth_maps"] = torch.as_tensor(depth_maps).float()
        dataset_dict["gproj"] = torch.as_tensor(range_matrices).float()
        dataset_dict["visibles"] = torch.as_tensor(visibles).float()
        dataset_dict["K"] = torch.as_tensor(Ks).float()
        dataset_dict["Rt"] = torch.as_tensor(Rts).float()

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
        # read and encode instance mask for segmentation label
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
