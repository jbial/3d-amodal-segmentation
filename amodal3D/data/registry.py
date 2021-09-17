"""Register SAILVOS data along with custom metadata
"""
import glob
import re
import logging
import os
import cv2
import json
import numpy as np

from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog
)
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SceneRegister:

    def __init__(self, dataroot, window_size=5, frame_strides=[1,2,3,4,5]):
        """Processes scene filenames and data into JSON format
        """
        # extract all paths to scene directories
        self.dataroot = dataroot
        self.scene_re = re.compile("^[a-zA-Z0-9_-]+$")
        self.scene_dirs =list(map(
            lambda s: f"{dataroot}/{s}", 
            filter(self.scene_re.match, os.listdir(dataroot))
        ))

        # hardcoded (property of dataset)
        self.H, self.W = 800, 1280  

        # for frame/sequence sampling
        self.window_size = window_size
        self.frame_strides = frame_strides

        # extract all the filenames for the scene
        self.scene_data = [self._extract_filenames(scene) for scene in self.scene_dirs]

        # extract labels (bit convoluted)
        label_re = re.compile("([A-Za-z])([A-Za-z\_]+)([A-Za-z])")
        self.label_map = lambda s: label_re.search(s).group(0)
        self.labels = set(map(
            self.label_map, 
            [label.split('/')[-1] for scene in self.scene_dirs for label in glob.glob(f"{scene}/[0-9]*")]
        ))
        self.idx2label = {i:l for i, l in enumerate(self.labels)}
        self.label2idx = {l:i for i, l in enumerate(self.labels)}

    def _extract_filenames(self, scene_path):
        """Read all relevant filenames from the scene
        """
        camera_folder = f"{scene_path}/camera"
        new_depth_folder = f"{scene_path}/depth"
        image_folder = f"{scene_path}/images"
        visible_folder = f"{scene_path}/visible"

        # get data folders
        obj_folders = glob.glob(f"{scene_path}/[0-9]*")
        cam_matrices = glob.glob(f"{camera_folder}/*.yaml")
        depth_maps = glob.glob(f"{new_depth_folder}/*.npy")
        images = glob.glob(f"{image_folder}/*.bmp")
        visibles = glob.glob(f"{visible_folder}/*.npy")

        # get range matrices for every frame by checking objects for every frame
        all_range_matrices = [f for obj in obj_folders for f in glob.glob(f"{obj}/*mesh/draw*/rage*")]
        range_matrices = list(
            set({re.split('/|_mesh', f)[-4]:f \
                 for f in all_range_matrices}.values())
        )
        visible_classes = np.load(f"{scene_path}/visible_objects.npy")

        num_frames = min(map(len, [images, visibles, depth_maps, cam_matrices]))
        logger.info(f"{scene_path} contains {num_frames} frames.")

        # segmentation mask filename for every object sorted by frame
        object_masks = []
        frame_id =  lambda f: re.split('/|\.', f)[-2]
        for obj in obj_folders:
            filenames = sorted(glob.glob(f"{obj}/*.png"), key=frame_id)
            frame_ids = [int(frame_id(file)) for file in filenames]
            object_masks.append(dict(zip(frame_ids, filenames)))

        # sort range matrix files
        range_matrices.sort(key=lambda f: re.split('/|_mesh', f)[-4])
        cam_matrices.sort(key=lambda f: re.split('/|\.', f)[-2])
        depth_maps.sort(key=lambda f: re.split('/|\.', f)[-2])
        images.sort(key=lambda f: re.split('/|\.', f)[-2])
        visibles.sort(key=lambda f: re.split('/|\.', f)[-2])

        return {
            "images": images,
            "depth_maps": depth_maps,
            "cameras": cam_matrices,
            "visibles": visibles,
            "range_matrices": range_matrices,
            "annotations": object_masks,
            "num_frames": num_frames,
            "visible_classes": visible_classes
        }

    def rolling_window(self, arr, stride=1):
        shape = arr.shape[:-1] + (arr.shape[-1] - self.window_size + 1 - stride + 1, self.window_size)
        strides = arr.strides + (arr.strides[-1] * stride,)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    def _extract_datadicts(self, dataset_name):
        """Extracts dict objects for the images and annotations to be written to a JSON file

        TODO: write train/test split logic
        """
        logger.info("Preprocessing dataset filenames and metadata into JSON format")
        path_parser = lambda fn, idx: '/'.join(fn.split('/')[-idx:])

        slices = []
        for i, scene in tqdm(enumerate(self.scene_dirs), desc="Parsing scene data"):

            # unpack data
            scene_data = self.scene_data[i]
            num_frames = scene_data["num_frames"]
            images = scene_data["images"]
            depth_maps = scene_data["depth_maps"]
            cameras = scene_data["cameras"]
            range_matrices = scene_data["range_matrices"]
            visibles = scene_data["visibles"]
            annotations = scene_data["annotations"]
            visible_classes = scene_data["visible_classes"]

            frames = np.arange(num_frames)
            strided_sequences = [self.rolling_window(frames, stride=fs) for fs in self.frame_strides]

            for j, sequences in enumerate(strided_sequences):
                for seq in tqdm(sequences, desc=f"Extracting [window_size={self.window_size}, stride={self.frame_strides[j]}]", leave=True):
                    if seq[-1] - seq[-2] != self.frame_strides[j]:
                        continue

                    # get every pathname for each data-type
                    img_filenames = list(map(lambda fn: path_parser(fn, 2), [images[ts] for ts in seq]))
                    depth_filenames = list(map(lambda fn: path_parser(fn, 2), [depth_maps[ts] for ts in seq]))
                    cam_filenames = list(map(lambda fn: path_parser(fn, 2), [cameras[ts] for ts in seq]))
                    range_filenames = list(map(lambda fn: path_parser(fn, 4), [range_matrices[ts] for ts in seq]))
                    vis_filenames = list(map(lambda fn: path_parser(fn, 2), [visibles[ts] for ts in seq]))

                    img_obj = {
                        "image_filenames": img_filenames,
                        "depth_filenames": depth_filenames,
                        "camera_filenames": cam_filenames,
                        "range_filenames": range_filenames,
                        "visible_filenames": vis_filenames,
                        # metadata
                        "scene_name": scene,
                        "height": self.H,
                        "width": self.W,
                        "start": int(seq[0]),
                        "end": int(seq[-1]),
                        "stride": int(self.frame_strides[j])
                    }

                    annos = []
                    for obj in annotations:
                        if seq[self.window_size // 2] not in obj.keys():
                            continue

                        # central frame in the window serves as the annotation
                        obj_file = obj[seq[self.window_size // 2]]
                        bitmask = cv2.imread(obj_file)
                        anno = {
                            "mask_filename": path_parser(obj_file, 2),
                            "bbox": cv2.boundingRect(np.argwhere(bitmask != 0)[:, :2]),
                            "category_id": self.label2idx[self.label_map(obj_file.split('/')[-2])]
                        }
                        annos.append(anno)
                    img_obj["annotations"] = annos

                    slices.append(img_obj)

        return slices

    def _json_data(self, dataset_name):
        """Returns the JSON objects containing the image and annotation data. Writes them to file if they dont exist
        """
        json_file = f"{self.dataroot}/{dataset_name}.json"
        if not os.path.isfile(json_file): 
            logging.info("JSON data files not yet created. Creating now (you only need to do this once).")
            imgs_json = self._extract_datadicts(dataset_name)

            # write the image and annotation JSON files
            with open(json_file, 'w') as f:
                json.dump(imgs_json, f)

            return imgs_json

        # load the JSON dicts if they already exist as files
        with open(json_file, 'r') as f:
            imgs_json = json.load(f)

        return imgs_json

    def load_sailvos(self, dataset_name="SAILVOS"):
        """Load preprocessed JSON dataset and create dataset for SAILVOS scene(s)

        Args:
            dataset_name (str): Detectron2 custom dataset name

        Returns:
            list[dict]: SAILVOS data in Detectron2 format
        """
        image_json = self._json_data(dataset_name)

        # create the catalog
        meta = MetadataCatalog.get(dataset_name)
        meta.thing_classes = list(self.label2idx)

        # create the Detectron2 dataset
        logger.info(f"Creating dataset {dataset_name}")
        dataset_dicts = []
        for img_dict in image_json:
            record = {
                f"{field}_filenames": map(lambda f: os.path.join(self.dataroot, f), img_dict[f"{field}_filenames"])
                for field in ["image", "depth", "camera", "range", "visible"]
            }

            # additional metadata
            record.update({
                "height": img_dict["height"],
                "width": img_dict["width"],
                "start": img_dict["start"],
                "end": img_dict["end"],
                "stride": img_dict["stride"],
                "scene_name": img_dict["scene_name"]
            })

            annos = [
                {
                    "mask_filename": os.path.join(self.dataroot, anno["mask_filename"]),
                    "bbox": anno["bbox"],
                    "category_id": anno["category_id"],
                    "bbox_mode": BoxMode.XYWH_ABS
                } for anno in img_dict["annotations"]
            ]
            record["annotations"] = annos

            dataset_dicts.append(record)

        return dataset_dicts

    def register(self, dataset_name):
        """Register the SAILVOS dataset
        """
        datadict = self.load_sailvos(dataset_name)
        DatasetCatalog.register(
            dataset_name, lambda: datadict
        )

        metadata = {"thing_colors": list(self.idx2label)}
        MetadataCatalog.get(dataset_name).set(image_root=self.dataroot, **metadata)
        
    
if __name__ == '__main__':
    """For testing the SAILVOS dataloader
    """
    from pprint import pprint

    meta = MetadataCatalog.get("SAILVOS")
    s = SceneRegister("data", window_size=3, frame_strides=[100])

    # pprint(s.scene_data[0]["annotations"])
    pprint(s.idx2label)
    pprint(s.label2idx)

    # test the initial extraction
    # datadicts = s._extract_datadicts("sailvos")
    # pprint(datadicts[-1])

    # test registration
    s.register("sailvos")
