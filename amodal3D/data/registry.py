"""Register SAILVOS data along with custom metadata
"""
import glob
from pickle import bytes_types
import re
import logging
import os
import cv2
import json
import pycocotools
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from tqdm import tqdm
from detectron2.config import get_cfg
from amodal3D.utils.utils import extract_annotation
from amodal3D.config.config import amodal3d_cfg_defaults 


class SAILVOSDataset:

    def __init__(
        self, 
        dataroot, 
        train_val_split=0.8, 
        window_size=5, 
        frame_strides=[1], 
        scale=1.0,
        occ_thresh=0.25
    ):
        """Processes scene filenames and data into JSON format
        """
        self.logger = logging.getLogger(__name__)
        self.train_val_split = train_val_split

        # extract all paths to scene directories
        self.dataroot = dataroot
        self.scene_re = re.compile("^[a-zA-Z0-9_-]+$")
        self.scene_dirs =list(map(
            lambda s: f"{dataroot}/{s}", 
            filter(self.scene_re.match, os.listdir(dataroot))
        ))
        self.scene_dirs = [f"{dataroot}/tonya_mcs_1"]
        self.logger.info(f"[SAILVOSDataset] Extracted scenes: {self.scene_dirs}")

        # hardcoded (property of dataset)
        self.H, self.W = 800, 1280  
        self.scale = scale

        # for frame/sequence sampling
        self.window_size = window_size
        self.frame_strides = frame_strides
        self.occ_thresh = occ_thresh

        # extract all the filenames for the scene
        self.scene_data = [self._extract_filenames(scene) for scene in self.scene_dirs]

        # extract labels (bit convoluted)
        label_re = re.compile("([A-Za-z])([A-Za-z\_]+)([A-Za-z])")
        self.label_map = lambda s: label_re.search(s).group(0)
        self.labels = sorted(set(map(
            self.label_map, 
            [label.split('/')[-1] for scene in self.scene_dirs for label in glob.glob(f"{scene}/[0-9]*")]
        )))
        self.idx2label = {i:l for i, l in enumerate(self.labels)}
        self.label2idx = {l:i for i, l in enumerate(self.labels)}
        self.logger.info(f"[SAILVOSDataset] Extracted thing classes: {list(self.label2idx.keys())}")

        # for creating unique integer IDs for each sequence
        self.seq_enc = self.sequence_encoding()

        # create the dataset and store on disk
        self._split_and_write()

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
        self.logger.info(f"[SAILVOSDataset] {scene_path} contains {num_frames} frames.")

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

    def sequence_encoding(self):
        """Create unique encoding for each sequence based off of the pivot frame and the scene name:

                    frame 1 ... frame N
            scene 1  (0, 0)      (0, N)
            ...
            scene M  (M, 0)      (M, N)
        """
        N = max(scene['num_frames'] for scene in self.scene_data)
        return {
            scene_name: list(range(i * N, i * N + scene['num_frames'])) 
            for i, (scene_name, scene) in enumerate(zip(self.scene_dirs, self.scene_data))
        }

    def rolling_window(self, arr, stride=1):
        shape = arr.shape[:-1] + (arr.shape[-1] - self.window_size + 1 - stride + 1, self.window_size)
        strides = arr.strides + (arr.strides[-1] * stride,)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    def _extract_datadicts(self):
        """Extracts dict objects for the images and annotations to be written to a JSON file
        """
        self.logger.info("[SAILVOSDataset] Preprocessing dataset filenames and metadata into JSON format")
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

            frames = np.arange(num_frames)
            strided_sequences = [self.rolling_window(frames, stride=fs) for fs in self.frame_strides]

            for j, sequences in enumerate(strided_sequences):
                for seq in tqdm(sequences, desc=f"Extracting [window_size={self.window_size}, stride={self.frame_strides[j]}]"):
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
                        "visible_cats": visibles[seq[self.window_size // 2]],
                        # metadata
                        "scene_name": scene,
                        "height": int(self.H * self.scale),
                        "width": int(self.W * self.scale),
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
                        obj_id = obj_file.split('/')[-2].split('_')[0]

                        anno = {
                            "mask_filename": obj_file,
                            "obj_id": int(obj_id),
                            "category_id": self.label2idx[self.label_map(obj_file.split('/')[-2])]
                        }
                        annos.append(anno)
                    img_obj["annotations"] = annos

                    slices.append(img_obj)

        return slices

    def _split_and_write(self, dataset_name="sailvos"):
        """Creates data set, splits into train/val and writes to JSON
        """
        filenames = [f"{self.dataroot}/{dataset_name}_{mode}.json" for mode in ["train", "val"]]
        if all(os.path.isfile(fn) for fn in filenames):  # dont extract
            return

        self.logger.info("[SAILVOSDataset] Extracting, splitting, and writing to JSON")
        datadicts = self._extract_datadicts()

        # randomly split into train and test
        N, partition = len(datadicts), int(self.train_val_split * len(datadicts))
        indices = np.random.permutation(N)
        train_inds, test_inds = indices[:partition], indices[partition:]

        for inds, name in zip([train_inds, test_inds], filenames):
            # write the image and annotation JSON files
            with open(name, 'w') as f:
                json.dump([datadicts[i] for i in inds], f)

        self.logger.info(f"[SAILVOSDataset] Wrote {dataset_name} data to {filenames}")

    def _read_json_data(self, dataset_name):
        with open(f"{self.dataroot}/{dataset_name}.json", 'r') as f:
            datadicts = json.load(f)
        return datadicts

    def load(self, dataset_name):
        """Load preprocessed JSON dataset and create dataset for SAILVOS scene(s)

        Args:
            dataset_name (str): Detectron2 custom dataset name

        Returns:
            list[dict]: SAILVOS data in Detectron2 format
        """
        datadicts = self._read_json_data(dataset_name)

        # create the Detectron2 dataset
        self.logger.info(f"[SAILVOSDataset] Creating dataset {dataset_name}")
        dataset_dicts = []
        for data in datadicts:
            record = {
                f"{field}_filenames": list(map(
                    lambda f: os.path.join(data["scene_name"], f), 
                    data[f"{field}_filenames"]
                ))
                for field in ["image", "depth", "camera", "range", "visible"]
            }

            # additional metadata
            pivot = len(record['image_filenames']) // 2
            record.update({
                "height": data["height"],
                "width": data["width"],
                # pivot frame unique encoding
                "image_id": self.seq_enc[data["scene_name"]][data["start"] + pivot * data["stride"]],  
                "file_name": record["image_filenames"][pivot],
                "start": data["start"],
                "end": data["end"],
                "stride": data["stride"],
                "scene_name": data["scene_name"]
            })

            vis_objs, counts = np.unique(np.load(data["visible_cats"]), return_counts=True)
            vis_counts = {ob: cnt for ob, cnt in zip(vis_objs, counts)}

            annos = []
            for anno in data["annotations"]:
                bitmask = cv2.imread(anno["mask_filename"])

                # check if object is even visible or is at least OCC_THRESH percent visible
                if (vis_counts.get(anno["obj_id"], -1) / (bitmask / 255.).sum()) < self.occ_thresh:  
                    continue

                annos.append({
                    "is_crowd": 0,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": anno["category_id"],
                    **{k: v for k, v in zip(
                        ["segmentation", "bbox"],
                        extract_annotation(bitmask, self.scale)
                    )}
                })
            record["annotations"] = annos
            dataset_dicts.append(record)

        return dataset_dicts
        
    def register(self, dataset_name):
        self.logger.info(f"[SAILVOSDataset] Registering {dataset_name}")
        DatasetCatalog.register(
            dataset_name, lambda: self.load(dataset_name)
        )
        metadata = {"thing_classes": self.idx2label}
        MetadataCatalog.get(dataset_name).set(**metadata)


if __name__.endswith("registry"):
    cfg = get_cfg()
    cfg = amodal3d_cfg_defaults(cfg)
    sailvos_ds = SAILVOSDataset(
        "datasets", 
        train_val_split=cfg.SAILVOS.TRAIN_TEST_SPLIT,
        window_size=cfg.SAILVOS.WINDOW_SIZE, 
        frame_strides=cfg.SAILVOS.FRAME_STRIDES,
        scale=cfg.SAILVOS.SCALE_RESOLUTION,
        occ_thresh=cfg.SAILVOS.OCC_THRESHOLD
    )
    for ds_name in ["sailvos_train", "sailvos_val"]:
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        sailvos_ds.register(ds_name)
