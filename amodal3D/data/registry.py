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


"""
Directive: INCOMPLETE

    [+] Create class loading filenames
    [+] implement __call__ to return the records for each record
    [+] data registry must be light-weight -> optimize heavily
    [+] Store filenames ONLY (depth, visibles, img, etc)
    [+] Okay to load camera intrinsics and range matrices, share I/O load with mapper
    [+] register the dataset thru an import
    [+] think about how to form the validation set/deal with multiple scenes (consult with alex)
        
"""
class SceneRegister:
    """Processes scene filenames and data into JSON format
    """

    def __init__(self, dataroot, frame_stride=3):
        self.dataroot = dataroot

        # for train/test split
        self.frame_stride = frame_stride

        # extract all the filenames for the scene
        self._extract_filenames()

        # metadata
        self.obj2id, self.id2obj = self.object_id_map()

    def _extract_filenames(self):
        """Read all relevant filenames from the scene
        """
        camera_folder = f"{self.dataroot}/camera"
        new_depth_folder = f"{self.dataroot}/depth"
        image_folder = f"{self.dataroot}/images"
        visible_folder = f"{self.dataroot}/visible"

        # get data folders
        self.obj_folders = glob.glob(f"{self.dataroot}/[0-9]*")
        self.cam_matrices = glob.glob(f"{camera_folder}/*.yaml")
        self.depth_maps = glob.glob(f"{new_depth_folder}/*.npy")
        self.images = glob.glob(f"{image_folder}/*.bmp")
        self.visibles = glob.glob(f"{visible_folder}/*.npy")

        all_range_matrices = [f for obj in self.obj_folders for f in glob.glob(f"{obj}/*mesh/draw*/rage*")]
        self.range_matrices = list(
            set({re.split('/|_mesh', f)[-4]:f \
                 for f in all_range_matrices}.values())
        )

        visible_classes = np.load(f"{self.dataroot}/visible_objects.npy")
        # exclude 0 class, assume visible objects file is presorted
        self.cat_id_dict = {ID:i for i, ID in enumerate(visible_classes[1:])}

        # get range matrices for every frame by checking objects for every frame
        self.num_frames = min([
            len(self.cam_matrices),
            len(self.depth_maps),
            len(self.images),
            len(self.visibles)
        ])

        # segmentation mask filename for every object sorted by frame
        self.object_masks = []
        frame_id =  lambda f: re.split('/|\.', f)[-2]
        for obj in self.obj_folders:
            filenames = sorted(glob.glob(f"{obj}/*.png"), key=frame_id)
            frame_ids = [int(frame_id(file)) for file in filenames]
            self.object_masks.append(dict(zip(frame_ids, filenames)))

        # sort range matrix files
        self.range_matrices.sort(key=lambda f: re.split('/|_mesh', f)[-4])
        self.cam_matrices.sort(key=lambda f: re.split('/|\.', f)[-2])
        self.depth_maps.sort(key=lambda f: re.split('/|\.', f)[-2])
        self.images.sort(key=lambda f: re.split('/|\.', f)[-2])
        self.visibles.sort(key=lambda f: re.split('/|\.', f)[-2])

    def object_id_map(self):
        """Gets object->ID and ID->object mapping
        """
        names = [d[list(d.keys())[0]].split('/')[-2] for d in self.object_masks]
        id_name_map = {i: name for i, name in enumerate(names)}
        return {v:k for k, v in id_name_map.items()}, id_name_map

    def _extract_datadicts(self, dataset_name):
        """Extracts dict objects for the images and annotations to be written to a JSON file
        """
        logger.info("Preprocessing dataset filenames and metadata into JSON format")

        imgs = []

        # strided sampling of frames for training and testing
        if "test" in dataset_name:
            it = range(0, self.num_frames, self.frame_stride) 
        else:
            it = [i for i in range(self.num_frames) if i not in range(0, self.num_frames, self.frame_stride)]

        for timestamp in tqdm(it, desc='Processing data into JSON format'):

            # process the image + depth image
            img_filename = self.images[timestamp]
            depth_file = self.depth_maps[timestamp]
            height, width = cv2.imread(img_filename).shape[:2]

            # ...and camera matrices
            cam_filename = self.cam_matrices[timestamp]
            rng_filename = self.range_matrices[timestamp]

            img_obj = {
                "filename": '/'.join(img_filename.split('/')[-2:]),
                "height": height,
                "width": width,
                "timestamp": timestamp,
                "depth_filename": '/'.join(depth_file.split('/')[-2:]),
                "camera_filename": '/'.join(cam_filename.split('/')[-2:]),
                "range_filename": '/'.join(rng_filename.split('/')[-4:])
            }

            # process all objects present in image
            visible_fname = self.visibles[timestamp]

            annos = []
            for obj in self.object_masks:
                if timestamp not in obj.keys():
                    continue

                # otherwise check if it is visible and get annotations
                obj_file = obj[timestamp]
                obj_id = int(obj_file.split('/')[-2].split('_')[0])
                bitmask = cv2.imread(obj_file)

                anno = {
                    "mask_filename": '/'.join(obj_file.split('/')[-2:]),
                    "bbox": cv2.boundingRect(np.argwhere(bitmask != 0)[:, :2]),
                    "category_id": obj_id
                }
                annos.append(anno)

            img_obj["visible_filename"] = '/'.join(visible_fname.split('/')[-2:])
            img_obj["annotations"] = annos

            imgs.append(img_obj)

        return imgs

    def _json_data(self, dataset_name):
        """Returns the JSON objects containing the image and annotation data. Writes them to file if they dont exist
        """
        json_file = f"{self.dataroot}/{dataset_name}_images.json"
        if not os.path.isfile(json_file): 
            logging.info("JSON data files not yet created. Creating now (you only need to do this once).")

            imgs_json = self._extract_datadicts(dataset_name)

            # write the image and annotation JSONs
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
        meta.thing_classes = list(self.obj2id.keys())
        meta.thing_dataset_id_to_contiguous_id = self.cat_id_dict

        # create the Detectron2 dataset
        logger.info(f"Creating dataset {dataset_name}")
        dataset_dicts = []
        for img_dict in image_json:

            record = {
                f"{field}filename": os.path.join(self.dataroot, img_dict[f"{field}filename"]) \
                for field in ["", "depth_", "camera_", "range_"]
            }

            record.update({
                "height": img_dict["height"],
                "width": img_dict["width"],
                "timestamp": img_dict["timestamp"],
                "visible_filename": os.path.join(self.dataroot, img_dict["visible_filename"])
            })

            # create the formatted annotations
            annos = [
                {
                    "mask_filename": os.path.join(self.dataroot, anno["mask_filename"]),
                    "bbox": anno["bbox"],
                    "category_id": self.cat_id_dict[anno["category_id"]],
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

        metadata = {
            "thing_dataset_id_to_contiguous_id": self.cat_id_dict,
            "thing_colors": list(range(len(self.cat_id_dict))),
        }
        MetadataCatalog.get(dataset_name).set(image_root=self.dataroot, **metadata)
        
    
if __name__ == '__main__':
    """For testing the SAILVOS dataloader
    """
    meta = MetadataCatalog.get("SAILVOS")
    s = SceneRegister("datasets/tonya_mcs_1")


