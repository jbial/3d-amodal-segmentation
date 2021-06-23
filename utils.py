"""Miscellaneous utility functions
"""
import torch
import cv2
import re
import yaml
import glob
import os
import hydra
import numpy as np
import torch
import yaml
import numpy as np

from utils import *
from tqdm import tqdm
from PIL import Image


def load_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data


def load_csv(filename):
    return np.fromfile(filename, dtype='float32')


def rangemat(filename):
    return load_csv(filename).reshape((4, 4, 4))


def cammat(filepath):
    camera_data = load_yaml(filepath)
    K = np.eye(4)
    K[:3, :3] = camera_data["K"]
    Rt = np.vstack([camera_data["Rt"], [0, 0, 0, 1]])
    return K, Rt


def get_object_ids():
    names = {}
    for d in object_masks:
        stamp = random.sample(d.keys(), 1)[0]
        filename = d[stamp]
        obj_id = int(filename.split('/')[-2].split('_')[0])
        obj = re.split('0422_|proc_|prop_|ped_|_000', filename)
        names[cat_id_dict[obj_id] - 1] = obj[-2]
    return names


class SceneDataset:

    def __init__(self, root_dir, hparams):
        self.hparams = hparams
        self.scenes = [os.path.abspath(name) for name in os.listdir(root_dir) \
                       if os.path.isdir(f"{root_dir}/{name}")]

        # data parameters
        self.skip = hparams.point_cloud.skip  # image subsampling
        self.H, self.W = hparams.point_cloud.image_size

        # extract the filenames for a scene
        self.filenames = self._get_raw_filenames(self.scenes[np.random.randint(len(self.scenes)])
        self.images, self.depths, self.rangemats, self.cameras, self.masks, self.instance_dict = filenames.values()

        self.num_frames = min(map(len, [self.images, self.depths, self.rangemats, self.cameras, self.masks]))

    def _get_raw_filenames(self, scene):
        """
        Extracts filenames for:
            images, depth maps, range matrices, camera matrices, visibility masks, & instance masks
        """
        camera_folder = f"{scene}/camera"
        depth_folder = f"{scene}/depth"
        image_folder = f"{scene}/images"
        visible_folder = f"{scene}/visible"
        obj_folders = glob.glob(f"{new_data_folder}/[0-9]*")
        rangemat_folders = [f for obj in obj_folders for f in glob.glob(f"{obj}/*mesh/draw*/rage*")]

        # extract filenames
        obj_folders = glob.glob(f"{scene}/[0-9]*")
        cam_matrices = glob.glob(f"{camera_folder}/*.yaml")
        depth_maps = glob.glob(f"{depth_folder}/*.npy")
        images = glob.glob(f"{image_folder}/*.bmp")
        visibles = glob.glob(f"{visible_folder}/*.npy")
        range_matrices = list(
            set({re.split('/|_mesh', f)[-4]:f \
                 for f in all_range_matrices}.values()
        ))

        # get list of instance masks for every object
        object_masks = []
        for obj in obj_folders:
            frame_id = key=lambda f: re.split('/|\.', f)[-2]
            filenames = sorted(glob.glob(f"{obj}/*.png"), key=frame_id)
            frame_ids = [int(frame_id(file)) for file in filenames]
            object_masks.append(dict(zip(frame_ids, filenames)))

        # sort the files by frame ID
        range_matrices.sort(key=lambda f: re.split('/|_mesh', f)[-4])
        cam_matrices.sort(key=lambda f: re.split('/|\.', f)[-2])
        depth_maps.sort(key=lambda f: re.split('/|\.', f)[-2])
        images.sort(key=lambda f: re.split('/|\.', f)[-2])
        visibles.sort(key=lambda f: re.split('/|\.', f)[-2])

        return {
            "image_files": images,
            "depth_files": depth_maps,
            "range_files": range_matrices,
            "camera_files": cam_matrices,
            "visible_files": visibles,
            "instance_files": object_masks
        }

    def project(self, frame):
        """Converts single frame to point cloud
        """
        assert frame >= 0 and frame < self.num_frames

        # extract raw data
        image = Image.open(self.images[frame])
        depth_map = np.load(self.depths[frame]) / 6 - 4e-5
        instance_mask = np.load(self.masks[frame])
        K, Rt = cammat(self.cameras[frame])
        range_matrix = rangemat(self.rangemats[frame])

        H, W, C = image.shape
        colors = image.reshape(H * W, C)

        # homogeneous pixel coords
        u, v = np.arange(0, W), np.arange(0, H)
        X, Y = np.meshgrid(u, v)

        # -> normalized device coords
        nd_X =  2 * (X - 1) / W - 1
        nd_Y = -2 * (Y - 1) / H + 1
        ndc = np.stack([nd_X.flatten(), nd_Y.flatten(), depth_map.flatten(), np.ones(H * W)])

        # -> camera coords
        gproj = np.linalg.inv(range_matrix[1, :, :]) @ range_matrix[2, :, :]
        cam_coords = np.linalg.inv(gproj).T @ ndc

        # -> world coords
        world_coords = np.linalg.inv(Rt) @ cam_coords
        world_coords /= world_coords[-1]

        return world_coords

    def backproject(self, pointcloud, colors, camera):
        """Converts point cloud into image given a camera
        """
        K, Rt = camera

        # world coords -> camera coords
        cam_coords = Rt @ pointcloud
        cam_coords[:3] /= -cam_coords[2]
        cam_coords = K @ cam_coords

        # clip points out of image plane
        visible = (np.abs(cam_coords[0]) <= self.W // 2) & (np.abs(cam_coords[1]) <= self.H // 2)

        masked_colors = np.zeros(colors.shape)
        masked_colors[visible] = colors[visible]

        return masked_colors.reshape(self.H, self.W, -1)


@hydra.main(config_path='./config/config.yaml')
def main(hparams):

    converter = PCDConverter()

    print(converter.scenes)



if __name__ == '__main__':
    main()
