"""Miscellaneous pre-compute steps
"""
import numpy as np
import detectron2.data.detection_utils as utils

from tqdm import tqdm
from detectron2.config import get_cfg
from amodal3D.config import amodal3d_cfg_defaults
from amodal3D.data.registry import SAILVOSDataset


if __name__ == '__main__':
    """Compute the image mean and std and print to stdout (may take a while)

    NOTE: set the cfg.MODEL.PIXEL_{MEAN, STD} to the output
    TODO: Make this faster (multiprocessed/multithreaded)
    """
    cfg = get_cfg()
    cfg = amodal3d_cfg_defaults(cfg)
    sailvos_ds = SAILVOSDataset(
        "datasets", 
        window_size=cfg.SAILVOS.WINDOW_SIZE, 
        frame_strides=cfg.SAILVOS.FRAME_STRIDES
    ).load("sailvos_train")

    mean, std = [], []   
    for record in tqdm(sailvos_ds, desc="Parsing dataset"):
        images = np.array([utils.read_image(img) for img in record["image_filenames"]])
        mean.append(images.mean(axis=(0, 1, 2)))
        std.append(images.std(axis=(0, 1, 2)))

    print(f"Image moments:\n\tmean: {np.mean(mean, axis=0)}\n\tstandard deviation: {np.std(std, axis=0)}")
