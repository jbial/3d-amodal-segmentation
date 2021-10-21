"""Default config file
"""
from detectron2.config import CfgNode as CN


def amodal3d_cfg_defaults(cfg):
    """Custom config for amdoal 3D segmentation
    """
    cfg.SAILVOS = CN()
    cfg.SAILVOS.WINDOW_SIZE = 5
    cfg.SAILVOS.FRAME_STRIDES = [1,2,4,6]
    cfg.SAILVOS.TRAIN_TEST_SPLIT = 0.999

    return cfg

