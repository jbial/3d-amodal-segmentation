"""Default config file
"""
from detectron2.config import CfgNode as CN


def amodal3d_cfg_defaults(cfg):
    """Custom config for amdoal 3D segmentation

    TODO
    """
    cfg.SAILVOS = CN()
    cfg.SAILVOS.WINDOW_SIZE = 5
    cfg.SAILVOS.FRAME_STRIDES = [1,2,4,6]

    return cfg

