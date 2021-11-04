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
    cfg.SAILVOS.SCALE_RESOLUTION = 0.5
    cfg.SAILVOS.OCC_THRESHOLD = 0.2

    cfg.MODEL.PRETRAINED_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    cfg.INPUT.BASE_H = 800
    cfg.INPUT.BASE_W = 1280

    return cfg

