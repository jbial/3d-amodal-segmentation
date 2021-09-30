"""Miscellaneous utilities
"""
import cv2
import pycocotools
import numpy as np
import detectron2.data.detection_utils as utils


def encode_seg_mask(mask_filename):
    """Encodes bitmask into COCO format
    """
    # read and encode instance mask for segmentation label
    bitmask = utils.read_image(mask_filename)
    encoded_mask = pycocotools.mask.encode(
        np.array((bitmask/255).astype('bool'), order='F')
    )
    return encoded_mask 


def blob_bbox(bitmask):
    """Gets bounding box for a single blob
    """
    blob = np.argwhere(bitmask != 0)
    blob_coords = blob[:,[1,0]]
    x, y, w, h = cv2.boundingRect(blob_coords)
    return x, y, x + w, y + h
