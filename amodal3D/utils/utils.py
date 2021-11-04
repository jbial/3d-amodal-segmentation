"""Miscellaneous utilities
"""
import cv2
import pycocotools
import numpy as np
import detectron2.data.detection_utils as utils


def blob_bbox(bitmask):
    """Gets bounding box for a single blob
    """
    blob = np.argwhere(bitmask != 0)
    blob_coords = blob[:,[1,0]]
    x, y, w, h = cv2.boundingRect(blob_coords)
    return x, y, x + w, y + h


def extract_annotation(bitmask, scale):
    """Extracts encoded instance mask and bounding box for a single annotation instance
    """
    new_shape = tuple([int(s * scale) for s in bitmask.shape[:2]][::-1])
    bitmask = cv2.resize(bitmask, dsize=new_shape, interpolation=cv2.INTER_NEAREST)
    seg_mask = pycocotools.mask.encode(
        np.array((bitmask/255).prod(axis=-1).astype('uint8'), order='F')
    )
    bbox = blob_bbox(bitmask)
    return seg_mask, bbox
