"""Custom data tranforms
"""
import torch
import numpy as np
import torch.nn.functional as F

from fvcore.transforms.transform import Transform
from PIL import Image


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp=None):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods, defaults to bilinear.
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        # assert img.shape[:2] == (self.h, self.w)
        # assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        # PIL only supports uint8
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).permute(0, 3, 1, 2) / 255.
        # shape = list(img.shape)
        # shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        # img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        _PIL_RESIZE_TO_INTERPOLATE_MODE = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }
        mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
        )

        return np.uint8(img.numpy())

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation

    def inverse(self):
        return ResizeTransform(self.new_h, self.new_w, self.h, self.w, self.interp)
