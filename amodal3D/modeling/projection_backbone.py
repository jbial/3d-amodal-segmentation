"""Script for building the projection backbone

TODO: add point processor model
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class ProjectionBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(),
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # take subset of resnet18 layers for feature extractor
        self.encoder = self._build_encoder()

        self.H, self.W = 800, 1280
        self.F, self.H_feats, self.W_feats = self.get_output_spatial_res()

        # point processing model
        self.point_model = self._build_point_processor()

        # model for aggregating features from different views
        self.aggregator = self._build_aggregator()

        # initialize homogeneous coordinates and NDC coordinates
        self.X, self.Y, self.ndc_X, self.ndc_Y = self._get_coords(self.H_feats, self.W_feats)

    def get_output_spatial_res(self):
        return self.encoder(torch.rand(1, 3, self.H, self.W)).shape[-3:]

    def _get_coords(self, H, W):
        u, v = np.arange(0, W), np.arange(0, H)
        X, Y = np.meshgrid(u, v)

        ndc_X = torch.tensor( 2 * (X - 1) / W - 1)
        ndc_Y = torch.tensor(-2 * (Y - 1) / H + 1)

        return X, Y, ndc_X.to(self.device), ndc_Y.to(self.device)

    def _build_encoder(self):
        """Builds encoder to reduce dimensionality of images
        """
        pretrained = models.resnet18(pretrained=False)
        return nn.Sequential(*list(pretrained.children())[:-4])

    def _build_point_processor(self):
        """Builds point cloud feature model
        """
        # for now use the identity function
        return lambda points, features: points

    def _build_aggregator(self):
        """Aggregates features after the backprojection step
        """
        return nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d((self.F + 3) * self.cfg.SAILVOS.WINDOW_SIZE, self.F, kernel_size=3),
            nn.BatchNorm2d(self.F),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(self.F, self.F, kernel_size=1)
        )

    def forward(self, images, depth, K, Rt, gproj):
        K = K.float()
        Rt = Rt.float()

        # extract features from images and downsample depth maps
        B, T, C, H, W = images.shape
        features = self.encoder(
            images.view(B * T, C, H, W)
        ).view(B, T, -1, self.H_feats, self.W_feats)

        # project to 3D, then project back to 2D in a single camera view
        pcd = self._to_pcd(features.shape, depth, Rt, gproj)

        projection = self._to_grid(pcd, features, K, Rt).view(B, -1, self.H_feats, self.W_feats)

        # reduce features along sequence length dimension
        agg = self.aggregator(projection)

        return {"aggregated": agg}

    def _to_pcd(self, shape, depth, Rt, gproj):
        B, T, C, H, W = shape

        downsampled_depth = F.interpolate(
            depth.unsqueeze(2), 
            size=(1, H, W),
            mode='nearest'
        ).squeeze()

        nd_coords = torch.stack(
            [
                self.ndc_X.flatten().repeat(B, T, 1),
                self.ndc_Y.flatten().repeat(B, T, 1),
                downsampled_depth.flatten(start_dim=2), 
                torch.ones(B, T, H * W)
            ],
            axis=2
        )  # shape: (B, T, 4, H*W)

        # map to world coords
        cam_coords = gproj.transpose(-1, -2) @ nd_coords.float()

        # map into world coords
        # TODO: make Rt inverse more efficient
        world_coords = torch.inverse(Rt) @ cam_coords
        world_coords = world_coords / world_coords[:, :, -1, :].unsqueeze(2)

        return world_coords

    def _to_grid(self, points, features, K, Rt):
        """Project back into 2D (rasterize), only in the radius camera
        """
        # concat image features with point features
        B, T, _, H, W = features.shape
        point_feats = torch.cat([
            features.view(B, T, -1, H * W),
            points[..., :-1, :]
        ], dim=2)
        _, _, F, _ = point_feats.shape

        radius = T // 2

        # convert world to camera coordinates
        intrinsics = K[:, radius, :3, :3].repeat(T, 1, 1, 1).transpose(0, 1)
        extrinsics = Rt[:, radius, :3, :].repeat(T, 1, 1, 1).transpose(0, 1)

        backproj_camcoords = (extrinsics @ points)
        backproj_camcoords.div_(-backproj_camcoords[..., -1, :].unsqueeze(2))

        # convert to image coordinates with intrinsic matrices
        intrinsics[..., :2, 2] = torch.tensor([-W / 2, -H / 2])
        intrinsics[..., 0, 0] *= W / self.W
        intrinsics[..., 1, 1] *= H / self.H

        backproj_imgcoords = (intrinsics @ backproj_camcoords)[..., :-1, :]

        # quantize the back projected image coordinates
        points_x = torch.cat(
            [
                torch.floor(backproj_imgcoords[..., 0, :]), 
                torch.floor(backproj_imgcoords[..., 0, :]), 
                torch.ceil(backproj_imgcoords[..., 0, :]), 
                torch.ceil(backproj_imgcoords[..., 0, :]), 
            ],
            dim=-1
        )
        points_y = torch.cat(
            [
                torch.floor(backproj_imgcoords[..., 1, :]),
                torch.ceil(backproj_imgcoords[..., 1, :]),
                torch.floor(backproj_imgcoords[..., 1, :]),
                torch.ceil(backproj_imgcoords[..., 1, :]),
            ],
            dim=-1
        )
        points_flat = (points_y * W + points_x).clamp(min=0, max=(H * W - 1)).long()
        points_idx = points_flat.unsqueeze(2).repeat(1, 1, F, 1)

        # rasterize with a scatter mean procedure
        pnt_mask = (points_flat > 0) & (points_flat < H * W)
        idx_mask = (points_idx > 0) & (points_idx < H * W)
        zeros = torch.zeros(B, T, F, H * W).float()
        values = point_feats.repeat(1,1,1,4)
        raster_sums = torch.scatter_add(zeros, -1, points_idx * idx_mask, values)
        raster_cnts = torch.scatter_add(zeros[..., 0, :], -1, points_flat * pnt_mask, torch.ones_like(points_flat).float())
        rasters = raster_sums / raster_cnts.unsqueeze(2).clamp(min=1)
        return rasters

    def output_shape(self):
        return {"aggregated": ShapeSpec(channels=self.F, stride=1)}


