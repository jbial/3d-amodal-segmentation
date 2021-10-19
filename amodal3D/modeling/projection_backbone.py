"""Script for building the projection backbone
"""
from numpy.core.defchararray import index
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class ProjectionBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(),
        self.cfg = cfg
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.window_size = cfg.SAILVOS.WINDOW_SIZE

        # take subset of resnet18 layers for feature extractor
        self.in_channels = 3 * (1 if cfg.DEBUG_BACKBONE else cfg.SAILVOS.WINDOW_SIZE) 
        self.encoder = self._build_encoder()

        # for visualization/debugging
        self.visualized = False

        self.H, self.W = 800, 1280
        self.F, self.H_feats, self.W_feats = self.get_output_spatial_res()

        # point processing model
        self.point_model = self._build_point_processor()

        # model for aggregating features from different views
        self.aggregator = self._build_aggregator()

        # initialize homogeneous coordinates and NDC coordinates
        coords = self._get_coords(self.H_feats, self.W_feats)
        self.ndc_X, self.ndc_Y = [c.to(self.device) for c in coords[2:]]

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
        if self.cfg.DEBUG_BACKBONE:
            # hardcoded target size
            return lambda x: F.interpolate(x, size=(self.H // 8, self.W // 8), mode='bilinear')

        pretrained = models.resnet18(pretrained=False)
        return nn.Sequential(*list(pretrained.children())[:-4])

    def _build_point_processor(self):
        """Builds point cloud feature model
        """
        if self.cfg.DEBUG_BACKBONE:
            return lambda points, features: features
        # TODO: create point processing model
        return lambda points, features: features

    def _build_aggregator(self):
        """Aggregates features after the backprojection step
        """
        if self.cfg.DEBUG_BACKBONE:
            # simply average over temporal dimension
            return lambda x: x.reshape(-1, self.window_size, 3, self.H_feats, self.W_feats).mean(axis=1)
        aggregator = nn.Sequential(
           nn.Conv2d(self.F * self.cfg.SAILVOS.WINDOW_SIZE, self.F, kernel_size=3, padding=1),
           nn.BatchNorm2d(self.F),
           nn.ReLU(inplace=True),
           nn.Conv2d(self.F, self.F, kernel_size=3, padding=1)
        )

        # init aggregator weights to zero so projection processing
        # starts out as the identity function
        # TODO
        return aggregator

    def forward(self, images, depth, K, Rt, gproj):
        K = K.float()
        Rt = Rt.float()

        # extract features from images and downsample depth maps
        B, T, C, H, W = images.shape
        features = self.encoder(images.view(B * T, C, H, W)).view(B, T, -1, self.H_feats, self.W_feats)

        # project to 3D, then project back to 2D in a single camera view
        pcds = self._to_pcd(depth, Rt, gproj)
        point_feats = self.point_model(pcds, features.reshape(B, T, -1, self.H_feats * self.W_feats))

        projections = self._to_grid(pcds, point_feats, K, Rt).view(B, -1, self.H_feats, self.W_feats)

        # fuse features along sequence length dimension
        agg = self.aggregator(projections + features.reshape(B, -1, self.H_feats, self.W_feats))

        if self.cfg.DEBUG_BACKBONE:
            self.visualize_feats(features, projections.reshape(B, T, C, self.H_feats, self.W_feats), agg)

        return {"aggregated": agg}

    def _to_pcd(self, depth, Rt, gproj):
        """Computes point cloud coordinates based on depth maps and RAGE matrices
        """
        B, T, _, _ = depth.shape
        H, W = self.H_feats, self.W_feats

        downsampled_depth = F.interpolate(depth, size=(H, W), mode='nearest')
        nd_coords = torch.stack(
            [
                self.ndc_X.flatten().repeat(B, T, 1),
                self.ndc_Y.flatten().repeat(B, T, 1),
                downsampled_depth.flatten(start_dim=2), 
                torch.ones(B, T, H * W).to(self.device)
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
        # concat image features with XYZ point features
        B, T, F, _ = features.shape
        H, W = self.H_feats, self.W_feats

        radius = T // 2

        # convert world to camera coordinates
        intrinsics = K[:, radius, ...].unsqueeze(1).repeat(1, T, 1, 1)
        extrinsics = Rt[:, radius, ...].unsqueeze(1).repeat(1, T, 1, 1)

        backproj_camcoords = (extrinsics @ points)
        backproj_camcoords[..., :2, :].div_(-backproj_camcoords[..., 2, :].unsqueeze(2))

        # convert to image coordinates with intrinsic matrices
        intrinsics[..., :2, -1] = torch.tensor([W / 2, H / 2])
        intrinsics[..., 0, 0] *= self.H_feats/self.H
        intrinsics[..., 1, 1] *= -self.W_feats/self.W

        backproj_imgcoords = intrinsics @ backproj_camcoords

        # visibility condition
        index_mask = (
            ((backproj_imgcoords[..., 0, :] > 0) & (backproj_imgcoords[..., 0, :] < W)) &\
            ((backproj_imgcoords[..., 1, :] > 0) & (backproj_imgcoords[..., 1, :] < H))         
        ).unsqueeze(2)

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

        indices = (points_y * W + points_x).long()
        indices_clamped = indices.clamp(min=0, max=H*W-1)
        values = features.repeat(1,1,1,4) * index_mask.repeat(1,1,F,4)
        
        # rasterize with a scatter mean procedure
        raster_sums = torch.zeros(B, T, F, H * W).to(self.device).scatter_add_(
            -1, 
            indices_clamped.unsqueeze(2).repeat(1, 1, F, 1), 
            values
        )
        raster_cnts = torch.zeros(B, T, H * W).to(self.device).scatter_add_(
            -1, 
            indices_clamped, 
            torch.ones_like(indices_clamped).float().to(self.device)
        )
        rasters = raster_sums / raster_cnts.unsqueeze(2).clamp(min=1)
        return rasters

    def output_shape(self):
        return {"aggregated": ShapeSpec(channels=self.F, stride=1)}

    def visualize_feats(self, img_feats, raster_feats, fused_feats):
        """For debug mode only. Visualize the intermediate features coming out of the backbone

        TODO: refactor this as a training hook
        """
        if not self.visualized:
            self.visualized = True
            index = np.random.choice(img_feats.shape[0])
            fig, axes = plt.subplots(3, 1, figsize=(30, 30))
            images = [
                torch.cat([im for im in img_feats[index]], dim=-1).permute(1,2,0) / 255.,
                torch.cat([im for im in raster_feats[index]], dim=-1).permute(1,2,0) / 255.,
                fused_feats[index].permute(1,2,0) / 255.
            ]
            for ax, im in zip(axes, images):
                ax.axis('off')
                ax.imshow(im)
            fig.savefig("figures/debug_feats.png", dpi=200)

    def visualize_pcds(self, pcds, feats):
        """For debug mode only. Visualize the pointclouds

        TODO: refactor this as a training hook
        """
        if not self.visualized:
            self.visualized = True
            B, T, _, _ = pcds.shape
            index = np.random.choice(B)
            fig = plt.figure(figsize=(15, 20))
            axes = [fig.add_subplot(1, T, t + 1, projection='3d') for t in range(T)]
            for t, ax in enumerate(axes):
                ax.axis('off')
                ax.view_init(10, 100)
                pcd = pcds[index, t]
                colors = feats[index, t].T / 255.
                ax.scatter3D(*pcd, c=colors, marker='.', s=0.5)
            fig.savefig("figures/debug_pcd.png", dpi=200)



