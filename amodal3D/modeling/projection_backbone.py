"""Script for building the projection backbone
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torch_scatter import scatter_max


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
        self.fusion_model = self._build_fusion_model()

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
        simple_mlp = nn.Sequential(
            nn.Linear(3 + self.F, 3),
            nn.ReLU(inplace=True),
            nn.Linear(3, self.F)
        ).to(self.device)        
        
        return lambda points, features: simple_mlp(torch.cat([points, features], dim=2).permute(0,1,3,2)).permute(0,1,3,2)

    def _build_fusion_model(self):
        """Aggregates features after the backprojection step
        """
        if self.cfg.DEBUG_BACKBONE:
            # simply average over temporal dimension
            return lambda x: x.reshape(-1, self.window_size, 3, self.H_feats, self.W_feats).mean(axis=1)

        fuser = nn.Sequential(
           nn.Conv2d(self.F * self.cfg.SAILVOS.WINDOW_SIZE, self.F // 2, kernel_size=3, padding=1),
           nn.BatchNorm2d(self.F),
           nn.ReLU(inplace=True),
           nn.Conv2d(self.F // 2, self.F, kernel_size=3, padding=1)
        ).to(self.device)

        return fuser

    def forward(self, images, depth, K, Rt, gproj):
        K = K.float()
        Rt = Rt.float()

        # extract features from images and downsample depth maps
        B, T, C, H, W = images.shape
        features = self.encoder(images.view(B * T, C, H, W)).view(B, T, -1, self.H_feats, self.W_feats)

        # downsmaple the depth maps with nearest neighbor (bilinear does not work)
        downsampled_depth = F.interpolate(depth, size=(self.H_feats, self.W_feats), mode='nearest')

        # project to 3D and extract point features
        pcds = self._to_pcd(downsampled_depth, Rt, gproj)
        point_feats = self.point_model(
            pcds[..., :3, :], 
            features.reshape(B, T, -1, self.H_feats * self.W_feats)
        )

        # project points back to 2D rasters
        rasters = self._to_grid(pcds, downsampled_depth, point_feats, K, Rt)

        # fuse features along sequence length dimension
        fused_feats = self.fusion_model(rasters.reshape(B, -1, self.H_feats, self.W_feats))

        if self.cfg.DEBUG_BACKBONE:
            self.visualize_debug(
                features, 
                rasters.reshape(B, T, C, self.H_feats, self.W_feats), 
                fused_feats,
                pcds,
                point_feats
            )

        # append temporally aggregated features with features fom central frame
        residual_feats = fused_feats + features[:, T // 2, :, :, :]

        return {"aggregated": residual_feats}

    def _to_pcd(self, depth, Rt, gproj):
        """Computes point cloud coordinates based on depth maps and RAGE matrices
        """
        B, T, _, _ = depth.shape
        H, W = self.H_feats, self.W_feats

        nd_coords = torch.stack(
            [
                self.ndc_X.flatten().repeat(B, T, 1),
                self.ndc_Y.flatten().repeat(B, T, 1),
                depth.flatten(start_dim=2), 
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

    def _to_grid(self, points, depth, features, K, Rt):
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

        # quantize to each corner of pixel grid cells in image
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
        depth_values = depth.flatten(start_dim=2).repeat(1,1,4) * index_mask.squeeze().repeat(1,1,4)

        # rasterize point clouds to camera with depth ordering
        _, visible_indices = scatter_max(
            depth_values,
            indices_clamped,
            dim=-1,
            out=torch.full((B, T, H * W), float('-inf'))
        )

        rasters = torch.gather(
            torch.cat([values, torch.zeros(B, T, F, 1)], dim=-1),
            dim=-1,
            index=visible_indices.unsqueeze(2).repeat(1,1,F,4)
        )
        # average over the 4 quantized images i.e. the images quantized via
        # {(floor(x), floor(y)), (ceil(x), floor(y)), (floor(x), ceil(y)), (ceil(x), ceil(y))}
        rasters = rasters.view(B, T, F, 4, -1).mean(dim=3)

        return rasters.view(B, -1, self.H_feats, self.W_feats)

    def output_shape(self):
        return {"aggregated": ShapeSpec(channels=self.F, stride=1)}

    def visualize_debug(self, imgs, rasters, fused_feats, pcds, point_feats):
        """Visualization for debugging purposes

        TODO: refactor this as a training hook
        """
        if not self.visualized:
            self.visualized = True
            print("[VISUALIZE DEBUG] In debug mode. Saving feature visualization to disk.")
            self.visualize_feats(imgs, rasters, fused_feats)
            self.visualize_pcds(pcds, point_feats)

    def visualize_feats(self, img_feats, raster_feats, fused_feats):
        """For debug mode only. Visualize the intermediate features coming out of the backbone

        TODO: refactor this as a training hook
        """
        # visualize random feature sequence
        index = np.random.choice(img_feats.shape[0])
        fig, axes = plt.subplots(3, 1, figsize=(30, 30))
        images = [
            torch.cat([im for im in img_feats[index]], dim=-1).permute(1,2,0).detach().cpu().numpy() / 255.,
            torch.cat([im for im in raster_feats[index]], dim=-1).permute(1,2,0).detach().cpu().numpy() / 255.,
            fused_feats[index].permute(1,2,0).detach().cpu().numpy() / 255.
        ]
        for ax, im in zip(axes, images):
            ax.axis('off')
            ax.imshow(im)
        fig.savefig("figures/debug_feats.png", dpi=200)

    def visualize_pcds(self, pcds, feats):
        """For debug mode only. Visualize the pointclouds

        TODO: refactor this as a training hook
        """
        # visualize random point sequence
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



