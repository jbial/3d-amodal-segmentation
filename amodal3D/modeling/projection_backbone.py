"""Script for building the projection backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class ProjectionBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(),

        self.encoder = self._build_encoder()

        self.H, self.W = 800, 1280

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize homogeneous coordinates and NDC coordinates
        self.X, self.Y, self.ndc_X, self.ndc_Y = self._get_coords(100, 160)

    def _get_coords(self, H, W):
        u, v = torch.arange(0, W), torch.arange(0, H)
        X, Y = torch.meshgrid(u, v)

        ndc_X = 2 * (X - 1) / W - 1
        ndc_Y = -2 * (Y - 1) / H + 1

        return X.to(self.device), Y.to(self.device), ndc_X.to(self.device), ndc_Y.to(self.device)

    def _build_encoder(self):
        """Builds encoder to reduce dimensionality of images
        """
        pretrained = models.resnet18(pretrained=False)
        return nn.Sequential(*list(pretrained.children())[:-4])

    def forward(self, images, depth, K, Rt, gproj):
        """TODO
        """
        K = K.float()
        Rt = Rt.float()

        # extract features from images and downsample depth maps
        features = self.encoder(images)
        B, C,H,W = features.shape

        pcd = self._to_pcd(features.shape, depth, Rt, gproj)

        projection = self._to_grid(pcd, features, K, Rt, H, W)

        return {"conv1": self.conv1(image)}

    def _to_pcd(self, shape, depth, Rt, gproj):
        B, C, H, W = shape

        downsampled_depth = F.interpolate(
            depth.unsqueeze(1), 
            size=(H, W),
            mode='nearest'
        ).squeeze()

        ndc_coords = torch.stack([
            self.ndc_X.flatten().repeat(B, 1),
            self.ndc_Y.flatten().repeat(B, 1),
            downsampled_depth.reshape(B, -1),
            torch.ones(B, H*W).to(torch.device(Rt.device))
        ], axis=1)

        cam_coords = torch.inverse(gproj).transpose(-2, -1) @ ndc_coords

        world_coords = torch.inverse(Rt) @ cam_coords
        world_coords.div_(world_coords[:, -1, :].unsqueeze(1))

        return world_coords

    def _to_grid(self, points, features, K, Rt, H, W):
        """Backprojects a point cloud to an image given a camera view
        """

        # convert world to camera coordinates
        cam_coords = (Rt @ points)
        cam_coords[:, :3].div_(-cam_coords[:, 2, :].unsqueeze(1))

        # # convert to image coordinates
        cam_to_img = torch.zeros(len(points), 3, 4)
        cam_to_img[..., :3] = K[:, :3, :3]

        cam_to_img[:, :2, 2] = torch.tensor([-W / 2, -H / 2]).repeat(len(points), 1)
        cam_to_img = cam_to_img.to(self.device)

        img_coords = torch.floor(cam_to_img @ cam_coords).long()

        # visibility condition
        vis_mask = ((img_coords[:, 0] >= 0) & (img_coords[:, 0] < W)) &\
                   ((img_coords[:, 1] >= 0) & (img_coords[:, 1] < H))

        transformed_imgs = torch.zeros(len(points), features.shape[1], H, W)

        pix_x = img_coords[:, 0]
        pix_y = img_coords[:, 1]


        print(vis_mask.shape)
        print(img_coords.shape)
        print((img_coords[:, 1] * W + img_coords[:, 0]).shape)
        indices = (img_coords[:, 1] * W + img_coords[:, 0])[..., vis_mask]

        # TODO: figure out the indexing
        print(indices.shape)
        print(transformed_imgs.shape)
        transformed_img[:, indices] = torch.stack([f[i] for f, i in zip(features.transpose(2, 1), vis_mask)])

        return torch.flipud(transformed_img.reshape(H, W, features.shape[1]))

    def output_shape(self):
        return {"conv1": ShapeSpec(channels=64, stride=16)}


