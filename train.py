"""Training script for 3D amodal segmentation model
"""
import cv2
import random
import torch
import amodal3D.modeling
import numpy as np
import detectron2.utils.comm as comm
import matplotlib.pyplot as plt

from detectron2.config import CfgNode
from torch.utils.data import SubsetRandomSampler
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, verify_results
from amodal3D.config import amodal3d_cfg_defaults  
from amodal3D.data import Amodal3DMapper



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=Amodal3DMapper(cfg, is_train=False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=Amodal3DMapper(cfg, is_train=True)
        )

    @classmethod
    def visualize_data(cls, cfg, dataset_name, grid_shape=(2, 2)):
        """Visualize raw data
        """
        X, Y = grid_shape
        fig = plt.figure(figsize=(30, 30), dpi=200)

        # create an image grid
        images = []
        loader = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        for d in random.sample(loader, X * Y):
            img = cv2.imread(d["file_name"])
            img = cv2.resize(
                img, 
                dsize=[int(d / 2) for d in img.shape[:2]][::-1], 
                interpolation=cv2.INTER_LINEAR
            )
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            images.append(out.get_image()[:, :, ::-1])

        images = np.array(images)
        _, H, W, C = images.shape
        image_grid = images.reshape(*grid_shape, H, W, C).swapaxes(1, 2).reshape(X * H, Y * W, C)
        
        plt.imshow(image_grid)
        plt.axis('off')
        fig.savefig(f"figures/test_{dataset_name}.png")

    @classmethod
    def visualize_eval_results(cls, cfg, dataset_name, model, grid_shape=(3, 3)):
        """Visualize results from trained model
        """
        X, Y = grid_shape
        fig = plt.figure(figsize=(20, 20), dpi=100)

        with torch.no_grad():
            images = []
            loader = cls.build_test_loader(cfg, dataset_name)
            metadata = MetadataCatalog.get(dataset_name)
            model.eval()
            for d, _ in zip(loader, range(X*Y)):
                im = cv2.imread(d[0]["file_name"])
                outputs = model(d)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
                v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW 
                )
                out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
                images.append(out.get_image())
                
            H, W, C = images[0].shape
            grid_size = (X * H, Y * W, C)
            image_grid = np.array(images).reshape(X, Y, H, W, C).swapaxes(1, 2).reshape(grid_size)

        plt.imshow(image_grid)
        plt.axis('off')
        fig.savefig(f"figures/eval_images_{dataset_name}.png")


def setup(args):
    """Customize and setup configs
    """
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_other_cfg(get_cfg())
    cfg = amodal3d_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="amodal-3d")
    return cfg


def main(args):
    cfg = setup(args)

    # visualize small data sample
    Trainer.visualize_data(cfg, cfg.DATASETS.TEST[0], grid_shape=(4, 4))

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)

        Trainer.visualize_eval_results(
            cfg, 
            cfg.DATASETS.TEST[0], 
            model, 
            grid_shape=(2, 2)
        )

        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
