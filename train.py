"""Training script for 3D amodal segmentation model
"""
import detectron2.utils.comm as comm

from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import setup_logger

from amodal3D.config import amodal3d_cfg_defaults  
from amodal3D.data import Amodal3DMapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        # TODO: Implement custom evaluator OR use COCOEvaluator
        return SAILVOSEvaluator(dataset_name, cfg, True)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # TODO: Implement custom SAILVOS data mappers
        return build_detection_test_loader(
            cfg, dataset_name, mapper=Amodal3DMapper(cfg, is_train=False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        # TODO: Implement custom SAILVOS data mappers
        return build_detection_train_loader(
            cfg, mapper=Amodal3DMapper(cfg, is_train=True)
        )


def setup(args):
    """Customize and setup configs
    """
    cfg = get_cfg()

    # TODO: implement,
    amodal3d_cfg_defaults(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "amodal-3d" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="amodal-3d")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

    loader = trainer.build_train_loader(cfg)
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

