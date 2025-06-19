import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from models.make_model import make_model      
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import traceback

import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/custom_vehicle.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # FIXED: Correct path for processed features
    cfg.DATASETS.ROOT_DIR = './data/processed'
    print(f"Dataset path: {cfg.DATASETS.ROOT_DIR}")
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # Load data
    try:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
        print(f"Data loaded successfully: {num_classes} classes, {camera_num} cameras")
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise
    
    # Create model
    try:
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        print(f"Model created successfully")
    except Exception as e:
        print(f"Model creation failed: {e}")
        raise
    
    # Loss and optimizer
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)
    
    try:
        do_train(
            cfg,
            model,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_func,
            num_query, 
            args.local_rank
        )
    except Exception as e:
        print("\n========== DETAILED ERROR ==========")
        print(f"Error: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        print("===================================\n")
        raise e