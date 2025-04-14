import argparse
import torch

from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from core.gdrn_modeling.lazyconfig_train_net import do_train

setup_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using a configuration file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument( "--resume", action="store_true", help="Flag to resume training.")

    args = parser.parse_args()
    config_path = args.config
    is_resume = args.resume

    class Args(argparse.Namespace):
        config_file=config_path
        eval_only=False
        num_machines=1
        resume=is_resume

    args = Args()
    cfg = LazyConfig.load(config_path)

    default_setup(cfg, args)
    do_train(args, cfg)
