import argparse
import os 
import detectron2
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg, get_available_datasets

if __name__ == "__main__":
    cfg_path = "./configs/gdrn/a6_cPnP_lumi_piano.py"
    cfg = Config.fromfile(cfg_path)



    #register_datasets_in_cfg(cfg)


    #print(get_available_datasets("lumi_piano"))