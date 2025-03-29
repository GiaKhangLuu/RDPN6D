from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset 
)