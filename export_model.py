#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig, instantiate

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.engine_utils import batch_data

def setup_cfg(config_path):
    cfg = LazyConfig.load(config_path)
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.dataloader.test.num_workers = 0
    return cfg

# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    #image = inputs[0]["image"]
    #inputs = [{"image": image}]  # remove other unused keys

    roi_img = inputs["roi_img"]
    roi_cls = inputs["roi_cls"]
    roi_cam = inputs["roi_cam"]
    roi_wh = inputs["roi_wh"]
    roi_center = inputs["roi_center"]
    resize_ratio = inputs["resize_ratio"]
    roi_coord_2d = inputs["roi_coord_2d"]
    roi_extent = inputs["roi_extent"]
    fps = inputs["fps"]

    inputs = [{
        "roi_img": roi_img,
        "roi_cls": roi_cls,
        "roi_cam": roi_cam,
        "roi_wh": roi_wh,
        "roi_center": roi_center,
        "resize_ratio": resize_ratio,
        "roi_coord_2d": roi_coord_2d,
        "roi_extent": roi_extent,
        "fps": fps
    }]

    def inference(model, inputs):
        return model.inference(inputs)

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "onnx":
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(
                traceable_model, 
                (roi_img, roi_cls, roi_cam, roi_wh, roi_center, resize_ratio, roi_coord_2d, roi_extent, fps),
                f, 
                opset_version=STABLE_ONNX_OPSET_VERSION
            )
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


def get_sample_inputs():
    # get a first batch from dataset
    test_loader = instantiate(cfg.dataloader.test)
    test_loader_iter = iter(test_loader)
    data = next(test_loader_iter)
    first_batch = batch_data(data, phase="test")
    return first_batch

def main() -> None:
    global logger, cfg, args
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["caffe2", "onnx", "torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["caffe2_tracing", "tracing", "scripting"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args.config_file)

    register_datasets_in_cfg(cfg)

    # create a torch model
    torch_model = instantiate(cfg.model)
    torch_model = torch_model.to(cfg.model.device)
    DetectionCheckpointer(torch_model).load(cfg.model.weights)
    torch_model.eval()

    # convert and save model
    if args.export_method == "tracing":
        sample_inputs = get_sample_inputs()
        exported_model = export_tracing(torch_model, sample_inputs)

    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")


if __name__ == "__main__":
    main()  # pragma: no cover