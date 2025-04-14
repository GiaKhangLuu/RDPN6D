#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import numpy as np
from collections import OrderedDict
from torch.utils.data import ConcatDataset

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import (
    AMPTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm

from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.gdrn_evaluator import gdrn_inference_on_dataset
from core.gdrn_modeling.trainer import RDPNTrainer

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            test_loader = instantiate(cfg.dataloader.test)
            evaluator = instantiate(cfg.dataloader.evaluator)
            results_i = gdrn_inference_on_dataset(
                model, 
                test_loader, 
                evaluator, 
                amp_test=False
            )
            results[dataset_name] = results_i
        
        if len(results) == 1:
            results = list(results.values())[0]

        for k in results['lumi_piano']:
            results['lumi_piano'][k] = np.mean(results['lumi_piano'][k])

        return results['lumi_piano']

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """

    register_datasets_in_cfg(cfg)

    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.model.device)  

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    cfg.lr_multiplier.optimizer = optim

    train_dataset = instantiate(cfg.train_dataset)
    if cfg.syn_train_dataset:
        syn_train_dataset = instantiate(cfg.syn_train_dataset) 
        train_dataset = ConcatDataset([train_dataset, syn_train_dataset])

    cfg.dataloader.train.dataset = train_dataset
    train_loader = instantiate(cfg.dataloader.train)

    max_iter = cfg.train.total_epochs * (len(train_dataset) // cfg.train.ims_per_batch)
    cfg.lr_multiplier.total_iters = max_iter 

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else RDPNTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            (
                hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
                if comm.is_main_process()
                else None
            ),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
            hooks.BestCheckpointer(
                eval_period=cfg.train.eval_period,
                mode='min',
                checkpointer=checkpointer,
                val_metric="re"
            )
        ]
    )

    checkpointer.resume_or_load(cfg.model.weights, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, max_iter)
