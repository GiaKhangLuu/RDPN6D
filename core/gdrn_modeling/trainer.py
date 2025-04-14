import time
import torch
from detectron2.engine import SimpleTrainer

from .engine_utils import batch_data
import core.utils.my_comm as comm

class RDPNTrainer(SimpleTrainer):
    """
    RDPN Trainer
    """

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """

        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        batch = batch_data(data)

        out_dict, loss_dict = self.model(
            batch["roi_img"],
            gt_xyz=batch.get("roi_xyz", None),
            gt_xyz_bin=batch.get("roi_xyz_bin", None),
            gt_mask_trunc=batch["roi_mask_trunc"],
            gt_mask_visib=batch["roi_mask_visib"],
            gt_mask_obj=batch["roi_mask_obj"],
            gt_region=batch.get("roi_region", None),
            gt_allo_quat=batch.get("allo_quat", None),
            gt_ego_quat=batch.get("ego_quat", None),
            gt_allo_rot6d=batch.get("allo_rot6d", None),
            gt_ego_rot6d=batch.get("ego_rot6d", None),
            gt_ego_rot=batch.get("ego_rot", None),
            gt_trans=batch.get("trans", None),
            gt_trans_ratio=batch["roi_trans_ratio"],
            gt_points=batch.get("roi_points", None),
            sym_infos=batch.get("sym_info", None),
            roi_classes=batch["roi_cls"],
            roi_cams=batch["roi_cam"],
            roi_whs=batch["roi_wh"],
            roi_centers=batch["roi_center"],
            resize_ratios=batch["resize_ratio"],
            roi_coord_2d=batch.get("roi_coord_2d", None),
            roi_extents=batch.get("roi_extent", None),
            do_loss=True,
            fps=batch.get("fps", None)
        )
        losses = sum(loss_dict.values())
        # loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # if comm.is_main_process():
        #     storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {k: v.item()
                                for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(
            loss for loss in loss_dict_reduced.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

