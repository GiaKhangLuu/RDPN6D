import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.regnet import BlockParams
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
from core.utils.pose_utils import quat2mat_torch
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils import quaternion_lf, lie_algebra
from core.utils.solver_utils import build_optimizer_with_params

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .cdpn_rot_head_region import RotWithRegionHead, get_xyz_mask_region_out_dim
from .cdpn_trans_head import TransHeadNet

# pnp net variants
from .conv_pnp_net import ConvPnPNet
from .model_utils import compute_mean_re_te, get_mask_prob
from .point_pnp_net import PointPnPNet, SimplePointPnPNet
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .resnet_backbone import ResNetBackboneNet, resnet_spec, RegNet, MyResNetBackboneNet
from scipy.spatial.distance import cdist
logger = logging.getLogger(__name__)


def xyz_to_region(xyz_crop, fps_points, pred_region, mask):
    # xyz_crop: (64, 64, 3)  fps_points: (16, 3)

    batch_size = xyz_crop.shape[0]
    xyz_crop = xyz_crop.permute(0, 2, 3, 1)
    xyz_crop = xyz_crop.reshape(xyz_crop.shape[0], -1, 3)
    region_softmax = F.softmax(
        pred_region[:, 1:, :, :], dim=1).argmax(dim=1) + 1
    #mask_crop = F.softmax(pred_region[:, :, :, :], dim=1).argmax(dim=1) != 0
    mask_crop = mask.squeeze(dim=1) > 0.01
    dis = torch.cdist(xyz_crop, fps_points)
    region_ids = dis.argmin(dim=2).reshape(batch_size, 64, 64) + 1
    region_ids = mask_crop * region_ids
    region_softmax = mask_crop * region_softmax

    # (bh, bw)
    return (torch.sum(region_ids == region_softmax) -
            torch.sum(mask_crop == 0)) / (torch.sum(mask_crop == 1)).item()  # 0 means bg


class GDRN(nn.Module):
    def __init__(
            self, 
            backbone, 
            rot_head_net, 
            trans_head_net=None, 
            pnp_net=None,
            concat=False,
            xyz_loss_type="L1",  # L1 | CE_coor
            mask_loss_type="L1",  # L1 | BCE | CE
            backbone_out_res=64,
            use_pnp_in_test=False,
            xyz_bin=64,
            num_regions=32,
            xyz_loss_mask_gt="visib",  # trunc | visib | obj,
            xyz_lw=1.0,
            mask_loss_gt="trunc",  # trunc | visib | gt
            mask_lw=1.0,
            region_loss_type="CE",  # CE
            region_loss_mask_gt="visib",  # trunc | visib | obj
            region_lw=1.0,
            with_2d_coord=True,
            region_attention=True,
            r_only=False,
            trans_type="centroid_z",  
            z_type="ABS",  
            pm_lw=1.0,
            pm_loss_type="L1",  # L1 | Smooth_L1
            pm_smooth_l1_beta=1.0,
            pm_norm_by_extent=True,
            pm_loss_sym=False,  # use symmetric PM loss
            pm_disentangle_t=False,  # disentangle R/T
            pm_disentangle_z=False,  # disentangle R/xy/z
            pm_t_use_points=False,
            pm_r_only=False,
            rot_lw=1.0,
            rot_loss_type="angular",  # angular | L2
            centroid_lw=1.0,
            centroid_loss_type="L1",
            z_lw=1.0,
            z_loss_type="L1",
            trans_lw=1.0,
            trans_loss_disentangle=True,
            trans_loss_type="L1",
            bind_lw=1.0,
            bind_loss_type="L1",
            use_mtl=False,
            device="cuda",
            weights=None,
        ):
        super().__init__()
        self.backbone = backbone

        self.rot_head_net = rot_head_net
        self.pnp_net = pnp_net

        self.trans_head_net = trans_head_net

        self.concat = concat
        self.xyz_loss_type = xyz_loss_type
        self.mask_loss_type = mask_loss_type
        self.backbone_out_res = backbone_out_res
        self.use_pnp_in_test = use_pnp_in_test 
        self.xyz_loss_mask_gt = xyz_loss_mask_gt
        self.xyz_lw = xyz_lw
        self.mask_loss_gt = mask_loss_gt
        self.mask_lw = mask_lw   
        self.region_loss_type = region_loss_type
        self.region_loss_mask_gt = region_loss_mask_gt
        self.region_lw = region_lw
        self.with_2d_coord = with_2d_coord
        self.region_attention = region_attention
        self.r_only = r_only
        self.trans_type = trans_type
        self.z_type = z_type
        self.pm_lw = pm_lw
        self.pm_loss_type = pm_loss_type
        self.pm_smooth_l1_beta = pm_smooth_l1_beta
        self.pm_norm_by_extent = pm_norm_by_extent
        self.pm_loss_sym = pm_loss_sym
        self.pm_disentangle_t = pm_disentangle_t
        self.pm_disentangle_z = pm_disentangle_z
        self.pm_t_use_points = pm_t_use_points
        self.pm_r_only = pm_r_only
        self.rot_lw = rot_lw
        self.rot_loss_type = rot_loss_type
        self.centroid_lw = centroid_lw
        self.centroid_loss_type = centroid_loss_type
        self.z_lw = z_lw
        self.z_loss_type = z_loss_type
        self.trans_lw = trans_lw
        self.trans_loss_disentangle = trans_loss_disentangle
        self.trans_loss_type = trans_loss_type
        self.bind_lw = bind_lw
        self.bind_loss_type = bind_loss_type
        self.use_mtl = use_mtl
        self.device = device
        self.weights = weights

        self.r_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_mask_region_out_dim(
            xyz_loss_type=xyz_loss_type,
            mask_loss_type=mask_loss_type,
            xyz_bin=xyz_bin,
            num_regions=num_regions,
        )
        self.consistent = 0
        self.item = 0
        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        if self.use_mtl:  
            self.loss_names = [
                "mask",
                "coor_x",
                "coor_y",
                "coor_z",
                "coor_x_bin",
                "coor_y_bin",
                "coor_z_bin",
                "region",
                "PM_R",
                "PM_xy",
                "PM_z",
                "PM_xy_noP",
                "PM_z_noP",
                "PM_T",
                "PM_T_noP",
                "centroid",
                "z",
                "trans_xy",
                "trans_z",
                "trans_LPnP",
                "rot",
                "bind",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor(
                        [0.0], requires_grad=True, dtype=torch.float32))
                )

    def forward(
        self,
        x,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_region=None,
        gt_allo_quat=None,
        gt_ego_quat=None,
        gt_allo_rot6d=None,
        gt_ego_rot6d=None,
        gt_ego_rot=None,
        gt_points=None,
        sym_infos=None,
        gt_trans=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        roi_extents=None,
        resize_ratios=None,
        do_loss=False,
        fps=None,
    ):
        # x.shape [bs, 3, 256, 256]
        if self.concat:
            features, x_f64, x_f32, x_f16 = self.backbone(
                x)  # features.shape [bs, 2048, 8, 8]
            # joints.shape [bs, 1152, 64, 64]

            mask, coor_x, coor_y, coor_z, region = self.rot_head_net(
                features, x_f64, x_f32, x_f16)
        else:
            features = self.backbone(x)  # features.shape [bs, 2048, 8, 8]

            # joints.shape [bs, 1152, 64, 64]
            mask, coor_x, coor_y, coor_z, region = self.rot_head_net(features)

        # TODO: remove this trans_head_net
        # trans = self.trans_head_net(features)

        consistent_map = None

        device = x.device
        bs = x.shape[0]
        num_classes = self.rot_head_net.num_classes

        out_res = self.backbone_out_res

        if self.rot_head_net.rot_class_aware: 
            assert roi_classes is not None
            coor_x = coor_x.view(
                bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(
                bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(
                bs, num_classes, self.r_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if self.rot_head_net.mask_class_aware:  
            assert roi_classes is not None
            mask = mask.view(
                bs, num_classes, self.mask_out_dim, out_res, out_res)
            mask = mask[torch.arange(bs).to(device), roi_classes]

        if self.rot_head_net.region_class_aware:  
            assert roi_classes is not None
            region = region.view(
                bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat(
                [coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW

        if self.with_2d_coord:  
            assert roi_coord_2d is not None
            coor_feat = torch.cat([coor_feat, roi_coord_2d], dim=1)

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)
        # * new
        region_softmax_argmax = torch.argmax(region_softmax.reshape(
            region_softmax.shape[0], region_softmax.shape[1], -1), dim=1).unsqueeze(2)
        #region_softmax_argmax = torch.topk(region_softmax, k=64, dim=3)[1][:,:,:,0]
        #region_softmax_argmax = torch.ones_like(region_softmax_argmax)
        region_fps = torch.gather(fps.unsqueeze(
            1).expand(-1, region_softmax_argmax.shape[1], -1, -1), 2, region_softmax_argmax.unsqueeze(3).expand(-1, -1, -1, 3))
        region_fps = region_fps.squeeze(2).reshape(
            region_fps.shape[0], 64, 64, 3)
        region_fps = region_fps.permute(0, 3, 1, 2)
        #region_fps = torch.zeros_like(region_fps)
        coor_feat = torch.cat([coor_feat, region_fps], dim=1)
        # * new end
        mask_atten = None
        if self.pnp_net.mask_attention_type != "none":  
            mask_atten = get_mask_prob(self.mask_loss_type, mask)
        check_consistent = False
        # * new
        if check_consistent:
            pred_xyz_coor = torch.cat([coor_x, coor_y, coor_z], dim=1)
            self.consistent += xyz_to_region(pred_xyz_coor, fps, region, mask)
            self.item += 1

        # * new end
        region_atten = None
        if self.region_attention:  
            region_atten = region_softmax

        pred_rot_, pred_t_ = self.pnp_net(
            coor_feat, region=region_atten, extents=roi_extents, mask_attention=mask_atten
        )
        if self.r_only:  # override trans pred  
            pred_t_ = self.trans_head_net(features)

        # convert pred_rot to rot mat -------------------------
        rot_type = self.pnp_net.rot_type
        if rot_type in ["ego_quat", "allo_quat"]:
            pred_rot_m = quat2mat_torch(pred_rot_)
        elif rot_type in ["ego_log_quat", "allo_log_quat"]:
            pred_rot_m = quat2mat_torch(quaternion_lf.qexp(pred_rot_))
        elif rot_type in ["ego_lie_vec", "allo_lie_vec"]:
            pred_rot_m = lie_algebra.lie_vec_to_rot(pred_rot_)
        elif rot_type in ["ego_rot6d", "allo_rot6d"]:
            pred_rot_m = ortho6d_to_mat_batch(pred_rot_)
        else:
            raise RuntimeError(f"Wrong pred_rot_ dim: {pred_rot_.shape}")
        # convert pred_rot_m and pred_t to ego pose -----------------------------
        if self.trans_type == "centroid_z":  
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=self.z_type, 
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif self.trans_type == "centroid_z_abs":
            # abs 2d obj center and abs z
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z_abs(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=roi_cams,
                eps=1e-4,
                is_allo="allo" in rot_type,
                # is_train=True
                is_train=do_loss,  # TODO: sometimes we need it to be differentiable during test
            )
        elif self.trans_type == "trans":
            # TODO: maybe denormalize trans
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m, pred_t_, eps=1e-4, is_allo="allo" in rot_type, is_train=do_loss
            )
        else:
            raise ValueError(
                f"Unknown pnp_net trans type: {self.trans_type}")

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            out_dict.update({"mask": mask, "coor_x": coor_x, "coor_y": coor_y,
                            "coor_z": coor_z, "region": region, "consistent_map": consistent_map})
            if self.use_pnp_in_test: 
                # TODO: move the pnp/ransac inside forward
                out_dict.update({"mask": mask, "coor_x": coor_x,
                                "coor_y": coor_y, "coor_z": coor_z, "region": region})
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            mean_re, mean_te = compute_mean_re_te(
                pred_trans, pred_ego_rot, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,
                # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,
                # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                out_mask=mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                # roi_classes=roi_classes,
            )

            if self.use_mtl: 
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(
                            self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        out_mask,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
    ):
        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc,
                    "visib": gt_mask_visib, "obj": gt_mask_obj}

        # rot xyz loss ----------------------------------
        if not self.rot_head_net.freeze: 
            gt_mask_xyz = gt_masks[self.xyz_loss_mask_gt] 
            if self.xyz_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz[:,
                                                         0:1] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz[:,
                                                         1:2] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz[:,
                                                         2:3] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif self.xyz_loss_type == "CE_coor":
                gt_xyz_bin = gt_xyz_bin.long()
                loss_func = CrossEntropyHeatmapLoss(
                    reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz_bin[:,
                                                             0] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz_bin[:,
                                                             1] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz_bin[:,
                                                             2] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(
                    f"unknown xyz loss type: {self.xyz_loss_type}")
            loss_dict["loss_coor_x"] *= self.xyz_lw
            loss_dict["loss_coor_y"] *= self.xyz_lw
            loss_dict["loss_coor_z"] *= self.xyz_lw

        # mask loss ----------------------------------
        if not self.rot_head_net.freeze: 
            mask_loss_type = self.mask_loss_type
            gt_mask = gt_masks[self.mask_loss_gt]  
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(
                    reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(
                    reduction="mean")(out_mask[:, 0, :, :], gt_mask)
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(
                    reduction="mean")(out_mask, gt_mask.long())
            else:
                raise NotImplementedError(
                    f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *=  self.mask_lw

        # roi region loss --------------------
        if not self.rot_head_net.freeze: 
            region_loss_type = self.region_loss_type
            gt_mask_region = gt_masks[self.region_loss_mask_gt] 
            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(
                    reduction="sum", weight=None)  # r_head_cfg.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region *
                    gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
                loss_dict["loss_region_my"] = nn.L1Loss(reduction="mean")(
                    gt_mask_visib, out_region[:, 0, :, :]) * self.region_lw
            else:
                raise NotImplementedError(
                    f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= self.region_lw

        # point matching loss ---------------
        if self.pm_lw > 0:  
            assert (gt_points is not None) and (
                gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=self.pm_loss_type, 
                beta=self.pm_smooth_l1_beta,  
                reduction="mean",
                loss_weight=self.pm_lw, 
                norm_by_extent=self.pm_norm_by_extent,  
                symmetric=self.pm_loss_sym,  
                disentangle_t=self.pm_disentangle_t,  
                disentangle_z=self.pm_disentangle_z,  
                t_loss_use_points=self.pm_t_use_points, 
                r_only=self.pm_r_only,  
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if self.rot_lw > 0:   
            if self.rot_loss_type == "angular": 
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif self.rot_loss_type == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(
                    f"Unknown rot loss type: {self.rot_loss_type}")
            loss_dict["loss_rot"] *= self.rot_lw

        # centroid loss -------------
        if self.centroid_lw > 0:   
            assert (
                self.trans_type == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if self.centroid_loss_type == "L1":   
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(
                    out_centroid, gt_trans_ratio[:, :2])
            elif self.centroid_loss_type == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(
                    out_centroid, gt_trans_ratio[:, :2])
            elif self.centroid_loss_type == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(
                    reduction="mean")(out_centroid, gt_trans_ratio[:, :2])
            else:
                raise ValueError(
                    f"Unknown centroid loss type: {self.centroid_loss_type}")
            loss_dict["loss_centroid"] *= self.centroid_lw 

        # z loss ------------------
        if  self.z_lw > 0:  
            if self.z_type == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif self.z_type == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            if self.z_loss_type == "L1":   
                loss_dict["loss_z"] = nn.L1Loss(
                    reduction="mean")(out_trans_z, gt_z)
            elif self.z_loss_type == "L2":
                loss_dict["loss_z"] = L2Loss(
                    reduction="mean")(out_trans_z, gt_z)
            elif self.z_loss_type == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(
                    reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(
                    f"Unknown z loss type: {self.z_loss_type}")
            loss_dict["loss_z"] *= self.z_lw

        # trans loss ------------------
        if self.trans_lw > 0:  
            if self.trans_loss_disentangle:   
                # NOTE: disentangle xy/z
                if self.trans_loss_type == "L1":  
                    loss_dict["loss_trans_xy"] = nn.L1Loss(
                        reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(
                        reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif self.trans_loss_type == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(
                        out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(
                        out_trans[:, 2], gt_trans[:, 2])
                elif self.trans_loss_type == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(
                        reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(
                        reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(
                        f"Unknown trans loss type: {self.trans_loss_type}")
                loss_dict["loss_trans_xy"] *= self.trans_lw
                loss_dict["loss_trans_z"] *= self.trans_lw
            else:
                if self.trans_loss_type == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(
                        reduction="mean")(out_trans, gt_trans)
                elif self.trans_loss_type == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(
                        reduction="mean")(out_trans, gt_trans)

                elif self.trans_loss_type == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(
                        reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(
                        f"Unknown trans loss type: {self.trans_loss_type}")
                loss_dict["loss_trans_LPnP"] *= self.trans_lw

        # bind loss (R^T@t)
        if self.bind_lw > 0.0:  
            pred_bind = torch.bmm(out_rot.permute(
                0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1),
                                gt_trans.view(-1, 3, 1)).view(-1, 3)
            if self.bind_loss_type == "L1":   
                loss_dict["loss_bind"] = nn.L1Loss(
                    reduction="mean")(pred_bind, gt_bind)
            elif self.bind_loss_type == "L2":
                loss_dict["loss_bind"] = L2Loss(
                    reduction="mean")(pred_bind, gt_bind)
            elif self.bind_loss_type == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(
                    reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(
                    f"Unknown bind loss (R^T@t) type: {self.bind_loss_type}")
            loss_dict["loss_bind"] *= self.bind_lw 

        if self.use_mtl: 
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * \
                    torch.exp(-cur_log_var) + \
                    torch.log(1 + torch.exp(cur_log_var))
        return loss_dict

def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.CDPN.BACKBONE
    r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
    t_head_cfg = cfg.MODEL.CDPN.TRANS_HEAD
    pnp_net_cfg = cfg.MODEL.CDPN.PNP_NET

    if "resnet" in backbone_cfg.ARCH:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        regnet = False
        if regnet:
            backbone_net = MyResNetBackboneNet(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
            )
        else:
            backbone_net = ResNetBackboneNet(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, rot_concat=r_head_cfg.ROT_CONCAT
            )
        if backbone_cfg.FREEZE:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # rotation head net -----------------------------------------------------
        r_out_dim, mask_out_dim, region_out_dim = get_xyz_mask_region_out_dim(
            cfg)
        rot_head_net = RotWithRegionHead(
            cfg,
            channels[-1],
            r_head_cfg.NUM_LAYERS,
            r_head_cfg.NUM_FILTERS,
            r_head_cfg.CONV_KERNEL_SIZE,
            r_head_cfg.OUT_CONV_KERNEL_SIZE,
            rot_output_dim=r_out_dim,
            mask_output_dim=mask_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
        )
        if r_head_cfg.FREEZE:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # translation head net --------------------------------------------------------
        if not t_head_cfg.ENABLED:
            trans_head_net = None
            assert not pnp_net_cfg.R_ONLY, "if pnp_net is R_ONLY, trans_head must be enabled!"
        else:
            trans_head_net = TransHeadNet(
                channels[-1],  # the channels of backbone output layer
                t_head_cfg.NUM_LAYERS,
                t_head_cfg.NUM_FILTERS,
                t_head_cfg.CONV_KERNEL_SIZE,
                t_head_cfg.OUT_CHANNEL,
                freeze=t_head_cfg.FREEZE,
                norm=t_head_cfg.NORM,
                num_gn_groups=t_head_cfg.NUM_GN_GROUPS,
            )
            if t_head_cfg.FREEZE:
                for param in trans_head_net.parameters():
                    with torch.no_grad():
                        param.requires_grad = False
            else:
                params_lr_list.append(
                    {
                        "params": filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                        "lr": float(cfg.SOLVER.BASE_LR) * t_head_cfg.LR_MULT,
                    }
                )

        # -----------------------------------------------
        if r_head_cfg.XYZ_LOSS_TYPE in ["CE_coor", "CE"]:
            pnp_net_in_channel = r_out_dim - 3
        else:
            pnp_net_in_channel = r_out_dim

        if pnp_net_cfg.WITH_2D_COORD:
            pnp_net_in_channel += 2

        if pnp_net_cfg.REGION_ATTENTION:
            pnp_net_in_channel += r_head_cfg.NUM_REGIONS

        # do not add dim for none/mul
        if pnp_net_cfg.MASK_ATTENTION in ["concat"]:
            pnp_net_in_channel += 1

        if pnp_net_cfg.ROT_TYPE in ["allo_quat", "ego_quat"]:
            rot_dim = 4
        elif pnp_net_cfg.ROT_TYPE in ["allo_log_quat", "ego_log_quat", "allo_lie_vec", "ego_lie_vec"]:
            rot_dim = 3
        elif pnp_net_cfg.ROT_TYPE in ["allo_rot6d", "ego_rot6d"]:
            rot_dim = 6
        else:
            raise ValueError(f"Unknown ROT_TYPE: {pnp_net_cfg.ROT_TYPE}")

        pnp_head_cfg = pnp_net_cfg.PNP_HEAD_CFG
        pnp_head_type = pnp_head_cfg.pop("type")
        if pnp_head_type == "ConvPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                num_regions=r_head_cfg.NUM_REGIONS,
                featdim=128,
                num_layers=3,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
            )
            pnp_net = ConvPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "PointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel, rot_dim=rot_dim, num_regions=r_head_cfg.NUM_REGIONS)
            pnp_net = PointPnPNet(**pnp_head_cfg)
        elif pnp_head_type == "SimplePointPnPNet":
            pnp_head_cfg.update(
                nIn=pnp_net_in_channel,
                rot_dim=rot_dim,
                mask_attention_type=pnp_net_cfg.MASK_ATTENTION,
                # num_regions=r_head_cfg.NUM_REGIONS,
            )
            pnp_net = SimplePointPnPNet(**pnp_head_cfg)
        else:
            raise ValueError(f"Unknown pnp head type: {pnp_head_type}")

        if pnp_net_cfg.FREEZE:
            for param in pnp_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
                }
            )
        # ================================================

        # CDPN (Coordinates-based Disentangled Pose Network)
        model = GDRN(cfg, backbone_net, rot_head_net,
                     trans_head_net=trans_head_net, pnp_net=pnp_net)
        if cfg.MODEL.CDPN.USE_MTL:
            params_lr_list.append(
                {
                    "params": filter(
                        lambda p: p.requires_grad,
                        [_param for _name, _param in model.named_parameters()
                         if "log_var" in _name],
                    ),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )

        # get optimizer
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        # backbone initialization
        backbone_pretrained = cfg.MODEL.CDPN.BACKBONE.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        else:
            # initialize backbone with official ImageNet weights
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            regnet = False
            if regnet:
                load_checkpoint(
                    model.backbone, "torchvision://regnet_y_3_2gf", strict=False, logger=logger)
            else:
                pass
                load_checkpoint(model.backbone, backbone_pretrained,
                                strict=False, logger=logger)

    model.to(torch.device(cfg.MODEL.DEVICE))

    return model, optimizer
