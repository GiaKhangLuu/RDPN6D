from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.solver.build import get_default_optimizer_params

from core.gdrn_modeling.models.GDRN import GDRN
from core.gdrn_modeling.models.resnet_backbone import ResNetBackboneNet, resnet_spec
from core.gdrn_modeling.models.cdpn_rot_head_region import RotWithRegionHead
from core.gdrn_modeling.models.conv_pnp_net import ConvPnPNet
from core.gdrn_modeling.models.resnet_backbone import ResNetBackboneNet, resnet_spec
from core.gdrn_modeling.data_loader import (
    build_gdrn_train_loader, 
    build_gdrn_test_loader, 
    GDRN_DatasetFromList,
    build_gdrn_augmentation
)
from core.gdrn_modeling.gdrn_custom_evaluator import GDRN_EvaluatorCustom
from lib.torch_utils.solver.lr_scheduler import flat_and_anneal_lr_scheduler
from lib.torch_utils.solver.ranger import Ranger
import ref

DATASETS=dict(
    TRAIN=("lumi_piano_train",),
    TRAIN2=("syn_lumi_piano_train",),
    TRAIN2_RATIO=1.0,
    TEST=("lumi_piano_test",),
)

train_obj_names = ref.lumi_piano.objects

resnet_num_layers = 34
block_type, layers, channels, name = resnet_spec[resnet_num_layers]

output_res=64
num_regions = 32
xyz_loss_type="L1"
mask_loss_type="L1"
xyz_loss_mask_gt="visib"
xyz_bin=64
pnp_rot_type="allo_rot6d"

# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py
train=dict(
    output_dir="output/gdrn/lumi_piano/2025_04_14_01",
    checkpointer=dict(
        period=5000,
        max_to_keep=100,
    ),  
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    log_period=20,
    eval_period=5000,
    total_epochs=50,
    ims_per_batch=24
)

dataset_common_cfg=dict(
    color_aug_prob=0.0,
    color_aug_type="code",
    color_aug_code=(
        "Sequential(["
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
    aug_depth=False,
    input_format="BGR",
    input_with_depth=False,
    pnp_net_num_pm_points=3000,
    pnp_net_rot_type=pnp_rot_type,
    input_res=256,
    output_res=output_res,
    test_bbox_type="gt",
    dzi_pad_scale=1.5,
    dzi_type="uniform",
    dzi_scale_ratio=0.25,   
    dzi_shift_ratio=0.25,   
    train_vis=False,
    pixel_mean=[0.0, 0.0, 0.0],
    pixel_std=[255.0, 255.0, 255.0],
    smooth_xyz=False,
    cdpn_name="GDRN",
    rot_head_num_regions=num_regions,
    rot_head_xyz_loss_type=xyz_loss_type,
    rot_head_xyz_bin=xyz_bin,
    rot_head_xyz_loss_mask_gt=xyz_loss_mask_gt,
    filter_visib_thr=0.1,
    load_dets_test=False,
    det_files_test=(),
    det_topk_per_obj=1,
    det_thr=0.0, 
    train_objs=train_obj_names,
    filter_empty_dets=True, 
)

dataloader = OmegaConf.create()

train_dataset=L(GDRN_DatasetFromList)(
    dataset_dicts=L(get_detection_dataset_dicts)(
        names=DATASETS.get("TRAIN"),
        filter_empty=True,
        min_keypoints=0,
        proposal_files=None
    ),
    augmentation=build_gdrn_augmentation(
        min_size=(480,),
        max_size=640,
        sample_style="choice",
        is_train=True
    ),
    split='train',
    copy=False,
    **dataset_common_cfg
)

if len(DATASETS.get("TRAIN2", ())) > 0 and DATASETS.get("TRAIN2_RATIO", 0.0) > 0.0:
    syn_train_dataset=L(GDRN_DatasetFromList)(
        dataset_dicts=L(get_detection_dataset_dicts)(
            names=DATASETS.get("TRAIN2"),
            filter_empty=True,
            min_keypoints=0,
            proposal_files=None
        ),
        augmentation=build_gdrn_augmentation(
            min_size=(480,),
            max_size=640,
            sample_style="choice",
            is_train=True
        ),
        split='train',
        copy=False,
        **dataset_common_cfg
    )
else:
    syn_train_dataset = {}

dataloader.train=L(build_gdrn_train_loader)(
    sampler_name="TrainingSampler",
    repeat_threshold=0.0,
    img_per_batch=train.get("ims_per_batch"),
    aspect_ratio_grouping=False, 
    num_workers=0
)

dataloader.test=L(build_gdrn_test_loader)(
    dataset=L(GDRN_DatasetFromList)(
        dataset_dicts=L(get_detection_dataset_dicts)(
            names=DATASETS.get("TEST"),
            filter_empty=False,
            proposal_files=None
        ),
        augmentation=build_gdrn_augmentation(
            min_size=480,
            max_size=640,
            sample_style="choice",
            is_train=False
        ),
        split='test',
        flatten=False,
        **dataset_common_cfg
    ),
    num_workers=4,
)

dataloader.evaluator=L(GDRN_EvaluatorCustom)(
    dataset_name=DATASETS.get("TEST")[0],
    distributed=False,
    output_dir="./output/gdrn/lumi_piano/2025_03_11_01",
    train_objs=train_obj_names,
    eval_precision=False,
    eval_cached=False,
    eval_print_only=False,
    use_pnp=False,  
    pnp_type="ransac_pnp", 
    rot_head_xyz_bin=xyz_bin,
    rot_head_mask_loss_type=mask_loss_type,
    rot_head_mask_thr_test=0.5,
    cdpn_task="rot",
    exp_id="a6_cPnP_lumi_piano",
    test_bbox_type=dataset_common_cfg.get("test_bbox_type"),
    sym_objs=[]
) 

optimizer=L(Ranger)(
    params=L(get_default_optimizer_params)(
        base_lr=5e-5,
        weight_decay=0.0,
        bias_lr_factor=1.0,
    ),
    lr=1e-4, 
    weight_decay=0
)

lr_multiplier=L(flat_and_anneal_lr_scheduler)(
    #total_iters=total_iters,  # NOTE: TOTAL_EPOCHS * len(train_loader)
    warmup_factor=0.001,
    warmup_iters=1000,
    warmup_method="linear",  # default "linear"  
    anneal_method="cosine",  # "cosine"
    anneal_point=0.72,  # default 0.72    
    steps=(0.5, 0.75),
    target_lr_factor=0,
    poly_power=0.9,  # poly power
    step_gamma=0.1,
)

model=L(GDRN)(
    backbone=L(ResNetBackboneNet)(
        block=block_type, 
        layers=layers, 
        in_channel=3,
        freeze=False,
        rot_concat=False
    ),
    rot_head_net=L(RotWithRegionHead)(
        in_channels=channels[-1],
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_kernel_size=1,
        freeze=False,
        num_classes=13,
        rot_class_aware=False,
        mask_class_aware=False,
        region_class_aware=False,
        num_regions=num_regions,
        norm="BN",
        num_gn_groups=32,
        concat=False,
        backbone_num_layers=resnet_num_layers,
        xyz_loss_type=xyz_loss_type,
        mask_loss_type=mask_loss_type,
        xyz_bin=xyz_bin,
    ),
    pnp_net=L(ConvPnPNet)(
        nIn=43,
        featdim=128,
        rot_type=pnp_rot_type,
        num_regions=num_regions,
        num_layers=3,
        norm="GN",
        num_gn_groups=32,
        drop_prob=0.0,
        dropblock_size=5,
        mask_attention_type="mul",
    ),
    use_mtl=False,
    device="cuda",
    weights="./output/gdrn/lumi_piano/2025_03_11_01/model_0145349.pth",
    concat=False,
    xyz_loss_type=xyz_loss_type,
    mask_loss_type=mask_loss_type,
    backbone_out_res=output_res,
    use_pnp_in_test=False,  
    xyz_bin=xyz_bin,
    num_regions=num_regions,
    xyz_loss_mask_gt=xyz_loss_mask_gt,
    xyz_lw=1.0,
    mask_loss_gt="trunc",  
    mask_lw=1.0,
    region_loss_type="CE", 
    region_loss_mask_gt="visib",  
    region_lw=1.0,
    with_2d_coord=True,
    region_attention=True,
    r_only=False,
    trans_type="centroid_z",  
    z_type="ABS",  
    pm_lw=1.0,
    pm_loss_type="L1",  
    pm_smooth_l1_beta=1.0,
    pm_norm_by_extent=True,
    pm_loss_sym=False,  
    pm_disentangle_t=False,  
    pm_disentangle_z=False,  
    pm_t_use_points=False,
    pm_r_only=False,
    rot_lw=1.0,
    rot_loss_type="angular", 
    centroid_lw=1.0,
    centroid_loss_type="L1",
    z_lw=1.0,
    z_loss_type="L1",
    trans_lw=1.0,
    trans_loss_disentangle=True,
    trans_loss_type="L1",
    bind_lw=1.0,
    bind_loss_type="L1"
)
