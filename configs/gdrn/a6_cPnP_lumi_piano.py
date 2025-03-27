_base_ = ["../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/lumi_piano/2025_03_11_01"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=5000,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=500,
)

DATASETS = dict(
    TRAIN=("lumi_piano_train",),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    TEST=("lumi_piano_test",),
)

MODEL = dict(
    WEIGHTS="/home/giakhang/Downloads/camera.pth",
    LOAD_DETS_TEST=False,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
            NUM_REGIONS=32,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            REGION_ATTENTION=True,
            MASK_ATTENTION="mul",
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",  
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=False,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_TYPE="ABS",  
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
            ROT_LW=1.0,
            TRANS_LW=1.0,
            BIND_LW=1.0,
        ),
        TRANS_HEAD=dict(
            ENABLED=False,
            FREEZE=False,
        ),
    ),
)

TEST = dict(EVAL_PERIOD=20000, VIS=True, TEST_BBOX_TYPE="gt")  
TRAIN = dict(VIS_IMG=True)

