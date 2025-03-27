import numpy as np
import mmcv
import os.path as osp
from lib.pysixd import inout, misc
import cv2
import json
import pickle
import joblib

def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
    """
    Args:
        width:
        height:
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32)
    y = np.linspace(low, high, height, dtype=np.float32)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy

def get_fps_points(num_fps_points, fps_points_path, objs, obj2id, with_center=False):
    """convert to label based keys.

    # TODO: get models info similarly
    """
    cur_fps_points = {}
    with open(fps_points_path, 'rb') as f:
        loaded_fps_points = pickle.load(f)
    for i, obj_name in enumerate(objs):
        obj_id = obj2id[obj_name]
        if with_center:
            cur_fps_points[i] = loaded_fps_points[str(
                obj_id)][f"fps{num_fps_points}_and_center"]
        else:
            cur_fps_points[i] = loaded_fps_points[str(
                obj_id)][f"fps{num_fps_points}_and_center"][:-1]
    return cur_fps_points

def get_extents(objs, obj2id, model_dir, vertex_scale):
    """label based keys."""

    cur_extents = {}
    for i, obj_name in enumerate(objs):
        obj_id = obj2id[obj_name]
        model_path = osp.join(model_dir, f"obj_{obj_id:06d}.ply")
        model = inout.load_ply(
            model_path, vertex_scale=vertex_scale)
        pts = model["pts"]
        xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
        ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
        zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        cur_extents[i] = np.array(
            [size_x, size_y, size_z], dtype="float32")

    return cur_extents

def normalize_image(pixel_mean, pixel_std, image):
    # image: CHW format
    assert len(pixel_mean) == 3
    assert len(pixel_std) == 3
    pixel_mean = np.array(pixel_mean).reshape(-1, 1, 1)
    pixel_std = np.array(pixel_std).reshape(-1, 1, 1)
    return (image - pixel_mean) / pixel_std
