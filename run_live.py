from ultralytics import YOLO
import time
import threading
import queue
import argparse
from pathlib import Path
import time

from camera_tools import *
from lib.pysixd import inout, misc
from preprocess_utils import *
from data_utils import (
    read_image_cv2, crop_resize_by_warp_affine, my_warp_affine, get_2d_coord_np
)
from trt_infer import TensorRTInfer

frame_getter_queue = queue.Queue()  
oak_mxid = "18443010613E940F00"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--yolo11_ckpt', type=str, default="/home/giakhang/dev/yolov11/runs/segment/piano_2025-03-05_01:22:29/weights/best.engine")
    parser.add_argument('--save_every_k_frame', type=int, default=-1)

    args = parser.parse_args()

    orig_cam_intr = np.array([
        [1.57876978e+03, 0.00000000e+00, 9.54540161e+02], 
        [0.00000000e+00, 1.57876978e+03, 5.55292480e+02], 
        [0.0, 0.0, 1.0]
    ])

    segmentator = YOLO(args.yolo11_ckpt, task="segment")
    classes = ['human_hand', 'lumi_piano']
    max_pianos_det = 2
    piano_conf = 0.5

    cad_path = "./datasets/lumi_piano_dataset/models/obj_000001.ply"
    engine_path = "./engine.trt"
    trt_infer = TensorRTInfer(engine_path)
    num_fps_points = 32
    objects = [
        "lumi_piano"
    ]
    id2obj = {
        1: "lumi_piano"
    }
    obj2id = {_name: _id for _id, _name in id2obj.items()}
    cls_id = 0
    dzi_pad_scale = 1.5
    out_res = 64
    input_res = 256
    pixel_mean=[0.0, 0.0, 0.0]
    pixel_std=[255.0, 255.0, 255.0]
    rows, cols = 256, 256

    lumi_piano_model = inout.load_ply(cad_path, vertex_scale=1)

    lumi_piano_model["bbox3d_and_center"] = misc.get_bbox3d_and_center(lumi_piano_model["pts"])
    kpts_3d = lumi_piano_model["bbox3d_and_center"]

    fps_points_path = "./datasets/lumi_piano_dataset/models/fps_points.pkl"
    fps_points = get_fps_points(num_fps_points, fps_points_path, objects, obj2id)[cls_id]

    pipeline_oak = initialize_oak_cam()

    oak_thread = threading.Thread(target=stream_oak, args=(pipeline_oak,  
        frame_getter_queue, oak_mxid), daemon=True)

    oak_thread.start()
    frame_count = 0
    start_time = time.time()
    fps = 0
    timestamp = 0

    while True:
        rgb, depth = frame_getter_queue.get()

        img_size = rgb.shape[:2]
        depth_size = depth.shape[:2]
        K = orig_cam_intr.copy()
        if img_size == depth_size == (1080, 1920):
            scale = 1/3
            new_size = (int(1920*scale), int(1080*scale))
            K[:-1] = K[:-1] * scale
            rgb = cv2.resize(rgb, new_size)
            depth = depth.astype(np.int64)
            depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
            depth_m = depth / 1e3  # mm to m
            depth_m[(depth_m<0.001) | (depth_m>=np.inf)] = 0
        
        mask_drawed_img = rgb.copy()
        sixd_drawd_img = rgb.copy()
        ism_result = {
            "classes": None,
            "bboxes_xyxy": None,
            "masks": None
        }
        pred_Rs_in_frame = []
        pred_ts_in_frame = []

        ism_det = segmentator(rgb)
        ism_det = ism_det[0]

        if ism_det.masks is not None:
            ism_masks = []
            orig_shape = ism_det.orig_shape
            for i in range(len(ism_det.masks.data)):
                mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
                xy_polygon = ism_det.masks.xy[i]
                polygon = xy_polygon.astype(np.int32).reshape(-1, 1, 2)
                mask = cv2.fillPoly(mask, [polygon], 1)  

                ism_masks.append(mask)

            ism_masks = np.array(ism_masks, dtype=np.uint8)
            ism_cls = ism_det.boxes.cls.cpu().numpy()
            ism_bboxes = ism_det.boxes.xyxy.cpu().numpy()
            ism_confs = ism_det.boxes.conf.cpu().numpy()

            piano_preds_idx = np.where(ism_cls == classes.index("lumi_piano"))[0]
            human_hand_preds_idx = np.where(ism_cls == classes.index("human_hand"))[0]

            human_hand_preds_mask = ism_masks[human_hand_preds_idx]
            human_hand_confs = ism_confs[human_hand_preds_idx]
            human_hand_bboxes = ism_bboxes[human_hand_preds_idx]
            human_hand_classes = ism_cls[human_hand_preds_idx]

            piano_preds_mask = ism_masks[piano_preds_idx]
            piano_confs = ism_confs[piano_preds_idx]
            piano_bboxes = ism_bboxes[piano_preds_idx]
            piano_classes = ism_cls[piano_preds_idx]

            selected_pianos_idx = piano_confs.argsort()[-max_pianos_det:]
            piano_bboxes = piano_bboxes[selected_pianos_idx]
            piano_preds_mask = piano_preds_mask[selected_pianos_idx]
            piano_confs = piano_confs[selected_pianos_idx]
            piano_classes = piano_classes[selected_pianos_idx]

            valid_pianos_pred = piano_confs > piano_conf
            piano_bboxes = piano_bboxes[valid_pianos_pred]
            piano_preds_mask = piano_preds_mask[valid_pianos_pred]
            piano_confs = piano_confs[valid_pianos_pred]
            piano_classes = piano_classes[valid_pianos_pred]

            ism_result["classes"] = np.concatenate((human_hand_classes, piano_classes))
            ism_result["bboxes_xyxy"] = np.concatenate((human_hand_bboxes, piano_bboxes))
            ism_result["masks"] = np.concatenate((human_hand_preds_mask, piano_preds_mask)) 

            ism_pred_cls = ism_result["classes"]
            ism_pred_boxes = ism_result["bboxes_xyxy"]
            human_hand_preds_idx = np.where(ism_pred_cls == classes.index("human_hand"))[0]
            lumi_piano_preds_idx = np.where(ism_pred_cls == classes.index("lumi_piano"))[0]

            pred_ins_masks = ism_result["masks"]
            mask = np.zeros((mask_drawed_img.shape[0], mask_drawed_img.shape[1]), dtype=np.uint8)
            for pred_idx in range(pred_ins_masks.shape[0]):
                mask += pred_ins_masks[pred_idx]
            mask_drawed_img = cv2.bitwise_and(mask_drawed_img, mask_drawed_img, mask=mask)

            for i, box in enumerate(ism_pred_boxes):
                color = (0, 0, 255) if i in human_hand_preds_idx else (0, 255, 0)
                cv2.rectangle(
                    mask_drawed_img, 
                    (int(box[0]), int(box[1])), 
                    (int(box[2]), int(box[3])), 
                    color,
                    2
                )

            im_H, im_W = image_shape = rgb.shape[:2]  # h, w
            coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
            depth_m = depth_m[:, :, np.newaxis]

            for bbox in piano_bboxes:
                x1, y1, x2, y2 = [int(x) for x in bbox]
                bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
                bw = max(x2 - x1, 1)
                bh = max(y2 - y1, 1)
                scale = max(bh, bw) * dzi_pad_scale
                scale = min(scale, max(im_H, im_W)) * 1.0

                roi_wh = np.array([bw, bh], dtype=np.float32)
                resize_ratio = out_res / scale

                roi_img = crop_resize_by_warp_affine(
                    rgb, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)

                roi_img = normalize_image(pixel_mean, pixel_std, roi_img)

                resize_ratio = out_res / scale

                depth_img2 = crop_resize_by_warp_affine(
                    depth_m, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
                )
                depth_img2 = depth_img2[:, :, np.newaxis]

                ymap = np.array([[j for i in range(cols)]
                                for j in range(rows)]).astype(np.float32)
                xmap = np.array([[i for i in range(cols)]
                                for j in range(rows)]).astype(np.float32)
            
                H = my_warp_affine(coord_2d, bbox_center, scale,
                    input_res, interpolation=cv2.INTER_LINEAR)
                offset_matrix = np.zeros((3, 3))
                offset_matrix[:2, :] = H
                offset_matrix[2][2] = 1

                depth_img2 = depth_img2 / resize_ratio
                newCameraK = np.matmul(offset_matrix, K)

                cam_cx = newCameraK[0][2]
                cam_cy = newCameraK[1][2]
                cam_fx = newCameraK[0][0]
                cam_fy = newCameraK[1][1]
                xmap_masked = xmap[:, :, np.newaxis]
                ymap_masked = ymap[:, :, np.newaxis]
                pt2 = depth_img2.astype(np.float32)
                pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
                depth_xyz = np.concatenate(
                    (pt0, pt1, pt2), axis=2).transpose(2, 0, 1)

                roi_img = np.concatenate((roi_img, depth_xyz), axis=0)

                roi_coord_2d = crop_resize_by_warp_affine(
                    coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
                ).transpose(2, 0, 1)  # HWC -> CHW

                roi_coord_2d = np.concatenate((depth_xyz[:, ::4, ::4], roi_coord_2d)).astype("float32")

                roi_img = np.array([roi_img], dtype=np.float32)
                roi_cam = np.array([K], dtype=np.float32)
                roi_wh = np.array([roi_wh], dtype=np.float32)
                roi_center = np.array([bbox_center], dtype=np.float32)
                roi_coord_2d = np.array([roi_coord_2d], dtype=np.float32)
                fps_p = np.array([fps_points], dtype=np.float32)

                result = trt_infer.infer(
                    roi_img, 
                    roi_cam, 
                    roi_wh, 
                    roi_center,
                    roi_coord_2d,
                    fps_p
                )
                pred_t = result["translation"][0]
                pred_R = result["rotation"][0]

                pred_Rs_in_frame.append(pred_R)
                pred_ts_in_frame.append(pred_t)

        for pred_R, pred_t in zip(pred_Rs_in_frame, pred_ts_in_frame):
            print(K)
            kpts_2d = misc.project_pts(kpts_3d, K, pred_R, pred_t)
            sixd_drawd_img = misc.draw_projected_box3d(sixd_drawd_img, kpts_2d)

        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        if timestamp % args.save_every_k_frame == 0 and args.save_every_k_frame > 0:
            pass
            
        vis_img = np.concatenate([mask_drawed_img, sixd_drawd_img], axis=1)
        cv2.putText(vis_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", vis_img)   

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        timestamp += 1

        time.sleep(0.01)
