from ultralytics import YOLO
from camera_tools import *
import time
import threading
import queue
import argparse
from pathlib import Path
import time
import os

frame_getter_queue = queue.Queue()  
oak_mxid = "18443010613E940F00"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rdpn_cfg', type=str, default="/home/giakhang/dev/rdpn/configs/rdpn.yaml")
    parser.add_argument('--yolo11_ckpt', type=str, default="/home/giakhang/dev/yolov11/runs/segment/piano_2025-02-18_22:08:37/weights/best.engine")
    parser.add_argument('--save_every_k_frame', type=int, default=-1)

    args = parser.parse_args()
    
    segmentator = YOLO(args.yolo11_ckpt, task="segment")

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
        if img_size == depth_size == (1080, 1920):
            new_size = (1920 // 3, 1080 // 3)
            rgb = cv2.resize(rgb, new_size)
            depth = depth.astype(np.int64)
            depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
            depth_m = depth / 1e3  # mm to m
            depth_m[(depth_m<0.001) | (depth_m>=np.inf)] = 0
        
        vis_img = rgb.copy()

        detection_rs = segmentator(rgb)
        detection_rs = detection_rs[0]
        if detection_rs.masks is not None: 
            orig_shape = detection_rs.orig_shape
            for ins_idx in range(len(detection_rs.masks.data)):
                mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
                xy_polygon = detection_rs.masks.xy[ins_idx]
                polygon = xy_polygon.astype(np.int32).reshape(-1, 1, 2)
                mask = cv2.fillPoly(mask, [polygon], 1)
                vis_img = cv2.bitwise_and(vis_img, vis_img, mask=mask)

        frame_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        if timestamp % args.save_every_k_frame == 0 and args.save_every_k_frame > 0:
            pass

        cv2.putText(vis_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Frame", vis_img)   

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        timestamp += 1

        time.sleep(0.01)
