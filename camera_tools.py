import depthai as dai
import numpy as np
import cv2
import time

def initialize_oak_cam():
    pipeline_oak = dai.Pipeline()

    cam_rgb = pipeline_oak.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)

    mono_left = pipeline_oak.create(dai.node.MonoCamera)
    mono_right = pipeline_oak.create(dai.node.MonoCamera)
    stereo = pipeline_oak.create(dai.node.StereoDepth)

    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline_oak.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline_oak.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline_oak

def stream_oak(pipeline_oak, oak_frame_queue, mxid=None):
    if mxid is not None:
        device_info = dai.DeviceInfo(mxid) 
        device_oak = dai.Device(pipeline_oak, device_info)
    else:
        device_oak = dai.Device(pipeline_oak)
    rgb_queue_oak = device_oak.getOutputQueue(name="rgb", maxSize=2, blocking=False)
    depth_queue_oak = device_oak.getOutputQueue(name="depth", maxSize=2, blocking=False)

    while True:
        rgb_frame_oak = rgb_queue_oak.get()
        depth_frame_oak = depth_queue_oak.get()

        frame_oak = rgb_frame_oak.getCvFrame()
        depth_oak = depth_frame_oak.getFrame()

        oak_frame_queue.put((frame_oak, depth_oak))
        #oak_frame_queue.put(frame_oak)

        if oak_frame_queue.qsize() > 1:
            oak_frame_queue.get()

        time.sleep(0.001)
