#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : video_demo.py
#   Author      : YunYang1994
#   Created date: 2018-11-30 15:56:37
#   Description :
#
#================================================================

import cv2
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import pyrealsense2 as rs

IMAGE_H, IMAGE_W = 416, 416
video_path = "./data/demo_data/road.mp4"
video_path = 0 # use camera
classes = utils.read_coco_names('./data/train/train.names')
num_classes = len(classes)
input_tensor, output_tensors = utils.read_pb_return_tensors(tf.get_default_graph(),
                                                            "./checkpoint/yolov3_cpu_nms.pb",
                                                            ["Placeholder:0", "concat_9:0", "mul_6:0"])
points = rs.points()
pipeline= rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

def play(label):
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture('penguin.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video  file")
    # Read until video is completed
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
      # Break the loop
      else:
        break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

with tf.Session() as sess:
    # vid = cv2.VideoCapture(video_path)
    while True:
        # return_value, frame = vid.read()
        # if return_value:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     image = Image.fromarray(frame)
        # else:
        #     raise ValueError("No image!")
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        img_resized = np.array(image.resize(size=(IMAGE_H, IMAGE_W)), dtype=np.float32)
        img_resized = img_resized / 255.
        prev_time = time.time()

        boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.4, iou_thresh=0.5)
        image = utils.draw_boxes(image, boxes, scores, labels, classes, (IMAGE_H, IMAGE_W), show=False)
        if labels is not None:
            for label in labels:
                play(label)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
