#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : quick_train.py
#   Author      : Wanfang
#   Created date: 2019-05-10 14:46:26
#   Description :
#
#================================================================

from CreateSamples import createSamples
from core.convert_tfrecord import image2tfrecord
from kmeans import compute_anchors
# import pyrealsense2 as rs
# import numpy as np
# import cv2
#
# points = rs.points()
# pipeline= rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# profile = pipeline.start(config)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# align_to = rs.stream.color
# align = rs.align(align_to)
# frames = pipeline.wait_for_frames()
# aligned_frames = align.process(frames)
# aligned_depth_frame = aligned_frames.get_depth_frame()
# color_frame = aligned_frames.get_color_frame()
# image = np.asanyarray(color_frame.get_data())
# cv2.imwrite("./data/original_template/image_0.png",image)

# generate new images with image augmentation
outputfolder = createSamples('Parameters.config')

# generate tfrecord file for training which is mush faster than image files
image2tfrecord('./data/train/TrainingData', './data/train/train')

# compute anchors
compute_anchors('./data/train/train.txt', cluster_num = 9)
