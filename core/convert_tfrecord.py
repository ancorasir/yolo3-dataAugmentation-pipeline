#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : YunYang1994
#   Created date: 2018-12-18 12:34:23
#   Description :
#
#================================================================

import sys
import argparse
import numpy as np
import tensorflow as tf
import glob, re
import cv2

#prepare data.txt: image_file_path boundingbox(top_left, bottom_right) class
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def image2tfrecord(outputfolder='./data/train/TrainingData',tfrecord_path_prefix='./data/train/train'):
    all_images = sorted(glob.glob(outputfolder + "/*.jpg"), key=numericalSort)
    all_labels = sorted(glob.glob(outputfolder + "/*.txt"), key=numericalSort)
    train_file = open("./data/train/train.txt", "w")

    for i in range(len(all_images)):
        label = all_images[i][32:33]
        image = cv2.imread(all_images[i])
        h,w = image.shape[:2]
        lines = open(all_labels[i]).readlines()
        u,v,du,dv = lines[0][1:].split(' ')[1:]
        u = float(u)*w - float(du)*w/2
        v = float(v)*h - float(dv)*h/2
        uu = u + float(du)*w
        vv = v + float(dv)*h
        train_file.write(all_images[i]+ " %s %s %s %s"%(int(u),int(v),int(uu),int(vv)) + " " + label+ "\n")

    train_file.close()

    dataset_txt = './data/train/train.txt'
    dataset = {}
    with open(dataset_txt,'r') as f:
        for line in f.readlines():
            example = line.split(' ')
            image_path = example[0]
            boxes_num = len(example[1:]) // 5
            boxes = np.zeros([boxes_num, 5], dtype=np.float32)
            for i in range(boxes_num):
                boxes[i] = example[1+i*5:6+i*5]
            dataset[image_path] = boxes

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" %images_num)

    tfrecord_file = tfrecord_path_prefix+".tfrecords"
    with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
        for i in range(images_num):
            image = tf.gfile.FastGFile(image_paths[i], 'rb').read()
            boxes = dataset[image_paths[i]]
            boxes = boxes.tostring()
            example = tf.train.Example(features = tf.train.Features(
                feature={
                    'image' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'boxes' :tf.train.Feature(bytes_list = tf.train.BytesList(value = [boxes])),
                }
            ))
            record_writer.write(example.SerializeToString())
        print(">> Saving %d images in %s" %(images_num, tfrecord_file))
