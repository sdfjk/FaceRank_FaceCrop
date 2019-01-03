#coding=utf-8
"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import detect_face
import matplotlib.pyplot as plt
import cv2
import math
from scipy import ndimage
from sklearn.externals import joblib


# def main(args):
def main():

    detect_image_folder = "./TestImg/"
    align_image_save_folder = "./TestImg_save/"
    detect_error_folder = "./TestImg_error/"

    image_size = 160
    margin = 44
    gpu_memory_fraction = 1.0

    load_and_align_data(detect_image_folder, align_image_save_folder, detect_error_folder, image_size, margin, gpu_memory_fraction)


def load_and_align_data(image_folder, save_folder, error_folder, image_size, margin, gpu_memory_fraction):
    # minsize = 20  # minimum size of face
    minsize = 60
    # threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    threshold = [0.5, 0.7, 0.9]
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    # result_path = "error_file_list_v2.txt"
    # write_file = open(result_path, "a+")
    # write_file.close()

    rote_th = 18
    eye_face_rate_th = 0.39
    # similarity_th = 0.96
    similarity_th = 0.945

    img_total_cnt = 0
    img_read_error = 0
    img_detect_error = 0
    img_detect_success = 0

    pathDir = os.listdir(image_folder)
    for allDir in pathDir:
        image_path = os.path.join('%s%s' % (image_folder, allDir))
        save_path = os.path.join('%s%s' % (save_folder, allDir))
        error_path = os.path.join('%s%s' % (error_folder, allDir))
        img_total_cnt += 1

        if img_total_cnt%50==0:
            print("-----------------------------------------------------")
            print("img_total_cnt:   %d" % img_total_cnt)
            print("img_read_error:   %d" % img_read_error)
            print("img_detect_error:   %d" % img_detect_error)
            print("img_detect_success:   %d" % img_detect_success)

        try:
            img_src = misc.imread(os.path.expanduser(image_path), mode='RGB')
            img_size = np.asarray(img_src.shape)[0:2]
            img_h = img_src.shape[0]
            img_w = img_src.shape[1]
        except:
            img_read_error += 1
            print("read image error, image_path:  %s" % (image_path))
            continue

        detect_flag, max_bounding, max_landmark, feature_points = detect_face.detect_max_face(img_src, minsize, pnet, rnet, onet, threshold, factor)

        rote_flag = 255
        if detect_flag == -1:
            # print("--- detect face error, path is %s"%image_path)
            # continue
            rote_flag = 1
        else:
            left_eye = feature_points["left_eye"]
            right_eye = feature_points["right_eye"]
            nose = feature_points["nose"]
            mouth_left = feature_points["mouth_left"]
            mouth_right = feature_points["mouth_right"]
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = math.atan2(dy, dx) * 180.0 / math.pi
            angle_pos = abs(angle)

            max_boxes = np.squeeze(max_bounding[0:4])
            eye_distance = np.sqrt(np.sum(np.square(np.subtract(left_eye[:], right_eye[:]))))
            face_width = max_boxes[2] - max_boxes[0]
            face_height = max_boxes[3] - max_boxes[1]
            eye_face_rate = eye_distance / face_width

            img_rote = ndimage.rotate(img_src, angle)
            crop_flag, crop_face = detect_face.face_crop(img_rote, minsize, pnet, rnet, onet, threshold, factor,
                                                         angle, margin, image_size, save_path)
            if crop_flag==-1:
                rote_flag = 1

        if rote_flag != 1:
            img_detect_success += 1
            resize_crop_face = misc.imresize(crop_face, (224, 224), interp='bilinear')
            misc.imsave(save_path, resize_crop_face)
            continue
        else:
            rote_flag = 255

    print("===========================================================")
    print("img_total_cnt:   %d" % img_total_cnt)
    print("img_read_error:   %d" % img_read_error)
    print("img_detect_error:   %d" % img_detect_error)
    print("img_detect_success:   %d" % img_detect_success)


    return 0


if __name__ == '__main__':
    main()
