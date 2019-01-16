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

    detect_image_folder = "E:/dataset/live_img_20190114/"#"E:/dataset/20190104 dataset/source_images/K_12356_20180925/" #'./TestImg/'#
    align_image_save_folder = "E:/dataset/live_img_20190114_crop/" #"./TestImg_save/"#"E:/dataset/20190104 dataset/K_12356_20180925_cropped20190104/"
    detect_error_folder = "E:/dataset/live_img_20190114_crop_error/" #"./TestImg_error/"

    image_size = 224 #160
    margin = 44
    gpu_memory_fraction = 1.0

    load_and_align_data(detect_image_folder, align_image_save_folder, detect_error_folder, image_size, margin, gpu_memory_fraction)


def load_and_align_data(image_folder, save_folder, error_folder, image_size, margin, gpu_memory_fraction):
    # minsize = 20  # minimum size of face
    minsize = 60
    # threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    threshold = [0.5, 0.7, 0.9] #[0.5, 0.7, 0.9]
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
    similarity_th = 0.97 #0.945

    img_total_cnt = 0
    img_read_error = 0
    img_detect_error = 0
    img_detect_success = 0

    subfolder = os.listdir(image_folder)
    pathDir = []
    for i in subfolder:
        subpath = os.path.join('%s%s' % (image_folder, i))
        if os.path.isdir(subpath):
            pathDir.extend([i + '/' + j for j in os.listdir(subpath)])
            save_subholder = os.path.join('%s%s' % (save_folder, i))
            if not os.path.exists(save_subholder):
                os.makedirs(save_subholder)
        elif '.jpg' in subpath:
            pathDir.append(i)
    print(len(pathDir))

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
            # img_src = misc.imread(os.path.expanduser(image_path), mode='RGB')
            img_src = accertain_face(image_path)
            img_size = np.asarray(img_src.shape)[0:2]
            img_h = img_src.shape[0]
            img_w = img_src.shape[1]

        except:
            img_read_error += 1
            print("read image error, image_path:  %s" % (image_path))
            continue

        detect_flag, max_bounding, max_landmark, feature_points = detect_face.detect_max_face(img_src, minsize, pnet, rnet, onet, threshold, factor, similarity_th)

        rote_flag = 255
        if detect_flag == -1:
            print("--- detect face error, path is %s"%image_path)
            # continue
            misc.imsave(os.path.join(error_folder, image_path.split('/')[-1]), img_src)
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

            img_rote = ndimage.rotate(img_src, angle, cval=255)
            crop_flag, crop_face = detect_face.face_crop(img_rote, minsize, pnet, rnet, onet, threshold, factor,
                                                         angle, margin, image_size, save_path)


            # plt.subplot(221)
            # plt.imshow(img_src)
            # plt.title('img_src')
            # plt.axis('off')
            # plt.subplot(222)
            # plt.imshow(img_rote)
            # plt.title('img_rote')
            # plt.axis('off')
            # plt.subplot(223)
            # plt.imshow(crop_face)
            # plt.title('crop_face')
            # plt.axis('off')
            # plt.show()

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

def accertain_face(image_path):
    """
    通过边缘检测方法检测图像中间部分是否有直线段，确定是否正在才艺PK
    :图片路径 image_path:
    : 50 150分别为检测的高低阈值，3是滤波器核的大小
    :存在直线段，则返回左边半部分的图像输入到后续网络
    """
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')#
    height = img.shape[0]  # 高度
    width = img.shape[1]  # 宽度
    cut_img = img[:, int(width/2 - 2):int(width/2 + 2)]
    gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    if lines is not None:
        return img[:, :320]
    else:
        return img

if __name__ == '__main__':
    main()
