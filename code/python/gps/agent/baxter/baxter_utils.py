#!/usr/bin/env python
import argparse
import copy
import sys
import time
import inspect
import os
import sys
import imageio

import geometry_msgs
from geometry_msgs.msg import (
                                PoseStamped,
                                Pose,
                                Point,
                                Quaternion,
                                )

from cv_bridge import CvBridge, CvBridgeError
import cv2
import moveit_commander
import moveit_msgs.msg
import numpy as np
from pdb import set_trace
import rospy
import roslib
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import PIL

import torch
from tcn import define_model, define_model_depth


IMAGE_SIZE = (299, 299)

class img_subscriber:
    def __init__(self, topic="/camera/color/image_raw"):
        self.image_sub = rospy.Subscriber(topic, Image, self._callback, queue_size=1)
        self.bridge = CvBridge()
        
    def _callback(self,data):       
        try:
            # tmp self.bridge.imgmsg_to_cv2(data, "bgr8")
            tmp = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")   
            
        except CvBridgeError as e:
            print(e)
        # switch channels from BGR to RGB   
        self.img = tmp.copy()

class depth_subscriber(object):
    def __init__(self, topic="/camera/depth/image_raw"):
        self.image_sub = rospy.Subscriber(topic, Image, self._callback, queue_size=1)
        self.bridge = CvBridge()
        
    def _callback(self,data):       
        try:
            # tmp self.bridge.imgmsg_to_cv2(data, "bgr8")
            tmp = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")   
            
        except CvBridgeError as e:
            print(e)
        # switch channels from BGR to RGB  
        # self.img = tmp.copy() 
        self.img = tmp.copy()

class array_subscriber:
    def __init__(self, topic="/detector/confidence"):
        self.array_sub = rospy.Subscriber(topic, numpy_msg(Floats), self._callback, queue_size=1)

    def _callback(self, data):
        try:
            tmp = data.data
            tmp2 = data
        except:
            print "could not get confidence subscriber data"
        # self.array = np.array(tmp).reshape([8, 3])
        self.array = tmp.reshape(8,3)

def resize_frame(frame, out_size):
    image = PIL.Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)

def save_image_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    imwrite(filepath, file)

def load_model(model_path, depth=False, use_cuda=True):
    if depth:
        tcn = define_model_depth(use_cuda)
    else:
        tcn = define_model(use_cuda)
    tcn = torch.nn.DataParallel(tcn, device_ids=range(torch.cuda.device_count()))

    # tcn = PosNet()
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    tcn.load_state_dict(state_dict)

    if use_cuda:
        tcn = tcn.cuda()
    return tcn

