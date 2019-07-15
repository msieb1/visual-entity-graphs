#!/usr/bin/env python
import argparse
import copy
import sys
import time
import inspect

import geometry_msgs
from geometry_msgs.msg import (
                                PoseStamped,
                                Pose,
                                Point,
                                Quaternion,
                                )

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from pdb import set_trace
import rospy
import roslib
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import Float32MultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

class img_subscriber(object):
    def __init__(self, topic="/camera/color/image_raw"):
        self.image_sub = rospy.Subscriber(topic, Image, self._callback, queue_size=3)
        self.bridge = CvBridge()
        
    def _callback(self,data):       
        try:
            # tmp self.bridge.imgmsg_to_cv2(data, "bgr8")
            tmp = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")   
            
        except CvBridgeError as e:
            print(e)
        # switch channels from BGR to RGB  
        # self.img = tmp.copy() 
        self.img = tmp.copy()[:,:,:]

class depth_subscriber(object):
    def __init__(self, topic="/camera/depth/image_raw"):
        self.image_sub = rospy.Subscriber(topic, Image, self._callback, queue_size=3)
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

class array_subscriber(object):
    def __init__(self, topic="/detector/confidence"):
        self.array_sub = rospy.Subscriber(topic, numpy_msg(Floats), self._callback, queue_size=5)

    def _callback(self, data):
        try:
            tmp = data.data
            tmp2 = data
        except:
            print "could not get confidence subscriber data"
        # self.array = np.array(tmp).reshape([8, 3])
        self.array = tmp.reshape(8,3)

class pc_subscriber(object):
    def __init__(self, width, height, sel=None, topic="/camera/depth/color/points"):
        self.array_sub = rospy.Subscriber(topic, PointCloud2, self._callback, queue_size=1)
        self.width = width
        self.height = height
        self.sel = sel

    def _callback(self, data):
        registered_pc = []
        sel_points = []
        try:
            tmp = pc2.read_points(data, skip_nans=False, field_names=("x", "y", "z"))#, uvs=[np.arange(self.height), np.arange(self.width) ])
            for p in tmp:
                registered_pc.append(p)
            if len(self.sel) >0:
                tmp = pc2.read_points(data, skip_nans=False, field_names=("x", "y", "z"), uvs=self.sel)
                for p in tmp:
                    sel_points.append(p)
            else:
                sel_points = None
        except:
            print("Error in point cloud reading")
        self.registered_pc = registered_pc
        self.sel_points = sel_points
        # self.pc = tmp