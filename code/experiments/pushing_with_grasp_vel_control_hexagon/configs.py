import os
from os.path import join
import numpy as np
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2])) # python folder
from pdb import set_trace as st
import math



###################### TRAINING CONFIG ######################


class Policy_Training_Config(object):
    CONTROL_TYPE = "task"
    COMPUTE_FEATURES = True
    T = 20
    WEIGHTS_FILE_PATH_MRCNN = '/home/zhouxian/projects/gps-lfd/experiments/baxter/giraffe_pushing_mrcnn/weight_files/mask_rcnn_baxter_0007.h5'
    MODEL_DIR = '/home/zhouxian/projects/gps/experiments/baxter_reaching/data_files'

###################### DEMO CONFIG ######################

class Demo_Config(object):#    
    T = 20
    SELECTED_VIEW = 0 # only record features of one view

    DEMO_DIR = '/home/msieb/projects/gps-lfd/demo_data'
    VIDEO_DIR = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/cvpr2019/video_data'
    DEMO_NAME = 'giraffe'
    SEQNAME = 'rotating_1' # FOR PILQR


###################### TRAJECTORY CONFIG ######################

class Trajectory_Config(Demo_Config):
    ### Task Space
    #agent.step_taskspace([0.1,0.1,0.1,0.1])
    #agent.step_taskspace([0.  ,0. ,0.  ,0.1])
    T = Demo_Config.T
    dx = (np.random.rand(1, T)-0.5).T * 30 + 0.01
    dy = (np.random.rand(1, T)-0.5).T * 30 - 0.02
    dz = (np.random.rand(1, T)-0.5).T * 30 - 0.01
    da = -(np.random.rand(1, T)).T * 11
    dphi = (np.random.rand(1, T)).T * 3
    dtheta = (np.random.rand(1, T)).T * 5  

    RANDOM = np.hstack([dx, dy, dz, da, dphi, dtheta])
    ### No Joint Space trajectory implemented yet in demo agent
