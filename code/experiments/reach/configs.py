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
    COMPUTE_TCN = False
    T = 25
    WEIGHTS_FILE_PATH_MRCNN = '/home/msieb/projects/Mask_RCNN/experiments/four_objects/training_logs/bullet20181009T1257/mask_rcnn_bullet_0023.h5'
###################### DEMO CONFIG ######################

class Demo_Config(object):#    
    T = 30
    SELECTED_VIEW = 0 # only record features of one view

    DEMO_DIR = '/home/msieb/projects/gps-lfd/demo_data'
    DEMO_NAME = 'cube'
    SEQNAME = '0' # FOR PILQR


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
