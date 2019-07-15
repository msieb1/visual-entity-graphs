import os
from os.path import join
import numpy as np
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2])) # python folder
sys.path.append(join('../', 'general-utils'))
from rot_utils import eulerAnglesToRotationMatrix, geodesic_dist_quat
from pdb import set_trace
from pyquaternion import Quaternion
import math

###################### CONFIG ######################
class Config(object):
    # Paths
    # Baxter machine
    EXP_DIR = '/media/msieb/data/tcn_data/experiments'
    #GPS_EXP_DIR = '/media/msieb/data/gps_data/experiments/'
    GPS_EXP_DIR = '/home/msieb/projects/gps-lfd/experiments'
    HOME_PATH = "/home/msieb"
    TCN_PATH = '/home/msieb/projects/LTCN'
    EXP_NAME = 'cube_and_bowl'


    # Training specific
    USE_CUDA = False
    NUM_VIEWS = 4
    MODE = 'test'

     # EMBEDDING_DIM = 42 # With depth
    EMBEDDING_DIM = 32 # Only RGB
    T = 40
    IMAGE_SIZE= (299, 299)
    FPS = 10
    N_PREV_FRAMES = 3

###################### TRAINING CONFIG ######################


class Policy_Training_Config(Config):
    CONTROL_TYPE = "task"
    EXP_DIR = '/home/msieb/projects/gps-lfd/experiments'
    GPS_EXP_DIR = '/home/msieb/projects/gps-lfd/experiments'
    MODE = 'train'
    FPS = 3
    COMPUTE_TCN = True
    T = 25
    MODEL_FOLDER = 'view-inv' 
    MODEL_NAME ='2018-09-26-13-57-51/tcn-epoch-10.pk'
    MODEL_PATH = '/media/msieb/data/tcn_data/experiments/duck_multiview_random/trained_models/view-inv-quat-single/2018-10-04-14-48-54/tcn-epoch-40.pk'
    WEIGHTS_FILE_PATH_MRCNN = '../../python/Mask_RCNN/weights/mask_rcnn_bullet_0023.h5'

    ACTION_DIM = 4
###################### DEMO CONFIG ######################

class Demo_Config(Config):#    
    OBJECT_TYPE = "duck_vhacd.urdf"
    CONTROL_TYPE = "joint"
    EXP_NAME = 'duck_multiview_camera'    
    FPS = 3
    T = 61
    EMBEDDING_DIM = 32 # For direct pose predictor
    DEMO_NAME = 'bowl'
    TRAJECTORY_PICKUP = None    
    SELECTED_VIEW = 0 # only record features of one view
    MODEL_FOLDER = 'view-inv' 
    MODEL_NAME ='2018-09-26-13-57-51/tcn-epoch-10.pk'
    MODEL_PATH = '/media/msieb/data/tcn_data/experiments/duck_multiview_camera/trained_models/view-inv-quat-double/2018-10-03-08-30-45/tcn-epoch-40.pk'

    ACTION_DIM = 4
    SELECTED_SEQ_FOR_FEATURE_COMPUTATION = '4'
    DEMO_DIR = '/home/msieb/projects/gps-lfd/demo_data'
    DEMO_NAME = 'bowl'
    SEQNAME = '16' # FOR PILQR

###################### CAMERA CONFIG ######################
class Camera_Config(object):
    IMG_H = 240
    IMG_W = 240
    DISTANCE = 1.5
    VIEW_PARAMS = []
    VIEW_PARAMS.append( # Use this view for demonstrations
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': -91, 
                'pitch': -48, 
                'roll': 0.0, 
                'upAxisIndex': 2
        }
        )    
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 270.6, 
                'pitch': -27, 
                'roll': 50.0, 
                'upAxisIndex': 2,
        }
        )
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 170, 
                'pitch':   -31, 
                'roll':0.0, 
                'upAxisIndex': 2

        }
        )
    VIEW_PARAMS.append(
        {       'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': 360, 
                'pitch':   -31, 
                'roll':0.0, 
                'upAxisIndex': 2
        }
        )
    PROJ_PARAMS = {
            'nearPlane':  0.01,
            'farPlane':  100,
            'fov':  8,
            'aspect':  IMG_W / IMG_H,
        }
    ROT_MATRICES = []
    for cam in VIEW_PARAMS:
        euler_angles = [cam['yaw'], cam['pitch'], cam['roll']]
        ROT_MATRICES.append(eulerAnglesToRotationMatrix(euler_angles))  

class Inference_Camera_Config(Camera_Config):
    TARGET_POSITION = [1.32, -0.33, 0.46]
    NEARVAL = 1
    FARVAL = 3
    DISTANCE = 2.0
    YAW = -90
    PITCH = -30
    ROLL = 0
    VIEW_PARAMS = []
    FOV = 30

    VIEW_PARAMS.append(
            {
                'cameraTargetPosition': TARGET_POSITION,
                'distance': DISTANCE, 
                'yaw': YAW, 
                'pitch': PITCH, 
                'roll': ROLL, 
                'upAxisIndex': 2
            }
            )
    PROJ_PARAMS = {
            'nearPlane':  NEARVAL,
            'farPlane':  FARVAL,
            'fov':  FOV,
            'aspect':  Camera_Config.IMG_W / Camera_Config.IMG_H,
        }

class Multi_Camera_Config(Camera_Config):
    n_cams = 100

    DISTANCE = 1.5
    yaw_low = -220
    yaw_high = 60
    pitch_low = -50
    pitch_high = 0
    roll_low = -120
    roll_high = 120
    VIEW_PARAMS = []
      # yaw_r = np.ones(n_cams,)*-90
    #yaw_r =  np.concatenate([np.linspace(yaw_low, yaw_high, n_cams/2),
    #           np.linspace(yaw_high, yaw_low, n_cams - n_cams/2)])                
    # pitch_r = np.ones(n_cams, ) * -50
    #pitch_r =  np.concatenate([np.linspace(pitch_low, pitch_high/2,n_cams/2),
     #           np.linspace(pitch_high - pitch_high/2, pitch_high, n_cams -n_cams/2)])

    pitch_r = (pitch_high - pitch_low)/2*np.sin(np.linspace(0, 360, n_cams) / 180 * math.pi*-3) +(pitch_high + pitch_low) / 2.0
    roll_r = np.zeros(n_cams,)

    for i in range(n_cams):
        yaw= np.random.randint(yaw_low, yaw_high)
        pitch = np.random.randint(pitch_low,pitch_high)
        roll = roll_r[i]

        VIEW_PARAMS.append(
            {
                'cameraTargetPosition': None,
                'distance': DISTANCE, 
                'yaw': yaw, 
                'pitch': pitch, 
                'roll': roll, 
                'upAxisIndex': 2
            }
            )
    PROJ_PARAMS = {
            'nearPlane':  0.01,
            'farPlane':  100,
            'fov':  8,
            'aspect':  Camera_Config.IMG_W / Camera_Config.IMG_H,
        }
    print(len(VIEW_PARAMS))
    ROT_MATRICES = []
    EULER_ANGLES = []
    ii = 0
    for cam in VIEW_PARAMS:
        euler_angles = [cam['roll'],cam['pitch'], cam['yaw']]
        EULER_ANGLES.append(np.array(euler_angles))
        rot = eulerAnglesToRotationMatrix(np.array(euler_angles)/180*math.pi)
        ROT_MATRICES.append(rot)


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
