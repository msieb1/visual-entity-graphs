""" Hyperparameters for peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import json
import sys
from os.path import join
import numpy as np
import importlib
import matplotlib.pyplot as plt
from shutil import copy2
from pdb import set_trace as st
import matplotlib.pylab as pl
import imageio
import pyquaternion as pq
import rospy
import pickle
import tf 
import cv2
import imageio
import math
np.set_printoptions(precision=2)
# GPS imports
from gps import __file__ as gps_filepath
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_visual import CostVisual

from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, IMAGE_FEATURE, OBJECT_POSE, ANCHOR_OBJECT_POSE, TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, RGB_IMAGE, \
        DEPTH_IMAGE

from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY

from gps.utility.utils import pixel_normalize, compute_view_pose_embedding, resize_frame, load_pixel_model, compute_pixel_target_distances
# Baxter imports 
from baxter_pykdl import baxter_kinematics
import baxter_interface

# Config & Custom imports
colors = pl.cm.jet(np.linspace(0,1,81))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/zhouxian/projects/pytorch-dense-correspondence')
sys.path.append('/home/zhouxian/projects/pytorch-dense-correspondence/modules')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import Policy_Training_Config, Demo_Config
ptconf, dconf = Policy_Training_Config(), Demo_Config()

### Training Parameters ###
EXP_NAME = os.path.realpath(__file__).split('/')[-2]
EXP_DIR = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/cvpr2019/gps_data/' + EXP_NAME

### Setup Paths ###
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
sys.path.append('../')
module = importlib.import_module(EXP_NAME +'.agent_baxter')
AgentClass = getattr(module, 'AgentBaxter')
####################
rospy.init_node('gps_agent_ros_node', disable_signals=True)
####################
IMG_HEIGHT = 480
IMG_WIDTH = 640


common = {
    'experiment_name': EXP_NAME,
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + '/data_files/' + '{date:%Y-%m-%d_%H-%M-%S}/'.format(date=datetime.now()),
    'target_filename': EXP_DIR + '/target.npz',
    'log_filename': EXP_DIR + '/log.txt',
    'conditions': 1,
    'trial_arm': 'left',
    'aux_arm': 'right',
}

kin_aux = baxter_kinematics(common['aux_arm'])
kin_trial = baxter_kinematics(common['trial_arm'])

EE_POSE = np.asarray(kin_trial.forward_position_kinematics())
EE_POINTS = EE_POSE[:3][None, :]
EE_ORIENTATION = EE_POSE[3:][None, :]
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3
## Define sensor and action dimension, set gains
SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0] + 1,
    OBJECT_POSE: 7,
    #END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 4,
} # Note that you can define what sensors to use in 'state_include' and 'obs_include' in the agent config a little further down
GP_GAINS = np.ones(SENSOR_DIMS[ACTION])
GP_GAINS[:3] /= 100
GP_GAINS[-1] = 700
TASKSPACE_DELTAS = np.array([0.05, 0.05, 0.05, 1.8])


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])
copy2(os.path.realpath(__file__).strip('.pyc') + '.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/agent_baxter.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/configs.py', common['data_files_dir'])


agent = {
    'ptconf': ptconf,
    'dconf': dconf,
    'type': AgentClass,
    'dt': 0.5,
    'substeps': 5,
    'conditions': common['conditions'],
    'trial_arm': common['trial_arm'],
    'aux_arm': common['aux_arm'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0.2, 0])]],
    'T': 30,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': False,
    'object_type': "duck_vhacd.urdf",
    'delta_taskspace': [0.05, 0.05], # XYZ and Angle
    'data_files_dir': common['data_files_dir'],
    'control_type': ptconf.CONTROL_TYPE, # task or joint
    'gripper_threshold': 0.105,
    'gripper_reset_position': 0,
    'num_rollouts_per_iteration': 6,
    'gps_gui_on': False,
    'debug': False,
    'reset_init_joints': True ,  # set to True if current joint config should be used as new init for current experiment
    'show_demo_plot': True,
    'cost_tgt': None,
}

# Get initial joint configuration
if not os.path.exists(os.path.join(common['experiment_dir'], 'x0.pkl')) or agent['reset_init_joints']:
    pos0 = list(baxter_interface.Limb('left').endpoint_pose()['position'])
    orn0 = list(baxter_interface.Limb('left').endpoint_pose()['orientation'])
    with open(os.path.join(common['experiment_dir'], 'x0.pkl'), 'wb') as fp:
        pickle.dump([pos0, orn0], fp, -1)        
with open(os.path.join(common['experiment_dir'], 'x0.pkl'), 'rb') as fp:
    x0ee = pickle.load(fp)
pos0 = x0ee[0]
orn0 = x0ee[1]
x0joints = kin_trial.inverse_kinematics(pos0, orn0)
u0 = np.zeros((10, SENSOR_DIMS[ACTION]))
agent['x0'] = np.zeros(int(np.sum([val for key, val in SENSOR_DIMS.items() if key in agent['state_include']])))
agent['x0'][:7] = x0joints # Set first 7 values (joints) to prespecified position (rest pose)
ee_tgts = []
reset_conditions = []
for i in xrange(common['conditions']):


    ee_tgt = [
                0.626410004856,
                -0.0430195859605,
                -0.198833010946
            ]
    ee_tgt_orn = [  
                    -0.3850956351, 
                    0.921535608765, 
                    0.0147111823011, 
                    0.0475084801185
                ]

    reset_condition = {
        TRIAL_ARM: {
            'mode': JOINT_SPACE,
            'data': x0joints,
        },
        AUXILIARY_ARM: {
            'mode': JOINT_SPACE,
            'data': x0joints,
        },
    }
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)
agent['ee_points_tgt'] = ee_tgts
agent['reset_conditions'] = reset_conditions

### Setup Cost functions and such ###
# cost_tgt[1:] = cost_tgt[0]
# cost_tgt = np.array([[0.09, 0.035, 0.32, 0.023, 0.04 , 0.308]] * agent['T']) 
# cost_tgt = np.array([[0.06977645, -0.07197795,  0.46231904,
#                      -0.00092908, -0.07011419,  0.45495265]] * agent['T']) 


#### HAND PROCESSING
# D435 intrinsics matrix
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		0.0, 615.323974609375, 240.33250427246094, 
		0.0, 0.0, 1.0]).reshape(3, 3)

def _deproject_pixel_to_point(p2d, depth_z, intrin):
	p3d = np.zeros(3,)
	p3d[0] = (p2d[0] - intrin[0, 2]) / intrin[0, 0]
	p3d[1] = (p2d[1] - intrin[1, 2]) / intrin[1, 1]
	p3d[2] = 1.0
	p3d *= depth_z
	return p3d

def _project_point_to_pixel(p3d, intrin):
    # returns (width, height) indexed
	p2d = np.zeros(2,)
	p2d[0] = p3d[0] / p3d[2] * intrin[0, 0] + intrin[0, 2]
	p2d[1] = p3d[1] / p3d[2] * intrin[1, 1] + intrin[1, 2]
	return p2d


hand_traj = np.load(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/bottle_in_mug/videos/train/poses/3_view0/right_hand_trajectory.npy'
)
depth = np.load(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/bottle_in_mug/depth/train/3_view0.npy'
)
reader  = imageio.get_reader(
'/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/imitation_learning/bottle_in_mug/videos/train/poses/3_view0/3_view0.mp4'
)
assert len(reader) == len(hand_traj)

if len(depth) > len(hand_traj): # depth for some reason 2 elements longer than hand_trajectory
    depth = depth[:len(hand_traj)]

l_f_traj = hand_traj[:, 4, :]
r_f_traj = hand_traj[:, 8, :]
imgs = []
reader = list(reader)
for i, img in enumerate(reader):
    imgs.append(img)
    mask = np.zeros(depth[i].shape, dtype=np.uint8)
    mask[np.where(depth[i] == 0)] = 1
    depth_inpainted = cv2.inpaint(depth[i],mask,3,cv2.INPAINT_TELEA)
    depth_l = depth_inpainted[int(l_f_traj[i, 1]), int(l_f_traj[i, 0])] * 0.001
    depth_r = depth_inpainted[int(r_f_traj[i, 1]), int(r_f_traj[i, 0])] * 0.001
    l_f_traj[i, 2] = depth_l
    r_f_traj[i, 2] = depth_r

# Truncate front and end if desired
front = 5
end = -30
l_f_traj = l_f_traj[front:end, :]
r_f_traj = r_f_traj[front:end, :]
imgs = imgs[front:end]
# Smooth temporally with savgol filter
window = 61
polyorder = 3
for j in range(3):
    l_f_traj[:, j] = savgol_filter(l_f_traj[:, j], window, polyorder)
    r_f_traj[:, j] = savgol_filter(r_f_traj[:, j], window, polyorder)

# 2D finger trajectories
l_f_2d_traj = []
r_f_2d_traj = []

# Deprojected 3D finger trajectories
dep_l_f_traj = []
dep_r_f_traj = []
dep_f_mean_traj = []
gripper_binary_traj = []


# Read images, plot loaded images and compute cost target
for i in range(agent['T']):
    step_size = int(np.floor(1.0*len(l_f_traj) / agent['T']))
    if i == 0 :
        plot_imgs = imgs[i]
    else:
        plot_imgs = np.hstack([plot_imgs, imgs[i*step_size]])

    l_f_p2d = [np.clip(int(l_f_traj[i*step_size, 1]), 0, IMG_HEIGHT-1), np.clip(int(l_f_traj[i*step_size, 0]), 0, IMG_WIDTH-1)]
    r_f_p2d = [np.clip(int(r_f_traj[i*step_size, 1]), 0, IMG_HEIGHT-1), np.clip(int(r_f_traj[i*step_size, 0]), 0, IMG_WIDTH-1)]
    
    depth_l = l_f_traj[i*step_size, 2]
    depth_r = r_f_traj[i*step_size, 2]

    dep_l_f = _deproject_pixel_to_point([l_f_p2d[0], l_f_p2d[1]], depth_l, INTRIN)[[1,0,2]]
    dep_r_f = _deproject_pixel_to_point([r_f_p2d[0], r_f_p2d[1]], depth_r, INTRIN)[[1,0,2]]  # x indexes column, y indexes row, so order is col - row - z, not row- col -z as usually in numpy

    l_f_2d_traj.append(l_f_p2d)
    r_f_2d_traj.append(r_f_p2d)
    dep_l_f_traj.append(dep_l_f)
    dep_r_f_traj.append(dep_r_f)
    dep_f_mean_traj.append((dep_l_f + dep_r_f) / 2.0)
    gripper_binary_traj.append(int(np.linalg.norm(dep_l_f - dep_r_f) > agent['gripper_threshold']))
    # print(np.linalg.norm(dep_l_f - dep_r_f))
    # print(np.linalg.norm(l_f_traj[i*step_size] - r_f_traj[i*step_size]))
    # print(depth_l, depth_r)
    # print('--')
    # cv2.imshow('1', imgs[i*step_size][:,:,::-1])
    # cv2.waitKey(5)

dep_l_f_traj = np.array(dep_l_f_traj)
dep_r_f_traj = np.array(dep_r_f_traj)
dep_f_mean_traj = np.array(dep_f_mean_traj)
l_f_2d_traj = np.array(l_f_2d_traj)
r_f_2d_traj = np.array(r_f_2d_traj)
f_mean_traj = (l_f_2d_traj + r_f_2d_traj) / 2.0
gripper_binary_traj = np.array(gripper_binary_traj)[..., None]
cost_tgt = np.hstack([dep_f_mean_traj, gripper_binary_traj])
# cost_tgt[:, -2] += 0.1
agent['cost_tgt'] = cost_tgt
col = ['b', 'r']
colors_gripper = [col[0] if cost_tgt[i, -1] == 0 else col[1] for i in range(len(cost_tgt)-1) ] + ['g']
print('\n',cost_tgt)
for tt in range(ptconf.T):

    if tt == 0 :
        plot_imgs = imgs[tt]
    else:
        plot_imgs = np.hstack([plot_imgs, imgs[tt*step_size]])

if agent['show_demo_plot']:
    fig1 = plt.figure(figsize=(20,5))
    plt.imshow(plot_imgs)
    plt.axis('off')
    # Plot in 2D
    fig2 = plt.figure()
    plt.scatter(f_mean_traj[:, 0], f_mean_traj[:, 1],color=colors_gripper)
    # plt.scatter(f_mean_traj,color=colors_gripper)
    plt.plot(f_mean_traj[:, 0], f_mean_traj[:, 1], color='black')
    # plt.figure()
    # plt.plot(dep_f_mean_traj[:,-1], color='black')

    ### Plot 3D Trajectory
    # fig = plt.figure()
    # plt.title('smoothed trajectories')
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dep_f_mean_traj[:, 0], dep_f_mean_traj[:, 1], dep_f_mean_traj[:, 2],color=colors[i])
    # ax.set_xlabel('x label')
    # ax.set_ylabel('y label')
    # ax.set_zlabel('z label')
    # ax.set_xlim(np.min(dep_f_mean_traj[:, 0]), np.max(dep_f_mean_traj[:, 0]))
    # ax.set_ylim(np.min(dep_f_mean_traj[:, 1]), np.max(dep_f_mean_traj[:, 1]))
    # ax.set_zlim(np.min(dep_f_mean_traj[:, 2]), np.max(dep_f_mean_traj[:, 2]))
    # # plt.savefig(save_path)
    # plt.show()
    # plt.close()

    plt.show(); plt.close(fig1); plt.close(fig2)



cost_wt = np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS])  * 1
cost_wt[-1] *= 1
########
fk_cost_emb = {
    'type': CostState,
    'l1': 0.5,
    'l2': 1.0,
    'alpha': 1e-5,
    'data_types': {
        END_EFFECTOR_POINTS: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
    'ramp_option': RAMP_FINAL_ONLY,
}

state_cost_0 = {
    'type': CostState,
    'l1': 1.0,
    'l2': 0.001,
    'alpha': 1e-6,
    'data_types': {
        END_EFFECTOR_POINTS: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
}
torque_wt =  5e-5 / GP_GAINS
torque_wt[-1] *= 0
torque_cost = {
    'type': CostAction,
    'wu': torque_wt,
}
## PILQR

## Configure algorithm's hyperparameters
algorithm = {
    'type': AlgorithmTrajOptPILQR,
    'conditions': common['conditions'],
    'iterations': 20,
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    'kl_step': np.linspace(0.6, 0.2, agent['T']),
    'min_step_mult': np.linspace(10, 5, agent['T']),
    'max_step_mult': np.linspace(0.01, 0.5, agent['T']),
}



algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost_0, torque_cost],
    'weights': [1.0, 1.0],
}


algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  np.ones(SENSOR_DIMS[ACTION]) / GP_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 10.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    'final_weight': 50,
    'dt': agent['dt'],
    'T': agent['T'],
}



algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}


algorithm['traj_opt'] = {
    'type': TrajOptPILQR,
    'kl_threshold': 1.0,
    'covariance_damping': 10.,
    #'min_temperature': 0.0001,
}

algorithm['policy_opt'] = {}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 7,
    'verbose_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': agent['gps_gui_on'],
    'algorithm': algorithm,
    'random_seed':0,
}

common['info'] = generate_experiment_info(config)

## PI2

## Configure algorithm's hyperparameters
# algorithm = {
#     'type': AlgorithmTrajOptPI2,
#     'conditions': common['conditions'],
#     'iterations': 20,
#     'step_rule': 'res_percent',
#     'step_rule_res_ratio_dec': 0.2,
#     'step_rule_res_ratio_inc': 0.05,
#     'kl_step': np.linspace(0.6, 0.2, agent['T']),
#     'min_step_mult': np.linspace(10, 5, agent['T']),
#     'max_step_mult': np.linspace(0.01, 0.5, agent['T']),
# }



# algorithm['cost'] = {
#     'type': CostSum,
#     'costs': [state_cost_0, torque_cost],
#     'weights': [1.0, 1.0],
# }


# algorithm['init_traj_distr'] = {
#     'type': init_lqr,
#     'init_gains':  np.ones(SENSOR_DIMS[ACTION]) / GP_GAINS,
#     'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
#     'init_var': 10.0,
#     'stiffness': 0.5,
#     'stiffness_vel': 0.25,
#     'final_weight': 50,
#     'dt': agent['dt'],
#     'T': agent['T'],
# }



# algorithm['dynamics'] = {
#     'type': DynamicsLRPrior,
#     'regularization': 1e-6,
#     'prior': {
#         'type': DynamicsPriorGMM,
#         'max_clusters': 20,
#         'min_samples_per_cluster': 40,
#         'max_samples': 20,
#     },
# }

# algorithm['traj_opt'] = {
#     'type': TrajOptPI2,
#     'kl_threshold': 1.0,
#     'covariance_damping': 10.,
#     #'min_temperature': 0.0001,
# }

# algorithm['policy_opt'] = {}

# algorithm['policy_prior'] = {
#     'type': PolicyPriorGMM,
#     'max_clusters': 20,
#     'min_samples_per_cluster': 40,
#     'max_samples': 40,
# }

# config = {
#     'iterations': algorithm['iterations'],
#     'num_samples': agent['num_rollouts_per_iteration'],
#     'verbose_trials': 1,
#     'common': common,
#     'agent': agent,
#     'gui_on': agent['gps_gui_on'],
#     'algorithm': algorithm,
#     'random_seed':0,
# }

# common['info'] = generate_experiment_info(config)


