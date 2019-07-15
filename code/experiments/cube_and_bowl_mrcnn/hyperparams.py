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

from gps import __file__ as gps_filepath
# from gps.agent.bullet.agent_bullet import AgentBullet
from gps.agent.bullet.bullet_utils import pixel_normalize

from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, IMAGE_FEATURE, OBJECT_POSE, ANCHOR_OBJECT_POSE
from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY

from pdb import set_trace

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import Policy_Training_Config, Demo_Config, Inference_Camera_Config
ptconf, dconf, camconf = Policy_Training_Config(), Demo_Config(), Inference_Camera_Config()

### Training Parameters ###
SEQNAME = dconf.SEQNAME
EXP_NAME = os.path.realpath(__file__).split('/')[-2]
FPS = ptconf.FPS

### Demo Parameters ###
DEMO_NAME = dconf.DEMO_NAME
DEMO_PATH = join(dconf.DEMO_DIR, DEMO_NAME)

### Setup Paths ###
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = join(ptconf.EXP_DIR, EXP_NAME)
DATA_DIR =  join(ptconf.GPS_EXP_DIR, EXP_NAME)
plt.ioff()
####################

print("loading demo sequence {} from {}".format(SEQNAME, DEMO_PATH))
print("Loading model ",ptconf.MODEL_PATH)

sys.path.append('../')
module = importlib.import_module(EXP_NAME +'.agent_bullet')
AgentClass = getattr(module, 'AgentBullet')
####################

SENSOR_DIMS = {
    JOINT_ANGLES: 10,
    JOINT_VELOCITIES: 10,
    END_EFFECTOR_POINTS: 3,
    OBJECT_POSE: 7,
    ANCHOR_OBJECT_POSE: 7,
    #END_EFFECTOR_POINT_VELOCITIES: 6,
    IMAGE_FEATURE: 32, # change line  71 as well to adjust dimension of x0
    ACTION: 4,
}

GP_GAINS = np.ones(SENSOR_DIMS[ACTION])
GP_GAINS[:3] /= 1500
GP_GAINS[-1] = 700



common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + '/data_files/' + '{date:%Y-%m-%d_%H-%M-%S}/'.format(date=datetime.now()),
    'target_filename': EXP_DIR + '/target.npz',
    'log_filename': EXP_DIR + '/log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])
copy2(os.path.realpath(__file__).strip('.pyc') + '.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/agent_bullet.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/configs.py', common['data_files_dir'])

with open('{}/{}_init_object_poses.json'.format(DEMO_PATH, SEQNAME), 'r') as f:
        init_object_poses = json.load(f)
with open('{}/{}_relevant_ids_names.json'.format(DEMO_PATH, SEQNAME), 'r') as f:
        relevant_ids_names = json.load(f)
with open('{}/{}_objects_centroid_mrcnn.json'.format(DEMO_PATH, SEQNAME), 'r') as f:
    objects_centroids_mrcnn = json.load(f)



agent = {
    'ptconf': ptconf,
    'dconf': dconf,
    'camconf': camconf,
    'env': 'bowlenv1',
    'type': AgentClass,
    'filename': '/home/arpit/pwieler/project_mujoco/gps/mjc_models/pr2_arm3d.xml',
    'dt': 0.2,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0.2, 0])]],
    'T': ptconf.T,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, OBJECT_POSE],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, OBJECT_POSE],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': False,
    'object_type': "duck_vhacd.urdf",
    'delta_taskspace': 0.04,
    'data_files_dir': common['data_files_dir'],
    'control_type': ptconf.CONTROL_TYPE, # task or joint
    'reset_condition': init_object_poses, 
    'relevant_ids_names': relevant_ids_names,
    'objects_centroids_mrcnn': objects_centroids_mrcnn,
    'bullet_gui_on': False,
    'plotting_on': False,
    'gps_gui_on': False,
}
agent['x0'] = np.zeros(int(np.sum([val for key, val in SENSOR_DIMS.items() if key in agent['state_include']])))

algorithm = {
    'type': AlgorithmTrajOptPILQR,
    'conditions': common['conditions'],
    'iterations': 10,
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    'kl_step': np.linspace(0.6, 0.2, agent['T']),

    # new, because those from config.py do not apply for variable horizon
    'max_step_mult': np.linspace(10.0, 5.0, agent['T']),
    'min_step_mult': np.linspace(0.01, 0.5, agent['T']),
    #'max_mult': np.linspace(5.0, 2.0, agent['T']), # sind bei Max nicht drin
    #'min_mult': np.linspace(0.1, 0.5, agent['T']), # sind bei Max nicht drin!
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': 1.0 / GP_GAINS ,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 10,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    # 'final_weight': 50,
    'dt': agent['dt'],
    'T': agent['T'],
}



# Load objects MRCNN data

cube_centroid = np.asarray(objects_centroids_mrcnn[relevant_ids_names["cube"]])[20:]
bowl_centroid = np.asarray(objects_centroids_mrcnn[relevant_ids_names["bowl"]])[20:]
# Specify cost targets and functions
cube_centroid_normalized = pixel_normalize(cube_centroid, max_x=camconf.IMG_W, max_y=camconf.IMG_H, max_z=255)
bowl_centroid_normalized = pixel_normalize(bowl_centroid, max_x=camconf.IMG_W, max_y=camconf.IMG_H, max_z=255)

cost_tgt = np.zeros((ptconf.T, cube_centroid.shape[1] + 4))
for tt in range(ptconf.T):
    step_size = int(np.floor(1.0*cube_centroid.shape[0] / ptconf.T))
    cost_tgt[tt, :3] = cube_centroid_normalized[tt*step_size] - bowl_centroid_normalized[tt*step_size]
    if tt == 0:
        plot_imgs = mrcnn_imgs[tt]
    else:
        plot_imgs = np.hstack([plot_imgs, mrcnn_imgs[tt*step_size]])
# plt.imshow(plot_imgs)
# plt.show()
# set_trace()
agent['debug_cost_tgt'] = cost_tgt

# print(cost_tgt[0])
# cost_tgt[1:] = cost_tgt[0]
cost_wt = np.ones(SENSOR_DIMS[OBJECT_POSE]) * 10
cost_wt[3:] = 0.0
cost_wt[1] *= 1
cost_wt[2] *= 1
cost_wt[2] *= 2
# cost_wt[2] = 3.0
# cost_wt[1] = 2.0
state_cost_1 = {
    'type': CostState,
    'l1': 1.0,
    'l2': 0.001,
    'alpha': 1e-6,
    'data_types': {
        OBJECT_POSE: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
}

cost_tgt = np.zeros((ptconf.T, bowl_centroid.shape[1] + 4))
for tt in range(ptconf.T):
    step_size = int(bowl_centroid.shape[0] / ptconf.T)
    cost_tgt[tt, :3] = bowl_centroid_normalized[0]
# print(cost_tgt)
cost_wt = np.ones(SENSOR_DIMS[ANCHOR_OBJECT_POSE]) /100
cost_wt[3:] = 0.0
state_cost_2 = {
    'type': CostState,
    'l1': 1.0,
    'l2': 0.001,
    'alpha': 1e-6,
    'data_types': {
        OBJECT_POSE: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
}

fk_cost_emb = {
    'type': CostState,
    'l1': 0.5,
    'l2': 1.0,
    'alpha': 1e-5,
    'data_types': {
        OBJECT_POSE: {
            'target_state': cost_tgt,
            'wp': cost_wt,
        },
    },
    'ramp_option': RAMP_FINAL_ONLY,
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / GP_GAINS,
}


algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost_1, state_cost_2, fk_cost_emb],
    'weights': [1.0, 0.0, 1.0],
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
    'num_samples': 20,
    'verbose_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': agent['gps_gui_on'],
    'algorithm': algorithm,
    'random_seed': 2,
}

common['info'] = generate_experiment_info(config)


