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
import cv2

import pyquaternion as pq
import rospy
import pickle
import tf

# GPS imports
from gps import __file__ as gps_filepath

from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_visual import CostVisual

from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, OBJECT_POSE, ANCHOR_OBJECT_POSE, TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, RGB_IMAGE, \
        DEPTH_IMAGE, IMAGE_FEATURE

from gps.gui.config import generate_experiment_info
from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
from gps.algorithm.traj_opt.traj_opt_pilqr import TrajOptPILQR
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY, RAMP_QUADRATIC, RAMP_LINEAR

# Baxter imports 
from baxter_pykdl import baxter_kinematics
import baxter_interface

# Mask RCNN imports 
sys.path.append('/home/zhouxian/projects/Mask_RCNN/samples')
from baxter.baxter_iccv import BaxterConfig
from mrcnn.config import Config
from mrcnn import visualize, utils
import mrcnn.model as modellib

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50


from gps.utility.utils import pixel_normalize, compute_view_pose_embedding, resize_frame, load_pixel_model, get_sample_points, compute_pixel_centroid_and_rotation_difference, compute_pixel_target_distances, compute_pixel_query_points, grabcut, depth_to_pc, depth_to_mask
from gps.utility.feature_utils import get_2d_depth_finger_trajectories, preprocess_hand_trajectory, get_unprojected_3d_mean_finger_and_gripper_trajectory, get_mrcnn_features, get_unprojected_3d_trajectory
from keras.backend.tensorflow_backend import set_session
import tensorflow


####################
rospy.init_node('gps_agent_ros_node')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tensorflow.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
# config.gpu_options.visible_device_list = "0"
set_session(tensorflow.Session(config=config))
plt.ion()
####################

### CONSTANTS ###
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		0.0, 615.323974609375, 240.33250427246094, 
		0.0, 0.0, 1.0]).reshape(3, 3) # D435 intrinsics matrix
DEPTH_SCALE = 0.001
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_RESIZED_WIDTH = 240
IMAGE_RESIZED_HEIGHT = 240
IMAGE_CHANNELS = 3
TRUNCATE_FRONT = 0   # 30, 70 for picking and placing (square on ring)
TRUNCATE_BACK = 1
################

# Config & Custom imports
colors = pl.cm.jet(np.linspace(0,1,111))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/home/zhouxian/projects/pytorch-dense-correspondence')
sys.path.append('/home/zhouxian/projects/pytorch-dense-correspondence/modules')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import Policy_Training_Config, Demo_Config
ptconf, dconf = Policy_Training_Config(), Demo_Config()

#### PATHS #####5
EXP_NAME = join(os.path.realpath(__file__).split('/')[-2])
VIDEO_DIR = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/demos'
MRCNN_MODEL_DIR = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/synthetic_data_generation/iccv2019_flipping/mask_rcnn_models_selected'
MRCNN_WEIGHTS_FILE = join(MRCNN_MODEL_DIR, 'can_and_mug/mask_rcnn_baxter_0045.h5')

### Demo Parameters ###
DEMO_NAME = 'lacan9_view0'
TASK_NAME = 'pour'
EXP_DIR = join('/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/gps_data/', TASK_NAME, DEMO_NAME)
DEMO_DIR = '/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/demos'
DEMO_PATH = join(DEMO_DIR, TASK_NAME)
T_CAMERA_WORLD_DEMO = np.load(join(VIDEO_DIR, TASK_NAME, 'tf_frame_vals', DEMO_NAME + '.npy'))
print('loading demo from {}'.format(DEMO_PATH))


### Setup Paths ###
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
sys.path.append('../')
module = importlib.import_module(EXP_NAME +'.agent_baxter')
AgentClass = getattr(module, 'AgentBaxter')



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

## Define sensor and action dimension, set gains
SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    OBJECT_POSE: 2 * 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    IMAGE_FEATURE: 2, # change line  71 as well to adjust dimension of x0
    ACTION: 5,
} # Note that you can define what sensors to use in 'state_include' and 'obs_include' in the agent config a little further down
GP_GAINS = np.ones(SENSOR_DIMS[ACTION])
GP_GAINS[:3] /= 800
GP_GAINS[3] /= 50
GP_GAINS[-1] /= 5

# Backup files
if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])
copy2(os.path.realpath(__file__).strip('.pyc') + '.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1])+ '/agent_baxter.py', common['data_files_dir'])
copy2('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/configs.py', common['data_files_dir'])

## Define Hyperparams and load used data files such as demonstrations and trained models


### Visual feature model from  training ###
# TODO: Ask Xian for Path
feature_model = load_pixel_model(basedir='/home/zhouxian/git/pixel-correspondence-imitation/src', use_cuda=True)
# feature_model = None

### Load Mask RCNN model ###
class InferenceConfig(BaxterConfig):
    # Make sure the Inference Config is imported from the correct folder! (line 53)
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()
mrcnn = modellib.MaskRCNN(mode='inference', model_dir=MRCNN_MODEL_DIR,
                            config=inference_config)
mrcnn.load_weights(MRCNN_WEIGHTS_FILE, by_name=True)

# Copy from trained MRCNN config file
class_names = ['BG', 'mugs', 'cans', 'hand', 'robot']

# Choose IDs to be detecyed (doesn't output bboxes of the other classes)
target_ids = [1, 2]    

agent = {
    'ptconf': ptconf,
    'dconf': dconf,
    'type': AgentClass,
    'dt': 0.4,
    'substeps': 5,
    'conditions': common['conditions'],
    'trial_arm': common['trial_arm'],
    'aux_arm': common['aux_arm'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [[np.array([0, 0.2, 0])]],
    'T': 30,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [OBJECT_POSE, JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, IMAGE_FEATURE],
    'obs_include': [OBJECT_POSE, JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, IMAGE_FEATURE],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
    'smooth_noise': False,
    'delta_taskspace': [0.001, 0.05], # XYZ and Angle
    'data_files_dir': common['data_files_dir'],
    'control_type': 'task', # task or joint
    'views': ['2'], # baxter view is 2
    'gripper_threshold': 100.502, #0.17
    'gripper_reset_position': 0, # 100 open, 0 close
    'feature_model': feature_model,
    'feature_fn': compute_pixel_centroid_and_rotation_difference,
    'mrcnn_model': mrcnn,
    'class_names': class_names,
    'target_ids': target_ids,
    'centroids_last_known': {id: None for id in target_ids},
    'masks_last_known': {id: None for id in target_ids},
    'demo_centroids_precomputed': False,
    'anchor_id': 2,
    'object1_id': 2,
    'object2_id': 1,
    'cost_tgt': None,
    'n_max_correspondences': 1000,
    'num_samples': 10,
    'debug': True,
    'all_object_traj': None,
    'all_mask_traj': None,
    'demo_imgs': [],
    'demo_finger_traj': [],
    'u0': None,
    'set_action_to_zero': np.array([1, 0, 1, 1, 1]), # x,y,z,rot,gripper
    'max_velocity': 0.03,
    'max_rot': 0.25,
    'figure_axes': plt.subplots(nrows=4, ncols=2),
    'gps_gui_on': False,
    'visualize_mrcnn': False,
    'recompute_demo_values': False,
    'show_demo_plot': False,
    'reset_init_joints': True,  # set to True if current joint config should be used as new init for current experiment

} 
assert len(agent['set_action_to_zero']) == SENSOR_DIMS[ACTION]

# OR:
# Get initial joint configuration
if not os.path.exists(os.path.join(common['experiment_dir'], 'x0joints.pkl')) or agent['reset_init_joints']:
    trial_limb = common['trial_arm']
    trial_arm = baxter_interface.Limb(trial_limb)
    joint_names = trial_arm.joint_names()
    x0joints = [trial_arm.joint_angle(j) for j in joint_names]
    with open(os.path.join(common['experiment_dir'], 'x0joints.pkl'), 'wb') as fp:
        pickle.dump([x0joints], fp, -1)        
with open(os.path.join(common['experiment_dir'], 'x0joints.pkl'), 'rb') as fp:
    x0joints = pickle.load(fp)
x0joints = np.array(x0joints)

# Get precomputed object centroids #TODO: Fix to load masks for DON as well
if agent['recompute_demo_values'] or not os.path.exists(os.path.join(common['experiment_dir'], 'all_object_traj.pkl')):
    print('cannot find pre-computed mrcnn data - process from scratch')
    agent['demo_centroids_precomputed'] = False
    agent['all_object_traj'] = {id: np.empty((0, 3)) for id in agent['target_ids']}
    agent['all_mask_traj'] = []

else:
    print('\nfound pre-computed mrcnn data')
    agent['demo_centroids_precomputed'] = True
    with open(os.path.join(common['experiment_dir'], 'all_object_traj.pkl'), 'rb') as fp:
        agent['all_object_traj'] = pickle.load(fp)
    with open(os.path.join(common['experiment_dir'], 'all_mask_traj.pkl'), 'rb') as fp:
        agent['all_mask_traj'] = pickle.load(fp)

u0 = np.zeros((10, SENSOR_DIMS[ACTION]))
agent['x0'] = np.zeros(int(np.sum([val for key, val in SENSOR_DIMS.items() if key in agent['state_include']])))
agent['x0joints'] = x0joints # Set first 7 values (joints) to prespecified position (rest pose)
ee_tgts = []
reset_conditions = []
for i in xrange(common['conditions']):
    ee_tgt = [
                0.626410004856,
                -0.0430195859605,
                -0.198833010946
            ]
    ee_tgt_orn = [  
                    -0.385095633851, 
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

##### LOAD DEMONSTRATION DATA ##### (USE THIS IF DEMO IS GIVEN AS MP4 VID FILE)
print('loading videos')

# Load init joint values for camera pose
with open(join(DEMO_PATH, 'baxter_init_vals', DEMO_NAME.split('_view')[0] + '.json'), 'r') as fp:
    agent['aux_arm_joint_vals'] = json.load(fp)['joint_angles'] 
# Load depth images/numpy arrays
depth = np.load(
    join(DEMO_PATH, 'depth', '{}.npy'.format(DEMO_NAME)))
# Load RGB images with hand keypoints overlayed
# reader_w_pose  = imageio.get_reader(
#     join(DEMO_PATH, 'videos/poses', DEMO_NAME, '{}.mp4'.format(DEMO_NAME)))
# Load original RGB images
reader  = imageio.get_reader(
    join(DEMO_PATH, 'videos', '{}.mp4'.format(DEMO_NAME)))
try:
    hand_traj_path = join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_trajectory.npy')
    hand_traj, hand_traj_deproj = preprocess_hand_trajectory(hand_traj_path, join(DEMO_PATH, 'depth', '{}.npy'.format(DEMO_NAME)))
except:
    print('hand trajectory not found')
    hand_traj = hand_traj_deproj = np.zeros((len(depth), 2, 3))
    

assert len(reader) == len(depth)
pose_offset = len(depth) - len(hand_traj) # offset because openpose clips a few frames away at the end
imgs = []
dimgs = []
ii = 0
for img, dimg in zip(reader, depth):
    if img.shape[-1] == 4:
        img = img[:, :, :-1]
    imgs.append(img)
    dimgs.append(dimg)
    ii += 1
print('videos loaded')
##### TRUNCATE DATA #####
#FIXME: Truncate front and end of videos if desired (find better solution to do this)
demo_len = len(hand_traj[TRUNCATE_FRONT:-TRUNCATE_BACK])
step_size = int(np.floor(1.0*(demo_len) / agent['T']))
step_offset = demo_len % agent['T']
step_offset_front = int(step_offset / 2) + step_offset % 2
step_offset_back = int(step_offset / 2)


### Hand Processing ###
# l_f_traj, r_f_traj = get_2d_depth_finger_trajectories(hand_traj, depth)
l_f_traj = hand_traj[:, 0, :]
r_f_traj = hand_traj[:, 1, :]
l_f_traj_orig = hand_traj_deproj[:, 0, :]
r_f_traj_orig = hand_traj_deproj[:, 1, :]


if agent['recompute_demo_values'] or not os.path.exists(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_f_mean_trajectory.npy')):
    print('cannot find pre-computed trajectory data - process from scratch')
    f_mean_traj, dep_f_mean_traj, gripper_binary_traj = get_unprojected_3d_mean_finger_and_gripper_trajectory(
                                                    l_f_traj, r_f_traj, l_f_traj_orig, r_f_traj_orig, agent['gripper_threshold'], intrin=INTRIN)#, img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH)
    gripper_binary_traj
    np.save(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_f_mean_trajectory.npy'), f_mean_traj)
    np.save(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_dep_f_mean_trajectory.npy'), dep_f_mean_traj)
    np.save(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_gr_bin_trajectory.npy'), gripper_binary_traj)
else:
    print('successfully loaded pre-computed trajectory data')
    f_mean_traj = np.load(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_f_mean_trajectory.npy'))
    dep_f_mean_traj = np.load(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_dep_f_mean_trajectory.npy'))
    gripper_binary_traj = np.load(join(DEMO_PATH, 'videos/poses', DEMO_NAME, 'right_hand_gr_bin_trajectory.npy'))

f_mean_traj = f_mean_traj[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-step_offset_back:step_size]
dep_f_mean_traj = dep_f_mean_traj[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-step_offset_back:step_size]
agent['u0'] = dep_f_mean_traj
l_f_traj = l_f_traj[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-step_offset_back:step_size]
r_f_traj = r_f_traj[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-step_offset_back:step_size]

gripper_binary_traj = gripper_binary_traj[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-step_offset_back:step_size]
imgs = imgs[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-pose_offset-step_offset_back:step_size]
dimgs = dimgs[TRUNCATE_FRONT+step_offset_front:-TRUNCATE_BACK-pose_offset-step_offset_back:step_size]
print('length of truncated demo video: ', len(imgs))

##### PLOTTING & COST PARSING #####
### Compute pixel feature target and plot images ###

tgt_imgs = []
plot_imgs = [] # target imgs stacked horizontally for plotting purposes
points_tgt = []

if agent['visualize_mrcnn']:    
    fig1, ax1 = visualize.get_ax()
    fig2, ax2 = visualize.get_ax()
else:
    ax = None

# reader_w_robot = imageio.get_reader('/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/gps_data/pour/lacan9_view0/data_files/2019-03-29_01-10-44/vids/rgb_sample_itr_9.mp4')
# imgs_r = []
# for img in reader_w_robot:
#     imgs_r.append(img)

# Compute MRCNN centroids & masks if not loaded from file
for tt in range(agent['T']):
    dimg = dimgs[tt]
    # img_r = imgs_r[10]
    img = imgs[tt]
    agent['demo_imgs'].append(img)
    agent['demo_finger_traj'].append(f_mean_traj[tt])
    if not agent['demo_centroids_precomputed']:
        
        all_centroids, all_masks, _ = get_mrcnn_features(mrcnn, img, dimg * DEPTH_SCALE, target_ids, class_names, ax=ax1, viz_image=img)
        # all_centroids, all_masks, _ = get_mrcnn_features(mrcnn, imgs[14], dimg * DEPTH_SCALE, target_ids, class_names, ax=ax2, viz_image=imgs[14])
        import ipdb; ipdb.set_trace()
        curr_masks = {}
        for id in agent['all_object_traj'].keys():
            if id in all_centroids.keys():
                agent['all_object_traj'][id] = np.vstack([agent['all_object_traj'][id], all_centroids[id]])
                curr_masks[id]= all_masks[id]
            else: # object wasnt found, use last known value
                agent['all_object_traj'][id] = np.vstack([agent['all_object_traj'][id], agent['centroids_last_known'][id]])
                curr_masks[id] = agent['masks_last_known'][id]

            try:
                agent['centroids_last_known'][id] = all_centroids[id] # assumes object is found in first frame at least
                agent['masks_last_known'][id] = all_masks[id]
            except:
                pass
        agent['all_mask_traj'].append(curr_masks)

        if tt == agent['T'] - 1:
            with open(os.path.join(common['experiment_dir'], 'all_object_traj.pkl'), 'wb') as fp:
                pickle.dump(agent['all_object_traj'], fp, -1)  
            with open(os.path.join(common['experiment_dir'], 'all_mask_traj.pkl'), 'wb') as fp:
                pickle.dump(agent['all_mask_traj'], fp, -1)  

    # Collect Don Features 
    mask = agent['all_mask_traj'][tt][agent['anchor_id']]
    # centroid = agent['all_object_traj'][agent['anchor_id']][tt]   
    centroid = np.mean(np.array(np.where(mask == 1)), axis=1)
    centroid = [int(centroid[1]), int(centroid[0])]
    feature_pts = get_sample_points(mask, agent['n_max_correspondences'])
    feature_pts_local_frame = feature_pts - centroid[:2]
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(feature_pts_local_frame)
    comp1 = pca.components_[0]
    comp2 = pca.components_[1]
    if tt == 0:
        if comp1[1] >= 0:
            comp1 *= -1
    else:
        if np.dot(comp1, points_tgt[-1][:2]) < 0:
            comp1 *= -1
    # if tt == 0:
    #     if comp2[0] >= 0:
    #         comp2 *= -1
    # else:
    #     if np.dot(comp2, points_tgt[-1][2:]) < 0:
    #         comp2 *= -1   
    # points_tgt.append(np.concatenate([comp1, comp2]))
    points_tgt.append(comp1)
    points_tgt[-1] = np.mean(np.array(points_tgt[-3:]), axis=0)
    com1 = points_tgt[-1]

    ## DEBUG PLOT for first image of demo to show sample of query points
    img_ = imgs[tt][:, :, ::-1].copy()
    cv2.line(img_,(int(centroid[0]),int(centroid[1])),(int(centroid[0])+int(comp1[0]*100),int(centroid[1])+int(comp1[1]*100)),(255,0,0),5)
    cv2.line(img_,(int(centroid[0]),int(centroid[1])),(int(centroid[0])+int(comp2[0]*100),int(centroid[1])+int(comp2[1]*100)),(0,255,0),5)
    cv2.imshow('test', img_)
    cv2.waitKey(150)


    tgt_imgs.append(imgs[tt])
    if tt == 0 :
        plot_imgs = imgs[tt]
    else:
        plot_imgs = np.hstack([plot_imgs, imgs[tt]])

### Plot demo images and gripper trajectory if enabled
col = ['b', 'r']
colors_gripper = ['black'] + [col[0] if gripper_binary_traj[i] == 0 else col[1] for i in range(len(gripper_binary_traj)-1) ] # black diamond for startpoint

if agent['show_demo_plot']:
    fig1 = plt.figure(figsize=(20,5))
    plt.imshow(plot_imgs)
    plt.axis('off')
    # Plot in 2D     # width, height indexed for scatter (x, y)
    fig2 = plt.figure()
    plt.scatter(f_mean_traj[:, 0], -f_mean_traj[:, 1],color=colors_gripper, marker='D')
    plt.plot(f_mean_traj[:, 0], -f_mean_traj[:, 1], color='black')
    plt.scatter(agent['all_object_traj'][agent['anchor_id']][:, 0], -agent['all_object_traj'][agent['anchor_id']][:, 1],s=5, color='g')
    plt.plot(agent['all_object_traj'][agent['anchor_id']][:, 0], -agent['all_object_traj'][agent['anchor_id']][:, 1], color='g')
    plt.scatter(agent['all_object_traj'][agent['object2_id']][:, 0], -agent['all_object_traj'][agent['object2_id']][:, 1],s=5, color='orange')
    plt.plot(agent['all_object_traj'][agent['object2_id']][:, 0], -agent['all_object_traj'][agent['object2_id']][:, 1], color='orange')
        
    plt.show()
    import ipdb; ipdb.set_trace()
    plt.close(fig1); plt.close(fig2)



### Compute Cost Target ###
dep_anchor_traj = get_unprojected_3d_trajectory(agent['all_object_traj'][agent['anchor_id']], INTRIN) # VERIFIED
dep_object2_traj = get_unprojected_3d_trajectory(agent['all_object_traj'][agent['object2_id']], INTRIN) # VERIFIED
geom_dist_hand_to_anchor = dep_f_mean_traj - dep_anchor_traj # VERIFIED
geom_dist_object2_to_anchor = dep_object2_traj - dep_anchor_traj # VERIFIED
geom_dist_hand_to_object2 = dep_f_mean_traj - dep_object2_traj # VERIFIED
agent['dep_anchor_traj'] = dep_anchor_traj
agent['figure_axes'][1][2, 0].clear()
agent['figure_axes'][1][2, 0].set_title('demo_hand_to_anchor_distance')
agent['figure_axes'][1][2, 0].plot(geom_dist_hand_to_anchor[:, 0],c='r', label='x')
agent['figure_axes'][1][2, 0].plot(geom_dist_hand_to_anchor[:, 1], c='g', label='y')
agent['figure_axes'][1][2, 0].plot(geom_dist_hand_to_anchor[:, 2], c='b', label='z')
agent['figure_axes'][1][2, 0].legend()

agent['figure_axes'][1][2, 1].clear()
agent['figure_axes'][1][2, 1].set_title('demo_object2_to_anchor_distance')
agent['figure_axes'][1][2, 1].plot(geom_dist_object2_to_anchor[:, 0],c='r', label='x')
agent['figure_axes'][1][2, 1].plot(geom_dist_object2_to_anchor[:, 1], c='g', label='y')
agent['figure_axes'][1][2, 1].plot(geom_dist_object2_to_anchor[:, 2], c='b', label='z')
agent['figure_axes'][1][2, 1].legend()    

agent['figure_axes'][1][0, 0].clear()
agent['figure_axes'][1][0, 0].set_title('demo_pca_pose')
agent['figure_axes'][1][0, 0].plot(np.array(points_tgt)[:, 0],c='r', label='x')
agent['figure_axes'][1][0, 0].plot(np.array(points_tgt)[:, 1],c='g', label='y')
agent['figure_axes'][1][0, 0].legend()    

assert len(geom_dist_hand_to_anchor) == len(geom_dist_object2_to_anchor) 
assert geom_dist_hand_to_anchor.shape[1] == geom_dist_object2_to_anchor.shape[1] == 3

# dim: 
cost_tgt = np.hstack([geom_dist_hand_to_anchor,  geom_dist_object2_to_anchor])
agent['cost_tgt'] = cost_tgt
agent['demo_gripper_trajectory'] = gripper_binary_traj


cost_wt = np.ones(SENSOR_DIMS[OBJECT_POSE])  
cost_wt[:3] *= 0. # decrease hand weight
cost_wt[[2, 5]] *= 1.0 # depth was halfed during deprojection
cost_wt *= 1
# cost_wt[-1] = 0
state_cost_geom = {
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
    # 'ramp_option': RAMP_LINEAR,
}

########
cost_wt[:3] *= .0 # decrease hand weight
fk_state_cost_geom = {
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
    'wu': 5e-6 / GP_GAINS,
}


agent['cost_tgt_image'] = np.array(tgt_imgs)
agent['cost_tgt_image_feature'] = points_tgt
### Setup Cost functions and such ###
cost_wt = np.ones(SENSOR_DIMS[IMAGE_FEATURE]) 
visual_cost = {
    'type': CostVisual,
    'l1': 1.0,
    'l2': 0.001,
    'alpha': 1e-6,
    'data_types': {
        IMAGE_FEATURE: {
            # 'target_state': np.zeros((agent['T'], 1)),
            'target_state': points_tgt,
            'wp': cost_wt,
        },
    },
    # 'ramp_option': RAMP_LINEAR,
}

fk_state_cost_visual = {
    'type': CostState,
    'l1': 0.5,
    'l2': 1.0,
    'alpha': 1e-5,
    'data_types': {
        IMAGE_FEATURE: {
            'target_state': points_tgt,
            'wp': cost_wt,
        },
    },
    'ramp_option': RAMP_FINAL_ONLY,
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-6 / GP_GAINS,
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
    'costs': [state_cost_geom, fk_state_cost_geom, fk_state_cost_visual, visual_cost, torque_cost],
    'weights': [1.0, 1.0, 1.0, 1.0, 0.0],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  np.ones(SENSOR_DIMS[ACTION]) / GP_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 20.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    # 'final_weight': 50,
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
    'num_samples': agent['num_samples'],
    'verbose_trials': 1,
    'verbose_policy_trials' : True,
    'common': common,
    'agent': agent,
    'gui_on': agent['gps_gui_on'],
    'algorithm': algorithm,
    'random_seed':0,
}

common['info'] = generate_experiment_info(config)

## PI2
# Configure algorithm's hyperparameters

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
#     'costs': [state_cost_geom, fk_state_cost_geom, fk_state_cost_visual, visual_cost, torque_cost],
#     'weights': [1.0, 0.5, 0.5, 1.0, 1.0],
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
#     'num_samples': agent['num_samples'],
#     'verbose_trials': 1,
#     'verbose_policy_trials' : True,
#     'common': common,
#     'agent': agent,
#     'gui_on': agent['gps_gui_on'],
#     'algorithm': algorithm,
#     'random_seed':0,
# }

# common['info'] = generate_experiment_info(config)


