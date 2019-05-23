""" This file defines an agent for the PR2 ROS environment. """
from __future__ import division

import copy
import time
import numpy as np
import os
import sys
import rospy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ROS
# from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
#         policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
# from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
# from gps_agent_pkg.msg import TrialCommand, SampleResult, PositionCommand, \
#         RelaxCommand, DataRequest, TfActionCommand, TfObsData
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, NOISE, TCN_EMBEDDING, RGB_IMAGE, RCNN_OUTPUT

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None

from gps.sample.sample import Sample

# ROS specific
import rospy
import rospkg

from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
    GetModelState,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

# Baxter specific
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface
from baxter_pykdl import baxter_kinematics

import moveit_commander
import moveit_msgs.msg
from moveit_commander import conversions


# Custom modules
from plans import BaxterInterfacePlanner
from baxter_utils import img_subscriber, depth_subscriber, resize_frame, load_model, IMAGE_SIZE
from ipdb import set_trace

import torch
from imageio import imwrite
from collections import OrderedDict
import cv2


# Mask RCNN imports
sys.path.append('/home/msieb/projects/Mask_RCNN/samples')
import tensorflow as tf
from baxter.baxter import BaxterConfig
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
print(tf.__version__)

class AgentBaxterRCNN(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """
    def __init__(self, hyperparams, init_node=False):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        # plt.ion()
        plt.ioff()
        config = copy.deepcopy(AGENT_ROS)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_ros_node', anonymous=True)
        # self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        conditions = self._hyperparams['conditions']

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        self.x0 = self._hyperparams['x0']


        self.use_tf = False
        self.observations_stale = True

        # Baxter specific
        limb = 'left'
        self._trial_limb = limb
        self._trial_arm = baxter_interface.Limb(self._trial_limb)
        self._trial_gripper = baxter_interface.Gripper(self._trial_limb, baxter_interface.CHECK_VERSION)
        self._auxiliary_limb = 'right'
        self._auxiliary_arm = baxter_interface.Limb(self._auxiliary_limb)
        self._auxiliary_gripper = baxter_interface.Gripper(self._auxiliary_limb, baxter_interface.CHECK_VERSION)
             
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
      
        self.joint_names = self._trial_arm.joint_names()
        print(self.joint_names)
        self.n_joints = len(self.joint_names)
        self.n_emb = self._hyperparams['sensor_dims'][TCN_EMBEDDING]
        self._joint_idx = list(range(self.n_joints))
        self._vel_idx = [i + self.n_joints for i in self._joint_idx]
        self._pos_idx = [i +2* self.n_joints for i in range(3)]
        self._emb_idx = [i + 3 + 2*self.n_joints for i in range(self.n_emb)]
        self.planner = BaxterInterfacePlanner(limb)
        self._rate = 1.0 / self._hyperparams['dt']
        self._control_rate = rospy.Rate(self._rate)
        self._missed_cmds = 20.0
        self._trial_arm.set_command_timeout((1.0 / self._rate) * self._missed_cmds)
        self._kin_aux = baxter_kinematics('right')
        self._kin_trial = baxter_kinematics('left')
        self.views_ids = self._hyperparams['views']
        self.taskspace_deltas = self._hyperparams['taskspace_deltas']
        topic_img_list = ['/camera' + view_id + '/color/image_raw' for view_id in self.views_ids]
        topic_depth_list = ['/camera' + view_id + '/aligned_depth_to_color/image_raw' for view_id in self.views_ids]
        # pub = rospy.Publisher('/tcn/embedding', numpy_msg(Floats), queue_size=3)
        self.img_subs_list = [img_subscriber(topic=topic_img) for topic_img in topic_img_list]
        self.depth_subs_list = [depth_subscriber(topic=topic_depth) for topic_depth in topic_depth_list]

        # Initialize Mask RCNN model
        self.debug_mode = self._hyperparams['debug_mode']
        self.plot_mode = True
        model_path = self._hyperparams['model_path']
        MODEL_DIR = '/home/msieb/projects/gps/experiments/baxter_reaching/data_files'
        class InferenceConfig(BaxterConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        inference_config = InferenceConfig()
        with tf.device('/device:GPU:1'):
            self.rcnn = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                                      config=inference_config)
            self.rcnn.load_weights(model_path, by_name=True)

        self.class_names = ['BG', 'blue_ring', 'green_ring', 'yellow_ring', 'tower', 'hand', 'robot']
        self.visual_idx_1 = self._hyperparams['visual_indices_1']
        self.visual_idx_2 = self._hyperparams['visual_indices_2']
        self.target_ids = [1, 4]

        # Take some ramp images to allow cams to adjust for brightness etc.
        img_subs = self.img_subs_list[0]
        depth_subs = self.depth_subs_list[0]
        for i in range(100):
            # Get frameset of color and depth
            color_image = img_subs.img
            depth_image = depth_subs.img
            resized_image = resize_frame(color_image, IMAGE_SIZE)[None, :]
            print('Taking ramp image %d.' % i)
        # DEBUG PICTURE: image = plt.imread('/home/msieb/projects/Mask_RCNN/datasets/baxter/test/2_view0/00000.jpg')
        results = self.rcnn.detect([color_image], verbose=0)
        r = results[0]
        self.colors = visualize.random_colors(7)
        # ax = visualize.display_instances(color_image, r['rois'], r['masks'], r['class_ids'], 
        #                         self.class_names, r['scores'], ax=visualize.get_ax()[1], colors=self.colors)

    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
        """
        relax_position =  [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]
        # relax_position = self._hyperparams['x0'][0][:7]
        zipped_joints = {key: val for key, val in zip(self.joint_names, relax_position)}

        arm.move_to_joint_positions(zipped_joints)
        self._control_rate.sleep()
        joints_curr = [self._trial_arm.joint_angle(j)
                        for j in self.joint_names]

        if not np.allclose(joints_curr, relax_position, atol=0.01):
            print("error reaching reset position")

    def reset_arm(self, arm, mode, data):
        """
        Issues a position command to an arm.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
            mode: An integer code (defined in gps_pb2).
            data: An array of floats.
        """
        if mode == 'ANGLE':
            zipped_joints = {key: val for key, val in zip(self.joint_names, data)}
            # self._trial_arm.set_joint_positions(zipped_joints)
            arm.move_to_joint_positions(zipped_joints)
            self._control_rate.sleep()
            joints_curr = [self._trial_arm.joint_angle(j)
                            for j in self.joint_names]
            if not np.allclose(joints_curr, data, atol=0.01):
                print("error reaching reset position")

        elif mode == 'CARTESIAN':
            self.planner.move_to_cartesian_pose(data)

        else:
            print("Invalid mode provided, either CARTESIAN or ANGLE must be chosen")


    def reset(self):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        # condition_data = self._hyperparams['reset_conditions'][condition]
        # self.relax_arm(self._trial_arm)
        self.reset_arm(arm=self._trial_arm, mode='ANGLE', data=self._hyperparams['x0'][0][:7])
        # self.relax_arm(self._auxiliary_arm)
        self._control_rate.sleep()  # useful for the real robot, so it stops completely

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        Returns:
            sample: A Sample object.
        """
        if TfPolicy is not None:  # user has tf installed.
            if isinstance(policy, TfPolicy):
                self._init_tf(policy.dU)

        self.reset()
        new_sample = self._init_sample(condition)
        b_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        noise /= 50
        # print(noise[0,:])

        # noise[5] /= 10
        # noise[4] /= 20

        # noise[1] *= 5
        # noise[3] *= 5 
        # Take the sample.
        t = 0
        while t < self.T:
            curr_time = rospy.get_time()
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            if self._hyperparams['curr_runs'] < 5:
                b_U = self._hyperparams['u0'][t, :] 
                b_U =  noise[t, :]
                 # print(b_U)

            else:
                b_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = b_U

            # clip deltas to the given limits!
            b_U[:-1] = np.clip(b_U[:-1], -self.taskspace_deltas[:-1], self.taskspace_deltas[:-1])        
            b_U[-1] = np.clip(b_U[-1], -self.taskspace_deltas[-1], self.taskspace_deltas[-1])

            if (t + 1) < self.T:
                # b_X, b_U_check, image = self._step(b_U, curr_time)
                b_X, b_U_check, image, rcnn_image = self._step_taskspace(b_U, X_t, curr_time)   
                if b_X is None:
                    self.reset()
                    rospy.sleep(0.5)
                    new_sample = self._init_sample(condition)
                    b_X = self._hyperparams['x0'][condition]
                    U = np.zeros([self.T, self.dU])
                    # Generate noise.
                    if noisy:
                        noise = generate_noise(self.T, self.dU, self._hyperparams)
                    else:
                        noise = np.zeros((self.T, self.dU))
                    noise /= 50
                    t = 0
                    continue
                else:
                    self._set_sample(new_sample, b_X, t, condition)
                    new_sample.set(RGB_IMAGE, image, t=t+1)
                    new_sample.set(RCNN_OUTPUT, image, t=t+1)

            t += 1
        if self._hyperparams['curr_runs'] < 5:
            self._hyperparams['curr_runs'] += 1
        
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        print("Took sample...")
        self.reset()
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _step(self, U, curr_time):
        within_bounds = self._within_bounds()

        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
        cmd = {}
        for idx, name in enumerate(self.joint_names):
                cmd[name] = U[idx]
        if within_bounds or True:
            self._trial_arm.set_joint_velocities(cmd)

            # self._trial_arm.set_joint_torques(cmd)
        else:
            print("End effector is out of bounds")
        # self._control_rate.sleep()
        step_time = rospy.get_time() - curr_time

        rospy.sleep(self._hyperparams['dt'] - step_time)
        X, U_, image, rcnn_image = self._get_current_state()

        return X, U, image, rcnn_image

    def _step_taskspace(self, U, X_t, curr_time):
        U_pos = list(U[:-1])
        U_pos[0] = 0.0
        U_rot = U[-1]
        within_bounds = self._within_bounds_task(U_pos, X_t)

        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])

        pos = [a + b for a, b in zip(pos0, U_pos)] # Check limits and additon

        joint_pos = self._kin_trial.inverse_kinematics(pos, orn0)
        if joint_pos is not None:
            cmd = {}
            for idx, name in enumerate(self.joint_names):
                    cmd[name] = joint_pos[idx]
                    if name == 'left_w2':
                        cmd[name] += U_rot
            if within_bounds:
                self._trial_arm.move_to_joint_positions(cmd, timeout=self._hyperparams['dt']*0.9)
            else:
                print("out of bounds")
        else:
            print("inverse kinematic calculation failed")
        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")

        # print('action: ', U)
        # print([self._trial_arm.joint_velocity(j) for j in self.joint_names])
        # self._control_rate.sleep()
        step_time = rospy.get_time() - curr_time
        rospy.sleep(self._hyperparams['dt'] - step_time)
        X, U_, image, rcnn_image = self._get_current_state()
        return X, U, image, rcnn_image



    def get_depth_img(self, depth_subs_obj):
        depth_scale = 0.001 # not precisely, but up to e-8
        clipping_distance_in_meters = 1.5 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        depth_image = depth_subs_obj.img
        depth_image[np.where(depth_image > clipping_distance)] = 0
        depth_rescaled = (((depth_image  - 0) / (clipping_distance - 0)) * (255 - 0) + 0).astype(np.uint8)
        return depth_rescaled

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """

        sample = Sample(self)

        # Initialize world/run kinematics
        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        dorn = list(self._trial_arm.endpoint_velocity()['angular'])
        jac = self._kin_trial.jacobian()

        sample.set(JOINT_ANGLES, np.asarray(q), t=0)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=0)
        sample.set(END_EFFECTOR_POINTS, np.asarray(pos), t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=0)
        
        img_subs = self.img_subs_list[0]
        depth_subs = self.depth_subs_list[0]     
        image = img_subs.img 
        depth_rescaled = self.get_depth_img(depth_subs)

        all_visual_features, all_centroids, fig = self._get_rcnn_features(image, depth_rescaled)
        try:
            delta_centroid = all_centroids[0] - all_centroids[1]
        except:
            delta_centroid = np.array([30, 30, 30])
        # print(all_centroids)
        # set_trace()# image_buffer.append(image[:,:,::-1])
        feat_visual_1, feat_visual_2, feat_visual_max_1, feat_visual_max_2 = self._apply_feature_selection(all_visual_features)

        embedding = np.concatenate([delta_centroid, feat_visual_1, feat_visual_max_1, feat_visual_2, feat_visual_max_2])

        sample.set(TCN_EMBEDDING, embedding, t=0)

        if fig is not None:
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            canvas.draw()       # draw the canvas, cache the renderer
            img = np.array(fig.canvas.renderer._renderer)
            sample.set(RCNN_OUTPUT, img, t=0)
            sample.set(RGB_IMAGE, image, t=0)
            plt.close(fig)
        else:
            sample.set(RGB_IMAGE, image, t=0)
            sample.set(RCNN_OUTPUT, np.zeros((800,800,4)), t=0)

        return sample


    def _get_current_state(self):
        U = [self._trial_arm.joint_effort(j) for j in self.joint_names]

        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        dorn = list(self._trial_arm.endpoint_velocity()['angular'])

        img_subs = self.img_subs_list[0]
        depth_subs = self.depth_subs_list[0]     
        image = img_subs.img 
        depth_rescaled = self.get_depth_img(depth_subs)

        all_visual_features, all_centroids, fig = self._get_rcnn_features(image, depth_rescaled)
        # print(all_centroids)
        try:
            delta_centroid = all_centroids[0] - all_centroids[1]
        except:
            return None, None, None
        # set_trace()# image_buffer.append(image[:,:,::-1])
        feat_visual_1, feat_visual_2, feat_visual_max_1, feat_visual_max_2 = self._apply_feature_selection(all_visual_features)
        embedding = np.concatenate([delta_centroid, feat_visual_1, feat_visual_max_1, feat_visual_2, feat_visual_max_2])
        X = np.concatenate([q, dq, pos, embedding])

        if fig is not None:
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            canvas.draw()       # draw the canvas, cache the renderer
            rcnn_image = np.array(fig.canvas.renderer._renderer)
            plt.close(fig)
        return X, U, image, rcnn_image

    def _apply_feature_selection(self, all_visual_features):
        # Define relevant feature sparisification here
        roi_features_1 = all_visual_features[0]
        roi_features_2 = all_visual_features[1]
        roi_features_1_map_flat = np.reshape(roi_features_1, [7*7, 256])
        roi_features_2_map_flat = np.reshape(roi_features_2, [7*7, 256])

        roi_features_1_map_mean = np.mean(roi_features_1_map_flat, axis=0)
        roi_features_2_map_mean = np.mean(roi_features_2_map_flat, axis=0)
        roi_features_1_map_max = np.max(roi_features_1_map_flat, axis=0)
        roi_features_2_map_max = np.max(roi_features_2_map_flat, axis=0)

        feat_visual_1 = roi_features_1_map_mean[self.visual_idx_1[0]]
        feat_visual_2 = roi_features_2_map_mean[self.visual_idx_2[0]]

        feat_visual_max_1 = roi_features_1_map_max[self.visual_idx_1[1]]
        feat_visual_max_2 = roi_features_2_map_max[self.visual_idx_2[1]]
        return feat_visual_1, feat_visual_2, feat_visual_max_1, feat_visual_max_2


    def _get_rcnn_features(self, image, depth_rescaled):
        results = self.rcnn.detect([image], verbose=0)
        r = results[0]
        encountered_ids = []
        all_cropped_boxes = []
        all_centroids_unordered = [] # X Y Z
        all_centroids = dict()
        all_visual_features_unordered = []
        all_visual_features = dict()
        for i, box in enumerate(r['rois']):
            class_id = r['class_ids'][i]
            if class_id not in self.target_ids or class_id in encountered_ids:
                continue
            encountered_ids.append(class_id)
            cropped = utils.crop_box(image, box, y_offset=20, x_offset=20)
            # cropped = utils.resize_image(cropped, max_dim=299)[0]
            cropped = cv2.resize(cropped, (299, 299))
            all_cropped_boxes.append(cropped)

            masked_depth = depth_rescaled * r['masks'][:, :, i]
            masked_depth = masked_depth[np.where(masked_depth > 0)]
            z = np.median(np.sort(masked_depth.flatten()))

            x, y = utils.get_box_center(box)

            all_centroids_unordered.append([x, y, z])
            all_visual_features_unordered.append(r['roi_features'][i])

        all_cropped_boxes = np.asarray(all_cropped_boxes)
        all_centroids_unordered = np.asarray(all_centroids_unordered)

        for i in range(all_cropped_boxes.shape[0]):
            all_visual_features[encountered_ids[i]] = all_visual_features_unordered[i]
            all_centroids[encountered_ids[i]] = all_centroids_unordered[i]
        all_centroids = np.asarray([val for key, val in all_centroids.items()])
        all_visual_features = np.asarray([val for key, val in all_visual_features.items()])
        if self.plot_mode:
            fig, ax = visualize.get_ax()
            ax = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],            
                        self.class_names, r['scores'], ax=ax, colors=self.colors)
        else:
            fig = None
        return all_visual_features, all_centroids, fig


    def _set_sample(self, sample, X, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
       
        sample.set(JOINT_ANGLES, X[self._joint_idx], t=t+1)
        sample.set(JOINT_VELOCITIES, X[self._vel_idx], t=t+1)
        sample.set(END_EFFECTOR_POINTS, X[self._pos_idx], t=t+1)
        sample.set(TCN_EMBEDDING, X[self._emb_idx], t=t+1)
        jac = self._kin_trial.jacobian()
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=t+1)
        return
    
    def _within_bounds(self):
        X, U, _ = self._get_current_state()
        pos = X[14:17]
        x0=0.408863125303
        y0= -0.19033274891
        z0= -0.292697992966
        x1=0.766521737946
        y1=0.374777249352
        z1=0.259107749513
        if pos[0] < x1 and pos[0] > x0 \
                and pos[1] < y1 and pos[1] > y0 \
                and pos[2] < z1 and pos[2] > z0:
            within_bounds = True
        else:
            within_bounds = False
        return within_bounds

    def _within_bounds_task(self, delta_pos, X_t):
        # X, U, _ = self._get_current_state()
        pos = X_t[14:17]
        x0=0.408863125303
        y0= -0.08033274891
        z0= -0.292697992966
        x1=0.776521737946
        y1=0.374777249352
        z1=0.010107749513
        if pos[0] + delta_pos[0] < x1 and pos[0] + delta_pos[0] > x0 \
                and pos[1] + delta_pos[1] < y1 and pos[1] + delta_pos[1] > y0 \
                and pos[2] + delta_pos[2] < z1 and pos[2] + delta_pos[2] > z0:
            within_bounds = True
        else:
            return True
            within_bounds = False
        return within_bounds

