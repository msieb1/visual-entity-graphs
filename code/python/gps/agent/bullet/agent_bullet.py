""" This file defines an agent for the Bullet simulator environment. """
import copy
import sys
import numpy as np
from os.path import join
from easydict import EasyDict
import pybullet as p
import time
import math
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pdb import set_trace
import torch
from PIL import Image
import imageio
import importlib
import cv2
from copy import deepcopy as copy

sys.path.append("../../python/")
# GPS Imports
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_BULLET
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE, IMAGE_FEATURE, OBJECT_POSE, ANCHOR_OBJECT_POSE
from gps.sample.sample import Sample
from bullet.simenv.kuka_iiwa import kuka_iiwa
import pyquaternion as pq

from gps.agent.bullet.bullet_utils import pixel_normalize, resize_frame, load_tcn_model, tinyDepth, list_add
sys.path.append("../../../")

# Custom imports
sys.path.append('/home/msieb/projects/LTCN')
sys.path.append('/home/msieb/projects/general-utils')
from rot_utils import sincos2rotm, rotationMatrixToEulerAngles, geodesic_dist_quat

# from tcn import define_model_depth as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model


# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class AgentBullet(Agent):
    """
    All communication between the algorithms and Bullet is done through
    this class.
    """
    def __init__(self, hyperparams):
        # Get configs, hyperparameters & initialize agent
        self.ptconf, self.dconf, self.camconf = hyperparams['ptconf'], hyperparams['dconf'], hyperparams['camconf']
        self.env_name = hyperparams['env']
        self.gui_on = hyperparams['bullet_gui_on']
        config = copy(AGENT_BULLET)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self.hyperparams = hyperparams

        # Setup bullet environment
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])
        self.setup_bullet()
        self.setup_inference_camera()

        # Get demo data
        self.demo_vid = imageio.get_reader(join(self.dconf.DEMO_DIR, self.dconf.DEMO_NAME, 'rgb/{}.mp4'.format(self.dconf.SEQNAME)))
        self.demo_frames = []
        self.reset_condition = hyperparams['reset_condition']
        for im in self.demo_vid:
            self.demo_frames.append(im)

        # Setup feature embedding network if enabled
        if self.ptconf.COMPUTE_FEATURES:
            self.tcn = hyperparams ['vf_model'] 
            # print("model path: {}".format(self.ptconf.MODEL_PATH))
        else:
            self.tcn = None
        self.vid_seqname = 0
        self.feature_fn = hyperparams['feature_fn']

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)


    def setup_bullet(self):
        module = importlib.import_module('python.bullet.simenv.' + self.env_name)
        envClass = getattr(module, 'UserEnv')
        self.env = envClass()

        cid = -1
        demo_path = None
        if demo_path is None:
            cid = p.connect(p.SHARED_MEMORY)

        if cid<0:
            if self.gui_on:
                cid = p.connect(p.GUI)
            else:
                cid = p.connect(p.DIRECT)
        p.resetSimulation()
        #disable rendering during loading makes it much faster
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Env is all loaded up here
        self.h, self.o = self.env.load()
        print('Total Number:', p.getNumBodies())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setGravity(0.000000,0.000000,0.000000)
        # p.setGravity(0,0,-10)

        ##show this for 10 seconds
        #now = time.time()
        #while (time.time() < now+10):
        #   p.stepSimulation()
        p.setRealTimeSimulation(1)

        self._numJoints = p.getNumJoints(self.o.kukaobject.kukaId)
        self.emb_dim = self._hyperparams['sensor_dims'][image_feature]
        self.n_pose = self._hyperparams['sensor_dims'][OBJECT_POSE]
        self._joint_idx = list(range(self._numJoints))
        self._vel_idx = [i + self._numJoints for i in self._joint_idx]
        self._pos_idx = [i +2* self._numJoints for i in range(7)]
        self._emb_idx = [i + 7 + 2*self._numJoints for i in range(self.emb_dim)]
        self._obj_cent_idx = [i + self._hyperparams['sensor_dims'][image_feature] + 3 + 2*self._numJoints for i in range(self.n_pose)]
        self._anchor_obj_cent_idx = [i + 7 + self._hyperparams['sensor_dims'][image_feature] + 3 + 2*self._numJoints for i in range(self.n_pose)]

    def setup_cameras(self):
        object_p3d = self.getObjectPose(self.h.duckid)[0:3]
        new_targets = [object_p3d]
        self.viewMatrices = []
        self.projectionMatrices = []
        for i in range(len(self.camconf.VIEW_PARAMS)):

          self.viewMatrices.append(p.computeViewMatrixFromYawPitchRoll(
              cameraTargetPosition=new_targets[0],
              distance=self.camconf.VIEW_PARAMS[i]['distance'], 
              yaw=self.camconf.VIEW_PARAMS[i]['yaw'], 
              pitch=self.camconf.VIEW_PARAMS[i]['pitch'], 
              roll=self.camconf.VIEW_PARAMS[i]['roll'], 
              upAxisIndex=self.camconf.VIEW_PARAMS[i]['upAxisIndex']
           ))

    
        for i in range(len(self.viewMatrices)):
          self.projectionMatrices.append(
            p.computeProjectionMatrixFOV(self.camconf.PROJ_PARAMS['fov'], 
              self.camconf.PROJ_PARAMS['aspect'], self.camconf.PROJ_PARAMS['nearPlane'], self.camconf.PROJ_PARAMS['farPlane']))
    
    def setup_inference_camera(self):
        self.viewMatrices = []
        self.projectionMatrices = []
        for i in range(len(self.camconf.VIEW_PARAMS)):

          self.viewMatrices.append(p.computeViewMatrixFromYawPitchRoll(
              cameraTargetPosition=self.camconf.VIEW_PARAMS[i]['cameraTargetPosition'],
              distance=self.camconf.VIEW_PARAMS[i]['distance'], 
              yaw=self.camconf.VIEW_PARAMS[i]['yaw'], 
              pitch=self.camconf.VIEW_PARAMS[i]['pitch'], 
              roll=self.camconf.VIEW_PARAMS[i]['roll'], 
              upAxisIndex=self.camconf.VIEW_PARAMS[i]['upAxisIndex']
           ))

        for i in range(len(self.viewMatrices)):
          self.projectionMatrices.append(
            p.computeProjectionMatrixFOV(self.camconf.PROJ_PARAMS['fov'], 
              self.camconf.PROJ_PARAMS['aspect'], self.camconf.PROJ_PARAMS['nearPlane'], self.camconf.PROJ_PARAMS['farPlane']))

        
    def _setup_world(self, filename):
        """
        Helper method for handling setup of the Bullet world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self.x0 = self._hyperparams['x0']
 


    def get_state(self, t):
        """
        Retrieves current system state
        """
        joint_pos = [p.getJointState(self.o.kukaobject.kukaId,i)[0] for i in range(self._numJoints)]
        joint_vel = [p.getJointState(self.o.kukaobject.kukaId, i)[1] for i in range(self._numJoints)]
        eepos = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)[0]   # is that correct?? or is it another index??
        eeorn = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)[1]
        result = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        zero_vec = [0.0] * len(joint_pos)

        jac_t, jac_r = p.calculateJacobian(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex, com_trn, joint_pos, zero_vec, zero_vec) # what to use???
        
        anchor_object_p3d = np.zeros(7,)
        rgb_img, depth_img = self.get_images(1)
        rgb_crop, depth_crop = self.get_images(0)
        # Compute visual features if desired
        if self.ptconf.COMPUTE_FEATURES:

            a_pred = self.feature_fn(self.tcn, rgb_crop )[0]
            image_feature = np.squeeze(a_pred)

        else:
            image_feature = np.zeros(self.emb_dim)

        tgt = self.hyperparams['debug_cost_tgt'][t+1]
        # print("centroid diff: {}".format(centroid_diff[:3]))
        # print("tgt: {}".format(tgt[:3]))
        print("cost current step {}: {}".format(t+1, np.linalg.norm(a_pred - tgt )))
        object_pose = np.array(self.getObjectPose(self.h.duckid)[:3])

        centroid_diff = np.zeros(7,)
        object_pose = duck_centroid # Cube - Bowl  (# Look at Mask RCNN General config for ID order)

        # Collect all states
        stateX = np.concatenate([np.array(joint_pos), np.array(joint_vel), np.array(eepos + eeorn),image_feature,centroid_diff, anchor_object_p3d]).flatten()
        image_data = [rgb_crop]

        return stateX, jac_t, image_data

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


        if self.vid_seqname % 4 == 0:
            rgb_writer = self.create_writers()
        else: 
            rgb_writer = None


        orn = [0.70603903128, 0.708148792076, 0, 0]
        #orn = p.getQuaternionFromEuler([-math.pi/2,0, math.pi/2])
        pos = [1.0, -0.400000, 0.9]

        self.env.reset(pos + orn)

        new_sample, image_data = self._init_sample(condition)
        U = np.zeros([self.T, self.dU])

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Take the sample.
        for t in range(self.T): # 100 steps
            if rgb_writer is not None:
                rgb_writer.append_data(image_data[0])
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :]) # get actions from policy
            U[t, :] = mj_U

            eepos = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)[0]   # is that correct?? or is it another index??
           
            deltas = self._hyperparams['delta_taskspace']
            mj_U = self.clip_actions(eepos, mj_U, deltas)

            if (t +1) < self.T:
                curr_time = time.time()

                if self._hyperparams['control_type'] == 'task':
                    self.step_taskspace_trans(mj_U, allow_gripper_change)
                    allow_gripper_change = False

                else:
                    self.step_jointspace(mj_U)

                stateX, jac_t, image_data = self.get_state(t)
                run_time = time.time() - curr_time

                time.sleep(max(self._hyperparams['dt'] - run_time, 0.0))
                self._set_sample(new_sample, stateX, jac_t, t, condition)   # is jac_t correct or should it be jac_r??
                
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise
            )
        self.vid_seqname += 1
        if rgb_writer is not None:
            rgb_writer.close()

        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: function to compute image features from the observation.
        """

        sample = Sample(self)
        t = -1
        stateX, jac_t, image_data= self.get_state(t)
        self._set_sample(sample, stateX, jac_t, t, condition)   # is jac_t correct or should it be jac_r??
        return sample, image_data


    def _set_sample(self, sample, X, jac, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            X: Data to set for sample (joint_angle and joint_velocity)
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """

        if t>180:
            None

        if t%15==0:
            None

        if condition==0:
            None

        if condition==1:
            None

        if condition==2:
            None

        if condition==3:
            None

        sample.set(JOINT_ANGLES, X[self._joint_idx], t=t+1)
        sample.set(JOINT_VELOCITIES, X[self._vel_idx], t=t+1)
        sample.set(END_EFFECTOR_POINTS, X[self._pos_idx], t=t+1)
        sample.set(image_feature, X[self._emb_idx], t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(jac), t=t+1)
        sample.set(OBJECT_POSE, X[self._obj_cent_idx], t=t+1)
        sample.set(ANCHOR_OBJECT_POSE, X[self._anchor_obj_cent_idx], t=t+1)

    # Initialize condition!
    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics


        #x0 = self._hyperparams['x0'][condition]

    def load_tcn_model(self, model_path, use_cuda=True):
        tcn = define_model(pretrained=True, action_dim=self.ptconf.ACTION_DIM)
        tcn = torch.nn.DataParallel(tcn, device_ids=range(1))
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        tcn.load_state_dict(state_dict)

        if use_cuda:
            tcn = tcn.cuda()
        return tcn

    def load_tcn_weights(self, model_path):
        # Change dict names if model was created with nn.DataParallel
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        # for k, v in state_dict.items():
        #     name = k[7:] # remove module.
        #     new_state_dict[name] = v
        # map_location allows us to load models trained on cuda to cpu.
        tcn.load_state_dict(state_dict)


    def change_camera_target(self, new_targets, view):

      self.viewMatrices[view] = p.computeViewMatrixFromYawPitchRoll(
          cameraTargetPosition=new_targets,
          distance=self.camconf.VIEW_PARAMS[view]['distance'], 
          yaw=self.camconf.VIEW_PARAMS[view]['yaw'], 
          pitch=self.camconf.VIEW_PARAMS[view]['pitch'], 
          roll=self.camconf.VIEW_PARAMS[view]['roll'], 
          upAxisIndex=self.camconf.VIEW_PARAMS[view]['upAxisIndex']
          )

    def step_taskspace_trans(self, actions, allow_gripper_change = 0):
        """
        Executes a step in taskspace providing endeffector deltas dx, dy, dz and endeffector rotation

        --------
        Parameters:
            actions: list(dx, dy, dz, dtheta)
        """
        actions /= 1

        
        joint_pos = [p.getJointState(self.o.kukaobject.kukaId, i)[0] for i in range(p.getNumJoints(self.o.kukaobject.kukaId))]

        link_state = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)
        curr_eepos = link_state[0]
        curr_orn = link_state[1]
        #print(actions)
        tgt_eepos = list_add(list(curr_eepos), actions[:3])
        target_joint_positions = p.calculateInverseKinematics(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex, tgt_eepos, curr_orn, ll, ul, jr, rp)
        
        #print(joint_pos[7])
        Id = self.env.o.kukaobject.kukaId
        for i in range(self._numJoints):
            jointInfo = p.getJointInfo(Id,i)
            qIndex = jointInfo[3]
            if i not in (self.env.o.kukaobject.lf_id, self.env.o.kukaobject.rf_id) and qIndex > -1:
                p.setJointMotorControl2(bodyIndex=Id, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                        targetPosition=target_joint_positions[i], targetVelocity=0,
                                        force=self.env.o.kukaobject.maxForce,
                                        positionGain=0.05,
                                        velocityGain=1)
        tgt_rot = joint_pos[7] + actions[-1]
        p.setJointMotorControl2(bodyIndex=Id, jointIndex=7, controlMode=p.POSITION_CONTROL,
                                targetPosition=tgt_rot, targetVelocity=0, force=200,
                            positionGain=0.05, velocityGain=1)              
        self.o.kukaobject.gripper_stabler_keep_commands()
        self.o.kukaobject.close_gripper()

        #turn only final angle


        p.stepSimulation()
        time.sleep(0.001)     

    def step_taskspace(self, actions, allow_gripper_change = 0):
        """
        Executes a step in taskspace providing endeffector deltas dx, dy, dz and endeffector rotation

        --------

        Parameters:

            actions: list(dx, dy, dz, dtheta)

        """
          
        joint_pos = [p.getJointState(self.o.kukaobject.kukaId, i)[0] for i in range(p.getNumJoints(self.o.kukaobject.kukaId))]

        link_state = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)
        curr_eepos = link_state[0]
        curr_orn = link_state[1]

        tgt_eepos = list_add(list(curr_eepos), actions[:3])
        # curr_euler_angle = p.getEulerFromQuaternion(curr_orn)
        #tgt_orn = list_add(curr_orn, p.getQuaternionFromEuler(list_add(curr_euler_angle,[actions[-2], actions[-3],actions[-1]])))
        # QUAT: X Y Z       W
        # PYQUATERNION : W   X Y Z
        # set_trace()

        curr_orn_qt = pq.Quaternion(np.array(curr_orn)[[3, 0, 1, 2]])
        action_qt = pq.Quaternion(axis=actions[3:6], radians=actions[6])
        #tgt_orn = list_add(curr_orn, p.getQuaternionFromEuler(list_add(curr_euler_angle,[0, 0,0])))
        tgt_orn_qt = action_qt*curr_orn_qt
        tgt_orn = tgt_orn_qt.elements[[1,2,3,0]]

        #pos = [-0.4, 0.2 * math.cos(t/10), 0. + 0.2 * math.sin(t/10)]
        #orn = list_add(curr_orn, p.getQuaternionFromEuler([0, -math.pi, 0]))
        target_joint_positions = p.calculateInverseKinematics(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex, tgt_eepos, tgt_orn, ll, ul, jr, rp)
        #print(joint_pos[7])
        for i in range(self._numJoints):
            p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_joint_positions[i], targetVelocity=0, force=300,
                            positionGain=0.05, velocityGain=1)
        self.o.kukaobject.gripper_stabler_keep_commands()

        p.stepSimulation()
        time.sleep(0.001)

    def step_jointspace(self, actions):
        """
        Executes a step in taskspace providing endeffector deltas dx, dy, dz and endeffector rotation

        --------
        Parameters:
            actions: list(dx, dy, dz, dtheta)
        """
        
        joint_pos = [p.getJointState(self.o.kukaobject.kukaId, i)[0] for i in range(p.getNumJoints(self.o.kukaobject.kukaId))]

        link_state = p.getLinkState(self.o.kukaobject.kukaId, self.o.kukaobject.kukaEndEffectorIndex)
        curr_eepos = link_state[0]
        curr_orn = link_state[1]

        tgt_eepos = list_add(list(curr_eepos), actions[:3])
        tgt_rot = joint_pos[7] + actions[-1]
        curr_euler_angle = p.getEulerFromQuaternion(curr_orn)
        tgt_orn = list_add(curr_orn, p.getQuaternionFromEuler(list_add(curr_euler_angle,[actions[-2], actions[-3],actions[-1]])))

        for i in range(self._numJoints):
            p.setJointMotorControl2(bodyIndex=self.o.kukaobject.kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_pos[i] + actions[i], targetVelocity=0, force=200,
                            positionGain=0.01, velocityGain=0.35)
        self.o.kukaobject.gripper_stabler_keep_commands()

        for kuka_sec in range(1):
            p.stepSimulation()
            #if self.render:
            if True:
                time.sleep(0.001) 

        # if actions[9] > 0:
        #     self.o.kukaobject.close_gripper()
        # else:
        #     self.o.kukaobject.open_gripper()

    def execute_waypoints(self, waypoints):
       pass

    def get_images(self, view=0):
        cur = time.time()
        object_p3d = self.getObjectPose(self.h.duckid)[0:3]

        if view == 0:
            self.change_camera_target(object_p3d, view)
        img_as_array = p.getCameraImage(self.camconf.IMG_W, self.camconf.IMG_H, self.viewMatrices[view],self.projectionMatrices[view])#, shadow=1,lightDirection=[1,1,1],renderer=p.ER_TINY_RENDERER)
        # set_trace()
        object_p3d_hom = np.asarray([object_p3d + [1]])
        object_p2d_hom =  np.reshape(np.asarray(self.projectionMatrices[view]), [4,4]).dot(np.reshape(np.asarray(self.viewMatrices[view]), [4,4])).dot(object_p3d_hom.T)
        object_p2d = object_p2d_hom[:-2] / object_p2d_hom[-1]
        object_p2d = object_p2d[::-1] # (height, width) indexed (x is width, y is height)
        object_mask = np.where(img_as_array[4] == self.h.duckid)

        rgb_img = img_as_array[2][:,:,:-1]
        depth_img_raw = img_as_array[3]
        depth_img = tinyDepth(np.repeat(depth_img_raw[:,:,None], 3, axis=2), self.camconf.NEARVAL, self.camconf.FARVAL) 
        test_depth = plt.imread('/home/msieb/projects/gps-lfd/demo_data/bowl/depth/16_00001.png')
        #rgb_crop = rgb_img[object_p2d[0]-CROP_SZ_Y : object_p2d[0]+CROP_SZ_Y, object_p2d[1]-CROP_SZ_X : object_p2d[1]+CROP_SZ_X][:,:,:-1]
        #depth_crop = depth_img[object_p2d[0]-CROP_SZ_Y : object_p2d[0]+CROP_SZ_Y, object_p2d[1]-CROP_SZ_X : object_p2d[1]+CROP_SZ_X]
        # cv2.imshow('rgb_crop',rgb_crop)
        # # cv2.imshow('depth_crop',img_as_array[3][object_p2d[0]-CROP_SZ_Y : object_p2d[0]+CROP_SZ_Y, object_p2d[1]-CROP_SZ_X : object_p2d[1]+CROP_SZ_X])
        # k = cv2.waitKey(1)
 
        return rgb_img, depth_img

    def getObjectPose(self, object_id):
        pos, pose = p.getBasePositionAndOrientation(object_id)
        return list(pos) + list(pose)

    def create_writers(self, fps=10):
        if not os.path.exists(join(self.hyperparams['data_files_dir'], 'vids')):
            os.makedirs(join(self.hyperparams['data_files_dir'], 'vids'))
        rgb_path = join(self.hyperparams['data_files_dir'], 'vids', "rgb_sample_{0:03d}.mp4".format(self.vid_seqname))
        rgb_writer = imageio.get_writer(rgb_path, fps=fps)
        return rgb_writer

    def clip_actions(self, eepos, mj_U, deltas):
        if eepos[2] + mj_U[2] < 0.8200000357627868: # Prevent hitting table
            mj_U[2] = 0
        norm = np.sqrt((np.sum(mj_U[:3]**2)))
        if norm >= deltas[0]:
            mj_U[:3] = mj_U[:3]*deltas[0]/norm
            print("saturated action")
        norm_a = np.linalg.norm(mj_U[3])
        if norm_a >= deltas[1]:
            mj_U[3] = mj_U[3]*deltas[1] / norm_a
        norm_check = np.sqrt((np.sum(mj_U[:3] ** 2)))
        return mj_U

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)

