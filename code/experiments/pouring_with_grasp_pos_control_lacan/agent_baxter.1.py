""" This file defines an agent for the PR2 ROS environment. """
import copy
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import rospy
import tf
from geometry_msgs.msg import PointStamped

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
        END_EFFECTOR_POINT_JACOBIANS, ACTION, NOISE, OBJECT_POSE, IMAGE_FEATURE

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
from std_msgs.msg import (
    UInt16,
)
import baxter_interface
from baxter_pykdl import baxter_kinematics
from gps.utility.baxter_utils import img_subscriber, resize_frame
from gps.utility.utils import create_writer
from gps.utility.feature_utils import get_2d_depth_finger_trajectories, project_point_to_pixel, get_unprojected_3d_trajectory, get_unprojected_3d_mean_finger_and_gripper_trajectory, get_mrcnn_features, EEFingerListener
# Setup GPU options
import tensorflow 

# import moveit_commander
# import moveit_msgs.msg
# from moveit_commander import conversions

# Mask RCNN imports 
from baxter.baxter_iccv import BaxterConfig
from mrcnn.config import Config
from mrcnn import visualize, utils
import mrcnn.model as modellib

# Custom modules
# from plans import BaxterInterfacePlanner

### CONSTANTS ###
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		0.0, 615.323974609375, 240.33250427246094, 
		0.0, 0.0, 1.0]).reshape(3, 3) # D435 intrinsics matrix
DEPTH_SCALE = 0.001
##################


class AgentBaxter(Agent):
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
        config = copy.deepcopy(AGENT_ROS)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_ros_node', disable_signals=True)
        # self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.
        self.ptconf, self.dconf = hyperparams['ptconf'], hyperparams['dconf']

        conditions = self._hyperparams['conditions']

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        self.x0 = self._hyperparams['x0']


        self.use_tf = False
        self.observations_stale = True

        # Baxter specific
        limb = self._hyperparams['trial_arm']
        self._trial_limb = limb
        self._trial_arm = baxter_interface.Limb(self._trial_limb)
        self._trial_gripper = baxter_interface.Gripper(self._trial_limb, baxter_interface.CHECK_VERSION)
        # self._trial_gripper.set_moving_force(30)
        # self._trial_gripper.set_holding_force(30)
        # self._trial_gripper.command_position(100)
        print('\nplace object in gripper!!!\n')
        rospy.sleep(1)
        self._trial_gripper.set_moving_force(0.1)
        self._trial_gripper.set_holding_force(0.1)
        self._auxiliary_limb = 'right'
        self._auxiliary_arm = baxter_interface.Limb(self._auxiliary_limb)
        self._auxiliary_gripper = baxter_interface.Gripper(self._auxiliary_limb, baxter_interface.CHECK_VERSION)

        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._pub_rate = rospy.Publisher('robot/joint_state_publish_rate',
                                         UInt16, queue_size=10)
        self._pub_rate.publish(100)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
      
        self.joint_names = self._trial_arm.joint_names()
        self.n_joints = len(self.joint_names)
        # self.planner = BaxterInterfacePlanner(limb)
        self._rate = 1.0 / self._hyperparams['dt']
        self._control_rate = rospy.Rate(self._rate)
        self._missed_cmds = 20.0
        self._trial_arm.set_command_timeout((1.0 / self._rate) * self._missed_cmds)
        self._kin_aux = baxter_kinematics('right')
        self._kin_trial = baxter_kinematics('left')
        self.idx_curr_rollout = 0
        self.idx_curr_itr = 0   
        self.take_video = False

        # self.reset_condition = hyperparams['reset_condition']

        # Reset auxiliary arm
        self.reset_arm(arm=self._auxiliary_arm, mode='ANGLE', data=self._hyperparams['aux_arm_joint_vals'])
        #
        self.views_ids = self._hyperparams['views']
        topic_img_list = ['/camera' + device_id + '/color/image_raw' for device_id in self.views_ids]
        topic_dimg_list = ['/camera' + device_id + '/aligned_depth_to_color/image_raw' for device_id in self.views_ids]
       
        # pub = rospy.Publisher('/tcn/embedding', numpy_msg(Floats), queue_size=3)
        self.img_subs_list = [img_subscriber(topic=topic_img) for topic_img in topic_img_list]
        self.dimg_subs_list = [img_subscriber(topic=topic_dimg) for topic_dimg in topic_dimg_list]

        self.finger_listener = EEFingerListener()
        # Jacobian
        # Jacobian Pseudo-Inverse (Moore-Penrose)
        self.gripper_movement_cnt = 0

        # MRCNN variables
        self.mrcnn = self._hyperparams['mrcnn_model']
        self.class_names = self._hyperparams['class_names']
        self.target_ids = self._hyperparams['target_ids']
        # self.fig, self.ax = visualize.get_ax()
        self.fig, self.axes = self._hyperparams['figure_axes']
        self.centroids_last_known = {id: None for id in self.target_ids}
        self.all_object_traj = {id: np.empty((0, 3)) for id in self.target_ids}
        self.total_runs = 0
        self.listener = tf.TransformListener()
        self.start_orn0 = list(self._trial_arm.endpoint_pose()['orientation'])

        # DON Features
        self.feature_model = self._hyperparams ['feature_model'] 
        self.feature_fn = self._hyperparams['feature_fn']
        self.last_known_feature = None
        self.feature_traj = []
        cv2.namedWindow('target')
        cv2.namedWindow('query')
        cv2.namedWindow('mask1')

        # Metrics
        self.geom_dist_ee_to_anchor_traj = []
        self.geom_dist_object2_to_anchor_traj = []
        

    def set_cartesian_velocity(self, velocity_pos, velocity_orn=[0, 0, 0], rot_gripper=None):
        velocity = np.concatenate([np.array(velocity_pos), np.array(velocity_orn)])
        joint_vels = self.jacobian_pseudo_inverse.dot(velocity)[0, :]
        assert joint_vels.shape == (1, 7)
        cmd = {}    
        for idx, name in enumerate(self.joint_names):
                cmd[name] = joint_vels[0, idx]
                if name == 'left_w2' and rot_gripper is not None:
                    cmd[name] += rot_gripper
        self._trial_arm.set_joint_velocities(cmd)



    def _reset_control_modes(self):
        rate = rospy.Rate(100)
        for _ in range(100):
            if rospy.is_shutdown():
                return False
            self._auxiliary_arm.exit_control_mode()
            self._trial_arm.exit_control_mode()
            self._pub_rate.publish(100)
            rate.sleep()
        return True

    def clean_shutdown(self):
        print("\nExiting example...")
        self.set_cartesian_velocity([0, 0, 0.0])
        #return to normal
        self._reset_control_modes()
        if not self._init_state:
            print("Disabling robot...")
            self._rs.disable()
        return True

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

        if not np.allclose(joints_curr, relax_position, atol=0.03):
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
            zipped_joints = {key: val for key, val in zip(arm.joint_names(), data)}
            # self._trial_arm.set_joint_positions(zipped_joints)
            arm.move_to_joint_positions(zipped_joints)
            self._control_rate.sleep()
            joints_curr = [self._trial_arm.joint_angle(j)
                            for j in self.joint_names]
            if not np.allclose(joints_curr, data, atol=0.03):
                # print("error reaching reset position")
                pass

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
        relax_position =  [-0.06672816427301549, -0.6883738785635795, -1.074170046716761, 1.57539826915832, 0.7267233982607147, 1.1493351053231462, 1.0032234352770606]
        # self.reset_arm(arm=self._trial_arm, mode='ANGLE', data=relax_position)
        self.reset_arm(arm=self._trial_arm, mode='ANGLE', data=self._hyperparams['x0joints'][0])
        self.reset_arm(arm=self._auxiliary_arm, mode='ANGLE', data=self._hyperparams['aux_arm_joint_vals'])

        # self.relax_arm(self._auxiliary_arm)
        self._control_rate.sleep()  # useful for the real robot, so it stops completely
        self._trial_gripper.command_position(self._hyperparams['gripper_reset_position']) # open gripper at reset (100 is fully opened)
        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])
        # print('reset EE pose: ', pos0, orn0)
        self._trial_gripper.command_position(0.0) # open gripper at reset (100 is fully opened)

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
        self.all_object_traj = {id: np.empty((0, 3)) for id in self.target_ids}
        if self.take_video:
            self.rgb_writer = create_writer(self._hyperparams['data_files_dir'], classifier='itr_{}'.format(self.idx_curr_itr), fps=5)

        X, new_sample = self._init_sample(condition)
        b_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        noise /= 1
        # Take the sample.
        for t in range(self.T):
            curr_time = rospy.get_time()
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            # print('cost: {}'.format(np.linalg.norm(self._hyperparams['cost_tgt'][t, :6] - X[:6])))

            b_U = policy.act(X_t, obs_t, t, noise[t, :])
            # print(b_U)
            U[t, :] = b_U
            if (t+1) % 10 == 0:
                print('sample policy action at t={}: {}'.format(t, b_U))
            b_U[0:3] = np.clip(b_U[0:3], -self._hyperparams['max_velocity'], self._hyperparams['max_velocity'])  # clip task space 
            b_U[3] = np.clip(b_U[3], -0.5, 0.5)  # clip rotation
            b_U[-1] = np.clip(b_U[-1], -30, 30) # clip gripper
            b_U *= self._hyperparams['set_action_to_zero']
            # print(_U)
            if (t + 1) < self.T:
                # b_X, b_U_check = self._step(b_U, cTruerr_time)
                # self._step_taskspace_vel(b_U, X, curr_time, t)
                self._step_taskspace(b_U, X, curr_time, t)
                X = self._set_sample(new_sample, t, condition)
        self._trial_gripper.command_position(self._hyperparams['gripper_reset_position']) # open gripper at reset (100 is fully opened)

        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)

        xx, image = self._get_current_state(t)
        if self.take_video:
            self.rgb_writer.append_data(image)
        self.axes[1, 0].clear()
        self.axes[1, 0].set_title('ee_to_anchor_distance')
        self.axes[1, 0].plot(np.array(self.geom_dist_ee_to_anchor_traj)[:, 0],c='r', label='x')
        self.axes[1, 0].plot(np.array(self.geom_dist_ee_to_anchor_traj)[:, 1], c='g', label='y')
        self.axes[1, 0].plot(np.array(self.geom_dist_ee_to_anchor_traj)[:, 2], c='b', label='z')
        self.axes[1, 0].legend()

        self.axes[1, 1].clear()
        self.axes[1, 1].set_title('object2_to_anchor_distance')
        self.axes[1, 1].plot(np.array(self.geom_dist_object2_to_anchor_traj)[:, 0],c='r', label='x')
        self.axes[1, 1].plot(np.array(self.geom_dist_object2_to_anchor_traj)[:, 1], c='g', label='y')
        self.axes[1, 1].plot(np.array(self.geom_dist_object2_to_anchor_traj)[:, 2], c='b', label='z')
        self.axes[1, 1].legend()    

        self.axes[2, 1].clear()
        self.axes[2, 1].set_title('pixel_feature_state')
        self.axes[2, 1].plot(np.array(self.feature_traj),c='r')
        self.axes[2, 1].legend()    


        self.geom_dist_ee_to_anchor_traj = []
        self.geom_dist_object2_to_anchor_traj = []
        self.feature_traj = []

        # print("reached endpoint: ",sxx[-6:].tolist())

        print('finished rollout {}'.format(self.idx_curr_rollout+1))

        self.idx_curr_rollout += 1
        if self.take_video:
            self.rgb_writer.close()

        if save:
            self._samples[condition].append(new_sample)
        
        return new_sample



    def _get_current_state(self, t):
        # U = [self._trial_arm.joint_effort(j) for j in self.joint_names]
        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        # dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        # dorn = list(self._trial_arm.endpoint_velocity()['angular'])
        vel = np.array(self._kin_trial.jacobian().dot(dq))[0]
        dpos = vel[:3]
        dorn = vel[3:]

        p3d_l = self.finger_listener.get_3d_pose(gripper='l', finger='l', frame="/camera{}_color_optical_frame".format(2))
        p3d_r = self.finger_listener.get_3d_pose(gripper='l', finger='r', frame="/camera{}_color_optical_frame".format(2))
        gripper_binary = np.array([int(np.linalg.norm(p3d_l - p3d_r) > 0.035)])
  
        for img_subs, dimg_subs in zip(self.img_subs_list, self.dimg_subs_list):
            image = img_subs.img   
            dimage_raw = dimg_subs.img
            # dimage = np.copy(dimage_raw)
            # clipping_distance = 1.5 / 0.001 
            # dimage[np.where(dimage > clipping_distance)] = 0
            # dimage = ((dimage  - 0) / (clipping_distance - 0)) * (255 - 0) + 0
            # image_buffer_curr[i] = image
            # image_buffers[i].append(image)

            break # do only one first view for now!
        
        # MRCNN values
        all_centroids, all_masks, _ = get_mrcnn_features(self.mrcnn, image, dimage_raw * DEPTH_SCALE, self.target_ids, self.class_names, ax=None)
        for id in self.all_object_traj.keys():
            if id in all_centroids.keys():
                self.all_object_traj[id] = np.vstack([self.all_object_traj[id], all_centroids[id]])
                self.centroids_last_known[id] = all_centroids[id] # assumes object is found in first frame at least  stamp: 
            else: # object wasnt found, use last known value
                self.all_object_traj[id] = np.vstack([self.all_object_traj[id], self.centroids_last_known[id]])

        # DON Features
        try:
            valid_target_ind = all_masks[self._hyperparams['anchor_id']]
            # a_pred = np.array(self.feature_fn(self.feature_model, 
            #     np.reshape(self._hyperparams['cost_tgt_image_feature'][t], [-1, 2]), self._hyperparams['cost_tgt_image'][t], valid_target_ind,  image, self._hyperparams['n_max_correspondences'], query_centroid=self._hyperparams['all_mask_traj'][t][self._hyperparams['anchor_id']][:-1], , target_centroid=self.all_object_traj[self._hyperparams['anchor_id']][:-1],debug=True) ) 
            a_pred = np.array(self.feature_fn(self.feature_model, 
                np.reshape(self._hyperparams['cost_tgt_image_feature'][t], [-1, 2]), self._hyperparams['cost_tgt_image'][t], valid_target_ind,  image, self._hyperparams['n_max_correspondences'],debug=True) ) 

            image_feature = np.mean(np.squeeze(a_pred))[None]
            self.last_known_feature = image_feature
        except:
            print('mask not found for current frame {}'.format(t))
            image_feature = self.last_known_feature

        cv2.imshow('mask1', (valid_target_ind*255.0).astype(np.uint8))
        self.feature_traj.append(image_feature)
        # FIXME: why was there a t+1 in the original code from giraffe_mrcnn?
        # FIXME: Investigate scale of output scalar feature

        # EE position 3D and 2D
        ee_mean = (p3d_l + p3d_r)/2.0
        ee_2d = project_point_to_pixel(ee_mean, INTRIN)
        ee_dep = get_unprojected_3d_trajectory(np.concatenate([ee_2d, np.array([ee_mean[-1]])])[None], INTRIN)[0]
        
        # Object 3D
        anchor_deproj = get_unprojected_3d_trajectory(self.all_object_traj[self._hyperparams['anchor_id']][-1][None], INTRIN)[0]
        object2d_deproj = get_unprojected_3d_trajectory(self.all_object_traj[self._hyperparams['object2_id']][-1][None], INTRIN)[0]
        
        # Relative Distances
        geom_dist_ee_to_anchor = ee_dep - anchor_deproj
        geom_dist_object2_to_anchor = object2d_deproj - anchor_deproj
        self.geom_dist_ee_to_anchor_traj.append(geom_dist_ee_to_anchor)
        self.geom_dist_object2_to_anchor_traj.append(geom_dist_object2_to_anchor)
        
        # Gather all state features
        X = np.concatenate([geom_dist_ee_to_anchor, geom_dist_object2_to_anchor, pos, dpos, gripper_binary, q, dq, image_feature.flatten(), np.array([0.0])])

        plt.ion()
        plt.pause(0.001)               
        return X, image

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: funciton to comptue image features from the observation.
        """
        sample = Sample(self)
        # Initialize world/run kinematicscost
        jac = self._kin_trial.jacobian()
        X, image = self._get_current_state(t=0)
        if self.take_video:
            self.rgb_writer.append_data(image)
        # X = np.concatenate([geom_dist_ee_to_anchor, geom_dist_object2_to_anchor, pos, dpos, gripper_binary])
        geom_dist_ee_to_anchor = X[0:3]
        geom_dist_object2_to_anchor = X[3:6]
        pos = X[6:9]
        dpos = X[9:12]
        gripper_binary = X[12]
        q = X[13:20]
        dq = X[20:27]
        emb = X[27:28]

        sample.set(OBJECT_POSE, np.concatenate([geom_dist_ee_to_anchor, geom_dist_object2_to_anchor]), t=0)
        sample.set(JOINT_ANGLES, np.asarray(q), t=0)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=0)
        sample.set(END_EFFECTOR_POINTS, pos, t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=0)
        sample.set(IMAGE_FEATURE, np.asarray(emb), t=0)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=0)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=0)
        return X, sample
        
    def _set_sample(self, sample, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """
        jac = self._kin_trial.jacobian()
        X, image = self._get_current_state(t)
        if self.take_video:
            self.rgb_writer.append_data(image)

        # X = np.concatenate([geom_dist_ee_to_anchor, geom_dist_object2_to_anchor, pos, dpos, gripper_binary])
        geom_dist_ee_to_anchor = X[0:3]
        geom_dist_object2_to_anchor = X[3:6]
        pos = X[6:9]
        dpos = X[9:12]
        gripper_binary = X[12]
        q = X[13:20]
        dq = X[20:27]
        emb = X[27:28]

        sample.set(OBJECT_POSE, np.concatenate([geom_dist_ee_to_anchor, geom_dist_object2_to_anchor]), t=t+1)
        sample.set(JOINT_ANGLES, np.asarray(q), t=t+1)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=t+1)
        sample.set(END_EFFECTOR_POINTS, pos, t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=t+1)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=t+1)
        sample.set(IMAGE_FEATURE, np.asarray(emb), t=t+1)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=t+1)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=t+1)
        return X



    # def _step(self, U):
    #     within_bounds = self._within_bounds()

    #     if not self._rs.state().enabled:
    #         rospy.logerr("Joint torque example failed to meet "
    #                      "specified control rate timeout.")
    #     cmd = {}
    #     for idx, name in enumerate(self.joint_names):
    #             cmd[name] = np.clip(U[idx], vel0, vel1)
    #     if within_bounds or True:
    #         self._trial_arm.set_joint_velocities(cmd)

    #         # self._trial_arm.set_joint_torques(cmd)
    #     else:
    #         print("End effector is out of bounds")
    #     # self._control_rate.sleep()
    #     step_time = rospy.get_time() - curr_time
    #     # rospy.sleep(self._hyperparams['dt'] - step_time)
    #     X, U_ = self._get_current_state()

    #     return X, U

    def _step_taskspace_vel(self, U, X, curr_time, t):
        U_list = list(U)

        # U_list[0:3] = [a*int(b) for a, b in zip(U_list[0:3], within_bounds)]

        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])
        within_bounds = self._within_bounds_check(pos0)
        print([int(a) for a in within_bounds])
        U_list[0:3] = self._back_bounce(U_list[0:3], pos0)
        
        if len(U_list) > 3:
            rot_gripper = U_list[3]
        else: 
            rot_gripper = None
        step_time = rospy.get_time() - curr_time
        # print(step_time)
        self.set_cartesian_velocity(velocity_pos=U_list[0:3], rot_gripper=rot_gripper)
        cur_gripper_pos = self._trial_gripper.position()

        if self.gripper_movement_cnt % int(self._hyperparams['T'] / self._hyperparams['T']) == 0 or True:
            self._trial_gripper.command_position((np.sign(U[-1]) + 1) * 50)
        self.gripper_movement_cnt += 1

        # Fix gripper movement through demo
        self._trial_gripper.command_position(self._hyperparams['demo_gripper_trajectory'][t][0] * 0.0)
        step_time = rospy.get_time() - curr_time        
        rospy.sleep(self._hyperparams['dt'] - step_time)

        # X = self._get_current_state()
        return 

    def _step_taskspace(self, U, X, curr_time, t):
        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])
        U_list = list(U)
        within_bounds = self._within_bounds(U_list[0:3],pos0)
        # within_bounds = self._within_bounds_check(pos0)
        print(within_bounds)
        # print(X[6:9])
        # print(within_bounds)
        # U_list[0:3] = self._back_bounce(U_list[0:3])
        print('delta command: ',U_list[0:3])

        U_list[0:3] = [a*int(b) for a, b in zip(U_list[0:3], within_bounds)]
        U_list[0:3] = [a + b for a, b in zip(pos0, U_list[0:3])] # Check limits and additon
        # U_list[2] = -0.114954124178

        # U_list[0] = pos0[0]
        # U_list[1] = pos0[1]
        # U_list[2] = 0.02 + pos0[2]
        # pos[2] = -0.05
        # print('actual pos: ', pos0)
        # print('commanded pos: ',U_list[0:3])
    #     des_orn = [0.324114005664,
    # 0.944468221642,
    # 0.0488340121575,
    # 0.023347995574]
        joint_pos_rot_old = self._trial_arm.joint_angle('left_w2') 
        for i in range(5):
            joint_pos = self._kin_trial.inverse_kinematics(U_list[0:3], self.start_orn0)
            if joint_pos is not None:
                break
            else:
                pass
        
        if joint_pos is None:
            joint_pos = [self._trial_arm.joint_angle(j) for j in self.joint_names]
            print('IK failed, keep going :)')

        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
        cmd = {}
        for idx, name in enumerate(self.joint_names):
            cmd[name] = joint_pos[idx]
            if name == 'left_w2':
                cmd[name] = joint_pos_rot_old
                cmd[name] += U_list[3]
        self._trial_arm.set_joint_position_speed(0.5)
        # self._trial_arm.set_joint_positions(cmd)
        self._trial_arm.move_to_joint_positions(cmd, timeout=self._hyperparams['dt']*0.95)

        # print('step time: ', step_time)
        step_time = rospy.get_time() - curr_time        
        rospy.sleep(self._hyperparams['dt'] - step_time)
        cur_gripper_pos = self._trial_gripper.position()

        if self.gripper_movement_cnt % int(self._hyperparams['T'] / self._hyperparams['T']) == 0 or True:
            self._trial_gripper.command_position((np.sign(U[-1]) + 1) * 50)
        self.gripper_movement_cnt += 1

        # Fix gripper movement through demo
        self._trial_gripper.command_position(self._hyperparams['demo_gripper_trajectory'][t][0] * 0.0)

        return 

    def _within_bounds(self, U, pos):
        x0= 0.44
        x1= 0.9
        y0= -0.132
        y1= 0.24
        z0= -0.033
        z1= 0.2
        within_bounds = [(U[0] + pos[0]) < x1 and (U[0] + pos[0]) > x0,
                         (U[1] + pos[1]) < y1 and (U[1] + pos[1]) > y0,
                         (U[2] + pos[2]) < z1 and (U[2] + pos[2]) > z0]
        return within_bounds

    def _within_bounds_check(self, pos):
        x0= 0.44
        x1= 0.9
        y0= -0.132
        y1= 0.3
        z0= -0.033
        z1= 0.2
        within_bounds = [(pos[0]) < x1 and (pos[0]) > x0,
                         (pos[1]) < y1 and (pos[1]) > y0,
                         (pos[2]) < z1 and (pos[2]) > z0]
        return within_bounds

    def _back_bounce(self, U, pos):
        x0= 0.44
        x1= 0.9
        y0= -0.132
        y1= 0.24
        z0= -0.033
        z1= 0.2
        within_bounds = [pos[0] < x1 and pos[0] > x0,
                         pos[1] < y1 and pos[1] > y0,
                         pos[2] < z1 and pos[2] > z0]
        # print(within_bounds)
        if pos[0] <= x0:
            x_ = 0.025
        elif pos[0] >= x1:
            x_ = -0.025
        else:
            x_ = U[0]
        if pos[1] <= y0:
            y_ = 0.025
        elif pos[1] >= y1:
            y_ = -0.025
        else:
            y_ = U[1]
        if pos[2] <= z0:
            z_ = 0.001
        elif pos[2] >= z1:
            z_ = -0.001   
        else:
            z_ = 0     
        back_bounce_commands = [x_, y_, z_]
        return back_bounce_commands
