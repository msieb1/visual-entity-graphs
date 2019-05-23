""" This file defines an agent for the PR2 ROS environment. """
import copy
import time
import numpy as np

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
        END_EFFECTOR_POINT_JACOBIANS, ACTION, NOISE

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

# import moveit_commander
# import moveit_msgs.msg
# from moveit_commander import conversions


# Custom modules
# from plans import BaxterInterfacePlanner
from ipdb import set_trace as st

class EEFingerListener(object):
    def __init__(self):
        self.listener = tf.TransformListener()

    def get_3d_pose(self, gripper='l', finger='r', frame=None):
        assert(gripper == 'r' or gripper == 'l')
        assert(finger == 'l' or finger == 'r')
        # camera frame is usually /camera{camID}_color_optical_frame (camID=2 in ourcase)
        self.listener.waitForTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0), rospy.Duration(4.0))
        (trans,rot) = self.listener.lookupTransform("/base", "/{}_gripper_{}_finger_tip".format(gripper, finger), rospy.Time(0))
        p3d=PointStamped()
        p3d.header.frame_id = "base"
        p3d.header.stamp =rospy.Time(0)
        p3d.point.x=trans[0]
        p3d.point.y=trans[1]
        p3d.point.z=trans[2]
        if frame is not None:
            self.listener.waitForTransform("/base", frame, rospy.Time(0),rospy.Duration(4.0))
            p3d_transformed = self.listener.transformPoint(frame, p3d)
        p3d_transformed = np.array([p3d_transformed.point.x, p3d_transformed.point.y, p3d_transformed.point.z])
        return p3d_transformed   

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
        # self.reset_condition = hyperparams['reset_condition']

        self.finger_listener = EEFingerListener()

        # Jacobian Pseudo-Inverse (Moore-Penrose)
        self.jacobian_pseudo_inverse =  self._kin_trial.jacobian_pseudo_inverse()

        self.gripper_movement_cnt = 0
     
    def set_cartesian_velocity(self, velocity_pos, velocity_orn=[0, 0, 0]):
        velocity = np.concatenate([np.array(velocity_pos), np.array(velocity_orn)])
        joint_vels = self.jacobian_pseudo_inverse.dot(velocity)[0, :]
        assert joint_vels.shape == (1, 7)
        cmd = {}    
        for idx, name in enumerate(self.joint_names):
                cmd[name] = joint_vels[0, idx]
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
        self._trial_gripper.command_position(self._hyperparams['gripper_reset_position']) # open gripper at reset (100 is fully opened)

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
        noise /= 1
        # Take the sample.
        for t in range(self.T):
            curr_time = rospy.get_time()
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            print('cost: {}'.format(np.linalg.norm(self._hyperparams['cost_tgt'][t, :-1] - X_t[-4:-1])))
            print('state:', X_t[-4:-1])
            print('demo:',self._hyperparams['cost_tgt'][t, :-1])

            if False and self._hyperparams['curr_runs'] < 5:
                b_U = self._hyperparams['u0'][t, :] 
                b_U +=  noise[t, :]
                 # print(b_U)

            else:
                b_U = policy.act(X_t, obs_t, t, noise[t, :])

            vel0 = 0.05
            U[t, :] = b_U

            # b_U[0] = 0
            # b_U[2] = 0
            b_U[0:-1] = np.clip(b_U[0:-1], -vel0, vel0)  # clip task space deltas
            b_U[-1] = np.clip(b_U[-1], -30, 30) # clTruep gripper
            b_U[-1] = 0
            # b_U[2] = 0
            # print(b_U)
            if (t + 1) < self.T:
                # b_X, b_U_check = self._step(b_U, cTruerr_time)
                b_X, b_U_check = self._step_taskspace(b_U, curr_time, t)
                
                self._set_sample(new_sample, b_X, t, condition)

        # if self._hyperparams['curr_runs'] < 5:
        #     self._hyperparams['curr_runs'] += 1
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        xx = self._get_current_state()
        # print("reached endpoint: ",sxx[-6:].tolist())
        print('finished rollout')
        self.reset()

        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _step(self, U):
        within_bounds = self._within_bounds()

        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
        cmd = {}
        for idx, name in enumerate(self.joint_names):
                cmd[name] = np.clip(U[idx], vel0, vel1)
        if within_bounds or True:
            self._trial_arm.set_joint_velocities(cmd)

            # self._trial_arm.set_joint_torques(cmd)
        else:
            print("End effector is out of bounds")
        # self._control_rate.sleep()
        step_time = rospy.get_time() - curr_time
        rospy.sleep(self._hyperparams['dt'] - step_time)
        X, U_ = self._get_current_state()

        return X, U

    def _step_taskspace(self, U, curr_time, t):
        U_list = list(U)
        within_bounds = self._within_bounds()
        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])
        U_list = [a*int(b) for a, b in zip(U_list, within_bounds)]
        step_time = rospy.get_time() - curr_time
        # print(step_time)
        self.set_cartesian_velocity(velocity_pos=U_list)
        rospy.sleep(self._hyperparams['dt'] - step_time)
        cur_gripper_pos = self._trial_gripper.position()

        if self.gripper_movement_cnt % int(self._hyperparams['T'] / self._hyperparams['T']) == 0 or True:
            self._trial_gripper.command_position((np.sign(U[-1]) + 1) * 50)
        self.gripper_movement_cnt += 1

        # Fix gripper movement through demo
        self._trial_gripper.command_position(self._hyperparams['cost_tgt'][t, -1] * 100.0)
        step_time = rospy.get_time() - curr_time        
        X = self._get_current_state()
        return X, U

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
        p3d_l = self.finger_listener.get_3d_pose(gripper='l', finger='l', frame="/camera{}_color_optical_frame".format(2))
        p3d_r = self.finger_listener.get_3d_pose(gripper='l', finger='r', frame="/camera{}_color_optical_frame".format(2))
        gripper_binary = np.array([int(np.linalg.norm(p3d_l - p3d_r) > 0.035)])

        sample.set(JOINT_ANGLES, np.asarray(q), t=0)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=0)
        sample.set(END_EFFECTOR_POINTS, np.concatenate([(p3d_l + p3d_r)/2.0, gripper_binary]), t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=0)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=0)
        #sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=0)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=0)
        return sample


    def _get_current_state(self):
        # U = [self._trial_arm.joint_effort(j) for j in self.joint_names]

        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        dorn = list(self._trial_arm.endpoint_velocity()['angular'])


        p3d_l = self.finger_listener.get_3d_pose(gripper='l', finger='l', frame="/camera{}_color_optical_frame".format(2))
        p3d_r = self.finger_listener.get_3d_pose(gripper='l', finger='r', frame="/camera{}_color_optical_frame".format(2))
        gripper_binary = np.array([int(np.linalg.norm(p3d_l - p3d_r) > 0.035)])
  
        # X = np.concatenate([q, dq, pos, orn, dpos, dorn])
        X = np.concatenate([q, dq, pos, (p3d_l + p3d_r)/2.0, gripper_binary])
        # print(np.concatenate([p3d_l, p3d_r]))
        return X

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
        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        dorn = list(self._trial_arm.endpoint_velocity()['angular'])
        jac = self._kin_trial.jacobian()
        p3d_l = self.finger_listener.get_3d_pose(gripper='l', finger='l', frame="/camera{}_color_optical_frame".format(2))
        p3d_r = self.finger_listener.get_3d_pose(gripper='l', finger='r', frame="/camera{}_color_optical_frame".format(2))
        gripper_binary = np.array([int(np.linalg.norm(p3d_l - p3d_r) > 0.035)])

        sample.set(JOINT_ANGLES, np.asarray(q), t=t+1)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=t+1)
        sample.set(END_EFFECTOR_POINTS, np.concatenate([(p3d_l + p3d_r)/2.0, gripper_binary]), t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=t+1)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=0)
   #     sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=0)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=0)
        return


    def _within_bounds(self):
        X = self._get_current_state()
        pos = X[14:17]
        x0=  0.4488863125303
        x1=  0.78521737946
        y0= -0.083733274891
        y1=  0.374777249352
        z0= -0.040697992966
        z1=  0.24107749513
        within_bounds = [pos[0] < x1 and pos[0] > x0,
                         pos[1] < y1 and pos[1] > y0,
                         pos[2] < z1 and pos[2] > z0]
        return within_bounds
