""" This file defines an agent for the PR2 ROS environment. """
import copy
import time
import numpy as np

import rospy

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

import baxter_interface
from baxter_pykdl import baxter_kinematics

import moveit_commander
import moveit_msgs.msg
from moveit_commander import conversions


# Custom modules
from plans import BaxterInterfacePlanner
from ipdb import set_trace



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
            rospy.init_node('gps_agent_ros_node')
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
        self.n_joints = len(self.joint_names)
        self.planner = BaxterInterfacePlanner(limb)
        self._rate = 1.0 / self._hyperparams['dt']
        self._control_rate = rospy.Rate(self._rate)
        self._missed_cmds = 20.0
        self._trial_arm.set_command_timeout((1.0 / self._rate) * self._missed_cmds)
        self._kin_aux = baxter_kinematics('right')
        self._kin_trial = baxter_kinematics('left')
    # def _init_pubs_and_subs(self):
    #     self._trial_service = ServiceEmulator(
    #         self._hyperparams['trial_command_topic'], TrialCommand,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )
    #     self._reset_service = ServiceEmulator(
    #         self._hyperparams['reset_command_topic'], PositionCommand,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )
    #     self._relax_service = ServiceEmulator(
    #         self._hyperparams['relax_command_topic'], RelaxCommand,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )
    #     self._data_service = ServiceEmulator(
    #         self._hyperparams['data_request_topic'], DataRequest,
    #         self._hyperparams['sample_result_topic'], SampleResult
    #     )

    # def _get_next_seq_id(self):
    #     self._seq_id = (self._seq_id + 1) % (2 ** 32)
    #     return self._seq_id

    # def get_data(self, arm=TRIAL_ARM):
    #     """
    #     Request for the most recent value for data/sensor readings.
    #     Returns entire sample report (all available data) in sample.
    #     Args:
    #         arm: TRIAL_ARM or AUXILIARY_ARM.
    #     """
    #     request = DataRequest()
    #     request.id = self._get_next_seq_id()
    #     request.arm = arm
    #     request.stamp = rospy.get_rostime()
    #     result_msg = self._data_service.publish_and_wait(request)
    #     # TODO - Make IDs match, assert that they match elsewhere here.
    #     sample = msg_to_sample(result_msg, self)
    #     return sample

    # TODO - The following could be more general by being relax_actuator
    #        and reset_actuator.



    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
        """
        relax_position =  [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]
        relax_position = self._hyperparams['x0'][0][:7]
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
        # move to initial state x0
        # self._trial_arm.move_to_neutral()
        if mode == 'ANGLE':
            zipped_joints = {key: val for key, val in zip(self.joint_names, data)}
            # self._trial_arm.set_joint_positions(zipped_joints)
            arm.move_to_joint_positions(zipped_joints)
            rospy.sleep(0.2)
            #TODO: Maybe verify that you reset to the correct position.
            joints_curr = [self._trial_arm.joint_angle(j)
                            for j in self.joint_names]
            if not np.allclose(joints_curr, x0_points, atol=0.01):
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
        self.relax_arm(self._trial_arm)
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
        # Take the sample.
        for t in range(self.T):
            curr_time = rospy.get_time()
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            if self._hyperparams['curr_runs'] < 5:
                b_U = self._hyperparams['u0'][t, :] 
                b_U +=  noise[t, :]
                 # print(b_U)

            else:
                b_U = policy.act(X_t, obs_t, t, noise[t, :])

            vel0 = -0.05
            vel1 = 0.05
            U[t, :] = b_U

            print "action", b_U
            b_U = np.clip(b_U, vel0, vel1) 
            if (t + 1) < self.T:
                # b_X, b_U_check = self._step(b_U, curr_time)
                b_X, b_U_check = self._step_taskspace(b_U, curr_time)
                
                self._set_sample(new_sample, b_X, t, condition)

        if self._hyperparams['curr_runs'] < 5:
            self._hyperparams['curr_runs'] += 1
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        xx, uu = self._get_current_state()
        print("reached endpoint: ",xx[14:17].tolist())
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

    def _step_taskspace(self, U, curr_time):
        U_list = list(U)
        within_bounds = self._within_bounds()

        pos0 = list(self._trial_arm.endpoint_pose()['position'])
        orn0 = list(self._trial_arm.endpoint_pose()['orientation'])
        pos = [a + b for a, b in zip(pos0, U_list)] # Check limits and additon
        for i in range(5):
            joint_pos = self._kin_trial.inverse_kinematics(pos, orn0)
            if joint_pos is not None:
                break
            else:
                print("inverse kinematic calculation failed")
        if not self._rs.state().enabled:
            rospy.logerr("Joint torque example failed to meet "
                         "specified control rate timeout.")
        cmd = {}
        for idx, name in enumerate(self.joint_names):
                cmd[name] = joint_pos[idx]
        if within_bounds:
            self._trial_arm.move_to_joint_positions(cmd, timeout=self._hyperparams['dt']*0.8)
        else:
            print("out of bounds")
            # self._trial_arm.set_joint_torques(cmd)
        # print('action: ', U)
        # print([self._trial_arm.joint_velocity(j) for j in self.joint_names])
        # self._control_rate.sleep()
        step_time = rospy.get_time() - curr_time
        rospy.sleep(self._hyperparams['dt'] - step_time)
        X, U_ = self._get_current_state()
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

        sample.set(JOINT_ANGLES, np.asarray(q), t=0)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=0)
        sample.set(END_EFFECTOR_POINTS, np.asarray(pos), t=0)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=0)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=0)
        #sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=0)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=0)
        return sample


    def _get_current_state(self):
        U = [self._trial_arm.joint_effort(j) for j in self.joint_names]

        q = [self._trial_arm.joint_angle(j) for j in self.joint_names]
        dq = [self._trial_arm.joint_velocity(j) for j in self.joint_names]
        pos = list(self._trial_arm.endpoint_pose()['position'])
        orn = list(self._trial_arm.endpoint_pose()['orientation'])
        dpos = list(self._trial_arm.endpoint_velocity()['linear'])
        dorn = list(self._trial_arm.endpoint_velocity()['angular'])
        # X = np.concatenate([q, dq, pos, orn, dpos, dorn])
        X = np.concatenate([q, dq, pos])

        return X, U

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

        sample.set(JOINT_ANGLES, np.asarray(q), t=t+1)
        sample.set(JOINT_VELOCITIES, np.asarray(dq), t=t+1)
        sample.set(END_EFFECTOR_POINTS, np.asarray(pos), t=t+1)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac[:3, :], t=t+1)

        # sample.set(END_EFFECTOR_ORIENTATIONS, np.asarray(orn), t=0)
   #     sample.set(END_EFFECTOR_POINT_VELOCITIES, np.asarray(dpos), t=0)
        # sample.set(END_EFFECTOR_ANGULAR_VELOCITIES, np.asarray(dorn), t=0)
        return


    def _within_bounds(self):
        X, U = self._get_current_state()
        pos = X[14:17]
        x0=  0.428863125303
        x1=  0.86521737946
        y0= -0.225733274891
        y1=  0.384777249352
        z0= -0.292697992966
        z1=  0.299107749513
        if pos[0] < x1 and pos[0] > x0 \
                and pos[1] < y1 and pos[1] > y0 \
                and pos[2] < z1 and pos[2] > z0:
            within_bounds = True
        else:
            within_bounds = False
        return within_bounds

    # def run_trial_tf(self, policy, time_to_run=5):
    #     """ Run an async controller from a policy. The async controller receives observations from ROS subscribers
    #      and then uses them to publish actions."""
    #     should_stop = False
    #     consecutive_failures = 0
    #     start_time = time.time()
    #     while should_stop is False:
    #         if self.observations_stale is False:
    #             consecutive_failures = 0
    #             last_obs = tf_obs_msg_to_numpy(self._tf_subscriber_msg)
    #             action_msg = tf_policy_to_action_msg(self.dU,
    #                                                  self._get_new_action(policy, last_obs),
    #                                                  self.current_action_id)
    #             self._tf_publish(action_msg)
    #             self.observations_stale = True
    #             self.current_action_id += 1
    #         else:
    #             rospy.sleep(0.01)
    #             consecutive_failures += 1
    #             if time.time() - start_time > time_to_run and consecutive_failures > 5:
    #                 # we only stop when we have run for the trial time and are no longer receiving obs.
    #                 should_stop = True
    #     rospy.sleep(0.25)  # wait for finished trial to come in.
    #     result = self._trial_service._subscriber_msg
    #     return result  # the trial has completed. Here is its message.

    # def _get_new_action(self, policy, obs):
    #     return policy.act(None, obs, None, None)

    # def _tf_callback(self, message):
    #     self._tf_subscriber_msg = message
    #     self.observations_stale = False

    # def _tf_publish(self, pub_msg):
    #     """ Publish a message without waiting for response. """
    #     self._pub.publish(pub_msg)

    # def _init_tf(self, dU):
    #     self._tf_subscriber_msg = None
    #     self.observations_stale = True
    #     self.current_action_id = 1
    #     self.dU = dU
    #     if self.use_tf is False:  # init pub and sub if this init has not been called before.
    #         self._pub = rospy.Publisher('/gps_controller_sent_robot_action_tf', TfActionCommand)
    #         self._sub = rospy.Subscriber('/gps_obs_tf', TfObsData, self._tf_callback)
    #         r = rospy.Rate(0.5)  # wait for publisher/subscriber to kick on.
    #         r.sleep()
    #     self.use_tf = True
    #     self.observations_stale = True
