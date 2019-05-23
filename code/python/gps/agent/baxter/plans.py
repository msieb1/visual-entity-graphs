#!/usr/bin/env python
'''
Simple Planner contains utils that help in planning.
Object Handler contains utils to add and remove collision objects.
'''
from copy import deepcopy as copy

import numpy as np
from pdb import set_trace
import rospy
import struct

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
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
import baxter_interface
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from moveit_commander import conversions
from moveit_msgs.msg import CollisionObject

from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


class BaxterInterfacePlanner(object):
    """
    Plans a trajectory given left or right limb and executes it based on 
    calculating IK with baxter interface not using Moveit API
    """
    def __init__(self, limb):
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        self._limb = baxter_interface.Limb(limb)
        self.gripper = baxter_interface.Gripper(limb)

    def _ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            print("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            # if self._verbose:
            #     print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
            #              (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            # if self._verbose:
            #     print("IK Joint Solution:\n{0}".format(limb_joints))
            #     print("------------------")
        else:
            print("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def ik_joint_and_gripper_plan_execution(self, waypoint_list):
        # executes a trajectory based on ik computation with baxter interface API 
        # if poses not as list of poses, will convert them into poses.
        # MUST BE PASSED A ZIPPED LIST OF JOINTS AND GRIPPERS
        # Example:
        #--      pose_action = [Pose1, Pose2, Pose3, ..., PoseN]
        # --     gripper_action = [0] * len(pose_action)
        # --     waypoint_list = zip(pose_action, gripper_action)
        # --     self.ik_planner.ik_joint_plan_execution(waypoint_list)
        for waypoint in waypoint_list:
            pose = waypoint[0]
            gripper_pos = waypoint[1]
            if type(pose) is list:
                pose = conversions.list_to_pose(pose)
            goal_joints = self._ik_request(pose)
            if not goal_joints:
                print "no ik solution found, terminate further execution and return False"
                return 1
            else:
                self._limb.move_to_joint_positions(goal_joints, timeout=5.0)
                self.gripper.command_position(gripper_pos)
                rospy.sleep(0.2)
        return 0

    def move_to_cartesian_pose(self, pose, gripper_pos=None):
        """Moves to specified pose or returns False if IK computation failed, can take gripper pose as additional argument

        Parameters
        ------
        pose : Pose object

        gripper_pos : float

        Returns
        ------
        boolean (success or not)

        """
        if len(pose) == 3:
            pose += [0,1,0,0]

        goal_joints = self._ik_request(pose)
        if not goal_joints:
            print "no ik solution found"
            return 1
        else:
            self._limb.move_to_joint_positions(goal_joints, timeout=5.0)
            if gripper_pos is not None:
                # if gripper argument not provided, leave gripper at current position
                self.gripper.command_position(gripper_pos)
            rospy.sleep(0.2)
            return 0

    def set_joint_positions(self, positions):
        names = self._limb.joint_names()
        command = dict(zip(names, positions))
        return self._limb.set_joint_positions(command)

    def move_gripper(self, gripper_pos):
        self.gripper.command_position(gripper_pos)
        return 0

    def get_joint_positions(self):
        return self._limb.joint_angles().values()

    def get_gripper_position(self):
        return self.gripper.position()

    def get_current_endpose(self):
        """Yields current endeffector pose as Pose object
        """
        endpoint_pose = self._limb.endpoint_pose()
        curr_pose = Pose(position=endpoint_pose['position'], orientation=endpoint_pose['orientation'])
        return curr_pose

    def get_current_endpoint_wrench(self):
        effort = self._limb.endpoint_effort()
        torque = effort['wrench']
        force = effort['force']
        return torque, force
        
    def set_joint_position_speed(self, speed):
        self._limb.set_joint_position_speed(speed)
        return 0



class SimplePlanner(object):
    '''
    Utils for planning.
    '''
    def __init__(self, group, planning_time=1.0):
        self._pt = planning_time
        self.group = group

    def any_plan(self, goal, **kwargs):
        assert len(goal) == 7, 'goal should be a list describing pose'
        pose_target = copy(goal)    
        pose_target = conversions.list_to_pose(pose_target)
        self.group.set_planning_time(self._pt)
        plan = self.group.plan(pose_target)
        return plan
     
    def cartesian_plan(self, waypoint_list, interpolation=0.01, jump_threshold=0.0):
        waypoints = [self.group.get_current_pose().pose]
        if type(waypoint_list[0]) is list:
            waypoints = waypoints + [conversions.list_to_pose(copy(waypoint)) for waypoint in waypoint_list]
        else:
            waypoints = waypoints + conversions.list_to_pose(copy(waypoint_list))
        self.group.set_planning_time(self._pt)
        plan, fraction = self.group.compute_cartesian_path(waypoints, interpolation, jump_threshold)
        rospy.sleep(self._pt)
        return plan,fraction

    def joint_plan(self, goal, **kwargs):
        assert len(goal) == 7, 'goal should be a list describing joint_angles'
        joint_target = copy(goal)
        self.group.set_joint_value_target(joint_target)
        self.group.set_planning_time(self._pt)
        plan = self.group.plan()
        return plan

    def multi_tries(self, goal, plan_type='any', num_tries=5, interp=0.01, jump=0.0):
        plan_length_threshold = 20
        plans = {
                'any' : self.any_plan,
                'cartesian' : self.cartesian_plan,
                'joint' : self.joint_plan
                }
        plan_method = plans[plan_type]
        best_plan_length = pow(10,8)
        best_plan_fraction = 0.0
        best_plan = None
        for tries in range(num_tries):
            out = plan_method(goal, interpolation=interp, jump_threshold=jump)
            if type(out) is tuple:
                temp_plan = out[0]
                temp_fraction = out[1]
            else:
                temp_plan = out
                temp_fraction = 1.0
            temp_plan_length = len(temp_plan.joint_trajectory.points)
            if best_plan_fraction>temp_fraction or temp_plan_length==0:
                continue
            elif (temp_fraction > best_plan_fraction) or (temp_plan_length < best_plan_length):
                best_plan_fraction = temp_fraction
                best_plan = temp_plan
                best_plan_length = temp_plan_length

            if best_plan_length<plan_length_threshold and best_plan_fraction==1.0:
                break
        return best_plan, best_plan_fraction

    def reverse_plan(self, init_plan):
        raise NotImplementedError
        cur_plan = copy(init_plan)
        final_plan = copy(init_plan)

        nPoints = len(init_plan.joint_trajectory.points)
        for point in range(0, nPoints):
            final_plan.joint_trajectory.points[point].positions = \
                    init_plan.joint_trajectory.points[nPoints-point-1].positions
        return final_plan

    def speed_plan(self, init_plan, spd=1.0):
        '''
        speed up the plan by spd
        '''
        assert spd>0., 'Speed should be greater than 0.'
        n_points = len(init_plan.joint_trajectory.points)
        assert n_points>0, 'The input plan should have atleast one point.'
        new_plan = copy(init_plan)
        for i in range(n_points):                                               
            new_plan.joint_trajectory.points[i].time_from_start = \
                    new_plan.joint_trajectory.points[i].time_from_start/spd
            new_plan.joint_trajectory.points[i].velocities = \
                    tuple(np.array(new_plan.joint_trajectory.points[i].velocities)*spd)
            new_plan.joint_trajectory.points[i].accelerations = \
                    tuple(np.array(new_plan.joint_trajectory.points[i].accelerations)*spd)
            new_plan.joint_trajectory.points[i].positions=new_plan.joint_trajectory.points[i].positions
        return new_plan

    def plan_to_trajectory_dict(self, plan):
        '''
        Convert plan to a dictionary that contains information about the plan.
        This was created mainly to get the end effector trajectory from the joint trajectory
        '''
        resps = []
        n_points = len(plan.joint_trajectory.points)
        timestamps = [p.time_from_start.to_sec() for p in plan.joint_trajectory.points]
        positions = [p.positions for p in plan.joint_trajectory.points]
        velocities = [p.velocities for p in plan.joint_trajectory.points]
        accelerations = [p.accelerations for p in plan.joint_trajectory.points]
        ns = "ExternalTools/" + "right" + "/PositionKinematicsNode/FKService"
        fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
        fkreq = SolvePositionFKRequest()
        for pos in positions:
            joints = JointState()
            joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                           'right_j4', 'right_j5', 'right_j6']
            joints.position = list(pos)
            fkreq.configuration.append(joints)
            fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(ns, 5.0)
            resp = fksvc(fkreq)
            endeffectors = np.array([conversions.pose_to_list(r.pose) for r in resp.pose_stamp])
        except (rospy.ServiceException, rospy.ROSException), e:
            resp = None
            rospy.logerr("Service call failed: %s" % (e,))
            endeffectors = None
        traj_dict = {
                        'time_steps' : timestamps,
                        'positions' : positions,
                        'velocities' : velocities,
                        'accelerations' : accelerations,
                        'end_effectors' : endeffectors,
        }
        return traj_dict

    def execute(self, plan):
        '''
        executes plan
        '''
        self.group.execute(plan)
        return 0
