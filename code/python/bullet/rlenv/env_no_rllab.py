    ######### RL Lab API ###########
from copy import deepcopy as copy
import random
import math
import os
import struct
import sys

import numpy as np
from pdb import set_trace



import pybullet as p
import time
import argparse
import numpy as np
import pybullet_data
import os
from os.path import join
from utils import euclidean_dist, pressed, readLogFile, clean_line
import importlib
from algos import FiniteDifferenceLearner
from trace import Trace
import pyquaternion as pq
from transformation import get_H, transform_trajectory, get_rotation_between_vectors
from reward import DecisionTree

from pdb import set_trace


ATTACH_DIST = 0.12

DEMO_PATH = './demos'

class BulletEnv(object):
    
    def init(self, args, connect=True):
        self.args = args
        self._setup_world(connect)

    def _setup_world(self, connect=True):    
        """
        Setup bullet robot environment and load all relevant objects
        """
        args = self.args
        mode = args.mode
        module = importlib.import_module('simenv.' + args.env)
        envClass = getattr(module, 'UserEnv')
        self._env = envClass()
        self.ids = None

        cid = -1
        if connect:
            if args.init is None:
                cid = p.connect(p.SHARED_MEMORY)

            if (cid<0):
                if mode == 'GUI':
                    p.connect(p.GUI)
                else:
                    p.connect(p.DIRECT)
        p.resetSimulation()
        #disable rendering during loading makes it much faster
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Env is all loaded up here
        h, o = self._env.load()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.setGravity(0.000000,0.000000,0.000000)
        p.setGravity(0,0,-10)

        ##show this for 10 seconds
        #now = time.time()
        #while (time.time() < now+10):
        #   p.stepSimulation()
        # p.setRealTimeSimulation(1)  

    def initialize_world(self, all_traj, object_ids, object_reset_poses):

        self.num_total_steps = 0
        self.ids = object_ids
        self.init_traj_B = np.asarray(all_traj['5'])[:, 2:9]
        self.init_traj_A = np.asarray(all_traj['4'])[:, 2:9]
        self.init_policy = np.asarray(all_traj['2'])[:, [2,3,4, -1]][4:]
        self.object_ids = [4, 5]
        self.n_waypoints = self.init_policy.shape[0]
        self.waypoint_dimension = self.init_policy.shape[1]
        self.action_dimension = self.init_policy.flatten().shape[0]
        self.observation_dimension = self.action_dimension + len(self.object_ids) * 7
        self.object_reset_poses = object_reset_poses
        self.primitives = MovementPrimitives(self._env)
        self.goal_detector = DecisionTree().load()

    def _reset(self):
        self._env.reset() # TODO investigate complete reset
        # cov_perturb = np.random.multivariate_normal(np.zeros(7), np.diag([1,1,1,0,0,0,0]))
        for i, pose in self.object_reset_poses.items():
            self._env.setObjectPose(int(i), pose[:3], pose[3:7])
            self._env.o.kukaobject.open_gripper()


    def _episodic_step(self, action):

        def _get_episodic_reward(observation):

            # robot coordinates are 7 dimensions (pos + orn )
            # object coordinates are 7 dimensions (pos + orn)
            robot_obs = observation[:7]

            end_pose_B = observation[14:]
            end_pose_A = observation[7:14]
            delta_dist = np.linalg.norm(np.asarray(end_pose_B[:3]) - np.asarray(end_pose_A[:3]))
            reward = -delta_dist
            obj_pos_start = [val[:3].tolist() for key, val in self.object_reset_poses.items()]
            obj_pos_end = [end_pose_A[:3], end_pose_B[:3]]
            reward_tree = self.goal_detector.predict(obj_pos_start, obj_pos_end)
            reward = reward_tree[0]
            reward = 1 - reward
            if delta_dist > 0.2:
                reward = 0
            return reward, delta_dist
        # implements one step action of waypoints
        orn = None
        object_trajects = {key: [] for key in self.ids[1:]}
        robot_traj = []
        policy = action.reshape([self.n_waypoints, self.waypoint_dimension])
        # self.primitives.grasp(self.object_reset_poses['4'][:3])
        for i in range(policy.shape[0]):
            pos = policy[i, :-1]
            self._env.o.kukaobject.moveKukaEndtoPos(pos, orn)

            if np.abs(policy[i, -1]) > 0.015:
                self._env.o.kukaobject.close_gripper()
                
            else:
                self._env.o.kukaobject.open_gripper()
            p.stepSimulation()
            time.sleep(0.01)
            robot_traj.append(self._env.getEndEffectorPose())
            for key in self.ids[1:]:
                obj_pose = self._env.getObjectPose(int(key))
                object_trajects[key].append(obj_pose)

        # observations are given by final object configuration and robot configuration
        observation = []
        observation += robot_traj[-1]
        for key, traj in object_trajects.items():
            observation += traj[-1]
        # we are always done because we act over entire episode
        reward, delta_dist = _get_episodic_reward(observation)

        eps = 0.10

        done = True
        return observation, [reward, delta_dist], done

######  API wrappers ##############

    def step(self, action):
        """Executes step in environment
        """
        self.num_total_steps += 1

        # print ("====Executing step", self.num_total_steps, " of current run")

        observation, reward, done = self._episodic_step(action)
        print("reward: ", reward)
        return Step(observation=observation, reward=reward, done=done)

    def reset(self):
        """Resets and returns initial state of environment
        """
        print( "==== Reset environment")
        observation = self._reset()
        return observation



    # def _step(self, action):
    #     def _get_reward(observation):
    #         # robot coordinates are 7 dimensions (pos + orn )
    #         # object coordinates are 7 dimensions (pos + orn)
    #         robot_obs = observation[:7]

    #         pose_B = observation[14:]
    #         pose_A = observation[7:14]

    #         delta_dist = np.linalg.norm(pose_B[:3] - pose_A[:3])
    #         reward = -delta_dist
    #         return reward, delta_dist
    #     # implements one step action of waypoints
    #     orn = None
    #     object_obs = {key: [] for key in self.ids[1:]}
    #     policy = action.reshape([self.n_waypoints, self.action_dimension])
    #     pos = action[:-1]
    #     if np.abs(policy[i, -1]) > 0.015:
    #         self._env.o.kukaobject.close_gripper()
    #     else:
    #         self._env.o.kukaobject.open_gripper()
    #     self._env.o.kukaobject.instantMoveKukaEndtoPos(pos, orn)
    #     p.stepSimulation()
    #     # time.sleep(0.001)
    #     robot_obs  = self._env.getEndEffectorPose()
    #     for key in object_obs.keys():
    #         obj_pose = self._env.getObjectPose(int(key))
    #         object_obs[key].append(obj_pose)

    #     rollout = (robot_traj, object_obs)
    #     reward, delta_dist = _get_reward(rollout)
    #     # observations are given by final object configuration and robot configuration
    #     observation = []
    #     observation.append(robot_obs)
    #     for key, obs in object_obs:
    #         observation.append(obs)
    #     # we are always done because we act over entire episode
    #     eps = 0.05
    #     if delta_dist < eps:
    #         done = True
    #     else:
    #         done = False
    #     return observation, reward, done

class MovementPrimitives(object):
    def __init__(self, env):
        self.env = env

    def grasp(self, goal_pos):
        hover_pos = goal_pos + np.asarray([0,0,0.2])
        self.env.o.kukaobject.instantMoveKukaEndtoPos(hover_pos, None)
        p.stepSimulation()
        time.sleep(0.5)
        hover_pos = goal_pos + np.asarray([0,0,0.1])
        self.env.o.kukaobject.instantMoveKukaEndtoPos(hover_pos, None)
        p.stepSimulation()
        time.sleep(0.5)

        self.env.o.kukaobject.instantMoveKukaEndtoPos(goal_pos, None)
        p.stepSimulation()
        time.sleep(0.5)
