import numpy as np
import os
from copy import deepcopy as copy
import multiprocessing as mp
from pdb import set_trace
import pybullet as p
import time

np.set_printoptions(precision=4)

class RandomSampler(object):
    def __init__(self, env, init_policy, n_iter=20, gamma=0.99):
        self.gamma = gamma
        self.env = env
        # Use waypoints as direct policy parametrization
        self.init_policy = init_policy
        self.n_iter = n_iter

    def train(self, n_parallel=1):
        cutoff = 0
        std = 0.003
        if n_parallel > 1:
            covariance= np.eye(self.init_policy.shape[0]) * std
            covariance[3::4] = 0 # do not sample gripper
            covariance[2::4] = 0 # do not sample z
            covariance[0:cutoff] = 0
            covariance[0:4] = 0
            for i in range(1,self.n_iter):
                print("="*20)
                print("Start iteration ", i)
                perturbances = np.random.multivariate_normal(np.zeros(self.init_policy.shape), covariance, n_parallel)
                pool = mp.Pool(processes=n_parallel)
                results = [pool.apply_async(self.rollout, args=(pert,)) for pert in perturbances]
                infos = [p.get() for p in results]
                for info in infos:
                    print('reward: ', info['reward'],' | end distance: ', info['delta_end_distance'])
                pool.close()
        else:
            for i in range(self.n_iter):
                if i == 0:
                    info = self.rollout(0.0)
                else:
                    info = self.rollout(0.005)
                print('reward: ', info['reward'],' | end distance: ', info['delta_end_distance'])





    def rollout(self, perturbance=None):
        self.env._reset()

        goal_achieved = False
        if perturbance is not None:
            policy = self.init_policy + perturbance
        else:
            policy = copy(self.init_policy)
        observation, reward, done = self.env._episodic_step(policy)
        if reward[0]  == 1: 
            goal_achieved = True
        info = dict()
        info['observation'] = observation
        info['reward'] = reward[0]
        info['delta_end_distance'] = reward[1]
        info['goal_achieved'] = goal_achieved
        return info


class FiniteDifferenceLearner(object):
    def __init__(self, env, init_policy, n_iter=5, gamma=0.99, l_rate=1.0, reward_scaling=0.1, n_ref_rollouts=10):
        self.gamma = gamma
        self.env = env
        # Use waypoints as direct policy parametrization
        self.init_policy = init_policy
        self.curr_policy = copy(init_policy)
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.reward_scaling = reward_scaling
        # get unnoisy estimate of reference return
        self.n_ref_rollouts = n_ref_rollouts
        self.primitives = MovementPrimitives(self.env._env)

    def train(self, n_parallel=1):
        print("Start Training...")
        cutoff = 10
        std = 0.003
        if n_parallel > 1:
            pool = mp.Pool(processes=n_parallel)

            covariance= np.eye(self.curr_policy.shape[0]) * std
            covariance[3::4] = 0 # do not sample gripper
            covariance[2::4] = 0 # do not sample z
            covariance[cutoff:0] = 0

            for i in range(1,self.n_iter):
                print("="*20)
                print("Start iteration ", i)
                # get reference return of current policy
                results = [pool.apply_async(self.rollout) for i in range(self.n_ref_rollouts)]
                infos = [p.get() for p in results]                    
                Js_ref = np.asarray([info['reward'] for info in infos])
                print('Js_ref: ', Js_ref)
                # if one is successful, original trajectory was already good
                if np.mean(Js_ref) > 0:
                    J_ref = 1
                else:
                    J_ref = 0
                print("J_ref: ", J_ref)
                # calculate perturbed rollout returns
                perturbances = np.random.multivariate_normal(np.zeros(self.curr_policy.shape), covariance, n_parallel)
                results = [pool.apply_async(self.rollout, args=(pert,)) for pert in perturbances]
                infos = [p.get() for p in results]
                Js_new = np.asarray([info['reward'] for info in infos])
                Js_delta = self.reward_scaling * (Js_new - J_ref)
                # calculate gradient: g_FD = (deltaTheta.T * deltaTheta)^{-1} *  deltaTheta.T * deltaJ
                gradient = np.linalg.pinv(perturbances.T.dot(perturbances)).dot(perturbances.T).dot(Js_delta)
                print("calculated gradient: ", gradient)

                # update policy with scaled gradient
                self.curr_policy += self.l_rate * gradient
                for info in infos:
                    print('reward: ', info['reward'],' | end distance: ', info['delta_end_distance'])
            pool.close()

        else:
            avg_reward = []
            for i in range(self.n_iter):
                if i == 0:
                    info = self.rollout(0.0)
                else:
                    info = self.rollout(0.005)
                print('reward: ', info['reward'],' | end distance: ', info['delta_end_distance'])
                avg_reward.append(info['reward'])
            return np.mean(avg_reward)


    def rollout(self, perturbance=None):
        self.env._reset()

        goal_achieved = False
        goal_pos = self.env.object_reset_poses['4'][:3]
        self.primitives.grasp(goal_pos)
        if perturbance is not None:
            policy = self.init_policy + perturbance
        else:
            policy = copy(self.init_policy)
        observation, reward, done = self.env._episodic_step(policy)
        if reward[0]  == 1: 
            goal_achieved = True
        info = dict()
        info['observation'] = observation
        info['reward'] = reward[0]
        info['delta_end_distance'] = reward[1]
        info['goal_achieved'] = goal_achieved
        return info


class MovementPrimitives(object):
    def __init__(self, env):
        self.env = env

    def grasp(self, goal_pos):
        hover_pos = goal_pos + np.asarray([0,0,0.3])
        self.env.o.kukaobject.moveKukaEndtoPos(hover_pos, None)
        hover_pos = goal_pos + np.asarray([0,0,0.15])
        self.env.o.kukaobject.moveKukaEndtoPos(hover_pos, None)

        # hover_pos = goal_pos + np.asarray([0,0,0.1])
        # self.env.o.kukaobject.instantMoveKukaEndtoPos(hover_pos, None)
        # p.stepSimulation()
        # pickup_pos = goal_pos + np.asarray([0,0,0.05])
        # self.env.o.kukaobject.moveKukaEndtoPos(goal_pos, None)
  
