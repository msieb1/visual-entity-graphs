import argparse
import sys
sys.path.append('./rlenv')

import numpy as np
import pybullet as p
import time
import argparse
import numpy as np
import pybullet_data
from copy import deepcopy as copy
import os
from os.path import join
from utils import euclidean_dist, pressed, readLogFile, clean_line
import importlib
from algos import FiniteDifferenceLearner, RandomSampler
from trace import Trace
import pyquaternion as pq
from transformation import get_H, transform_trajectory, get_rotation_between_vectors

from rllab.algos.trpo import TRPO
from rllab.algos.ddpg import DDPG
from rllab.algos.cma_es import CMAES

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import DeterministicMLPPolicy
# from rllab.policies.gaussian_control_policy import GaussianControlPolicy

from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from rllab_env import BulletEnv

from pdb import set_trace

import tensorflow as tf



from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides

DEMO_PATH = '/home/msieb/projects/lang2sim/bullet/demos'
DISCRETIZATION = 200

import random
random.seed(100)

class GaussianControlPolicy(Policy, Serializable):
    """
    Implements a state independent Gaussian policy with means as output, 
    where the output dimension equals the action space (if entire waypoint trajectory is to be predicted at once)
    """
    def __init__(
            self,
            env_spec,
            init_mean,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianControlPolicy, self).__init__(env_spec=env_spec)
        self.params = init_mean
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim

    @overrides
    def get_action(self, observation):
        return self.params, dict()# +  np.ndarray.tolist(np.random.multivariate_normal(np.zeros(self.action_dim), cov=0.0001*np.eye(self.action_dim, self.action_dim))), dict()

    @overrides
    def get_param_values(self):
        return self.params

    @overrides
    def set_param_values(self, updated_params):
        self.params = updated_params
        return 0

        
def main(args):
    # Command-line flags are defined here.
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-a', '--algo', type=str, default='CMAES',
    #     help='Algorithm to be used for training'
    # )
    # parser.add_argument(
    #     '-p', '--policy', type=str, default='GaussianControlPolicy',
    #     help='Policy used for the agent'
    # )

    # args = parser.parse_args(rospy.myargv()[1:])    # moveit_commander.roscpp_initialize(sys.argv)
    # policy = args.policy
    # algo = args.algo
    algo = 'CMAES'
    all_traj, object_ids = load_data()
    discretization = 200
    object_reset_poses_demo = {key: [] for key in object_ids[1:]}
    for i in object_ids[1:]:
        object_reset_poses_demo[i] = np.asarray(all_traj[i])[0, 2:9]
    object_reset_poses_test = {key: [] for key in object_ids[1:]}

    object_reset_poses_test['4'] = np.hstack([np.random.uniform(0.6, 0.85), np.random.uniform(-0.4, -0.3), \
                np.asarray(all_traj['4'])[0,4:9]])

    object_reset_poses_test['5'] = np.hstack([np.random.uniform(0.6, 0.85), np.random.uniform(0.1,0.2), \
                np.asarray(all_traj['5'])[0,4:9]])

                            # np.random.multivariate_normal(np.zeros(7), np.diag([0.05,0.05,0.0,0,0,0,0]))
        # object_reset_poses_test[i] = np.asarray(all_traj[i])[0, 2:9]
    H, trajectory_scaling = get_transformation_params(all_traj, object_reset_poses_test, \
            object_reset_poses_demo, object_ids)
    parametrized_policy = np.asarray(all_traj['2'])[:, [2,3,4, -1]]
    parametrized_policy_original = copy(parametrized_policy)
    manipulated_object_start_pose = object_reset_poses_test[object_ids[-2]]
    parametrized_policy[:, :3] -= object_reset_poses_demo[object_ids[-2]][:3]
    parametrized_policy[:, :3] = transform_trajectory(parametrized_policy[:, :3], H) * trajectory_scaling
    parametrized_policy[:, :3] += object_reset_poses_test[object_ids[-2]][:3]
    args.mode = 'GUI'
    env = BulletEnv()
    env.init(args)
    env.initialize_world(all_traj, object_ids, object_reset_poses_demo)
    # print(object_reset_poses_test['4'][:3])
    # print(object_reset_poses_demo['4'][:3])
    # print(H)
    # print(trajectory_scaling)
    print('==== Policy prior set up')

    ##############################

    #### initialize policy ####
    policy = GaussianControlPolicy(
        env_spec=env.spec,
        init_mean=parametrized_policy.flatten())

    #### Initialize algorithm ####
    if algo == 'CMAES':
        # CMA ES
        algo = CMAES(
            env=env,
            policy=policy,
            max_path_length=50,
            n_itr=10,
            sigma0=0.005,
        )

    ###########################
    print('==== Training algorithm created')
    algo = RandomSampler(env=env, init_policy=parametrized_policy.flatten())
    algo = FiniteDifferenceLearner(env=env, init_policy=parametrized_policy_original.flatten())
    print("number of policy parameters: ", len(parametrized_policy.flatten()))

    # run original demo
    # algo.rollout()

    # run random initial config
    env.initialize_world(all_traj, object_ids, object_reset_poses_test)
    algo.init_policy = parametrized_policy.flatten()
    algo.rollout()
    p.resetSimulation()
    p.disconnect()

    # run learning algo headless
    args.mode = 'DIRECT'
    env.init(args)
    env.initialize_world(all_traj, object_ids, object_reset_poses_test)
    algo.train(100)


def load_data():
        demo_name = 'putAInFrontOfB_cubeenv2_1.bin.txt'
        # Robot ID is 2,  the object IDs are given as 4 and 5... (A and B)
        with open(join(DEMO_PATH, demo_name), 'r') as f:
            lines = f.readlines()
        #read joint names specified in file
        IDs = lines[0].rstrip().split(' ')
        all_traj = {key: [] for key in IDs}
        for idx, values in enumerate(lines[1:]):
            if idx < 50:
                continue
            if idx > 300:
                disc = DISCRETIZATION
            else:
                disc = 50
            #clean each line of file
            ID, values = clean_line(values)
            # only read lines after sufficient time to get less dense waypoints
            if ID in IDs and idx % disc == 0:
                all_traj[str(int(ID))].append(values)
        return all_traj, IDs

def get_transformation_params(all_traj, object_reset_poses_test, object_reset_poses_demo, object_ids):

         # get transformation between object pair
    static_object_test = object_reset_poses_test[object_ids[-1]]
    manipulated_object_test = object_reset_poses_test[object_ids[-2]]
    static_object_demo = object_reset_poses_demo[object_ids[-1]]
    manipulated_object_demo = object_reset_poses_demo[object_ids[-2]]
    t =  manipulated_object_test[:3] - manipulated_object_demo[:3]
    static_object_demo_trans = static_object_demo[:3] + t
    distance_between_demo = -manipulated_object_demo[:3] + static_object_demo[:3]
    distance_between_test = -manipulated_object_test[:3] + static_object_test[:3]
    R_rel = get_rotation_between_vectors(distance_between_demo*np.array([1,1,0.0]), \
                distance_between_test*np.array([1,1,0.0]))
    H = get_H(R_rel, np.array([0,0, 0.0]))
    trajectory_scaling = np.abs(distance_between_test) / np.abs(distance_between_demo)
    return H, trajectory_scaling


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VR Demo and Play")
    parser.add_argument('--log', type=str, default='demo', help='The path to the log file for demo recording.')
    parser.add_argument('--init', type=str, default='demos/putABehindB_cubeenv2_4.bin', help='The path to the log file to be initialized with.')
    parser.add_argument('--env', type=str, default='cubeenv2', help='The selected environment')
    parser.add_argument('--mode', type=str, default='DIRECT', help='The selected mode, GUI or DIRECT')

    args = parser.parse_args()
    main(args)
