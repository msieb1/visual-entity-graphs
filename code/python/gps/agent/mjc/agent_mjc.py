""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, NOISE

from gps.sample.sample import Sample



class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        if not isinstance(filename, list):
            world = mjcpy.MJCWorld(filename)
            self._world = [world for _ in range(self._hyperparams['conditions'])]
            self._model = [self._world[i].get_model().copy()
                           for i in range(self._hyperparams['conditions'])]
        else:
            for i in range(self._hyperparams['conditions']):
                self._world.append(mjcpy.MJCWorld(self._hyperparams['filename'][i]))
                self._model.append(self._world[i].get_model())

        #mj_name2id(self._model, mjOBJ_BODY, "test")

        # for i in range(self._hyperparams['conditions']):
        #     for j in range(len(self._hyperparams['pos_body_idx'][i])):
        #         idx = self._hyperparams['pos_body_idx'][i][j]
        #         self._model[i]['body_pos'][idx, :] += \
        #                 self._hyperparams['pos_body_offset'][i][j]
        #
        #         # Influences orientation of the body, indexed with 10 --> should be deleted in future (here: just to try it out!)
        #         self._model[i]['body_quat'][10, :] = np.array([1,-1,0,0])



        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        # Initialize x0.
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
                # initialize condition
                self._init(i)
                eepts = self._world[i].get_data()['site_xpos'].flatten() # eepts=end-effector-points= position + position of sites!
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
            elif END_EFFECTOR_POINTS_NO_TARGET in self.x_data_types:
                self._init(i)
                eepts = self._world[i].get_data()['site_xpos'].flatten()
                eepts_notgt = np.delete(eepts, self._hyperparams['target_idx'])
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts_notgt, np.zeros_like(eepts_notgt)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])
            if IMAGE_FEAT in self.x_data_types:
                self.x0[i] = np.concatenate([self.x0[i], np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],))])

        cam_pos = self._hyperparams['camera_pos']
        for i in range(self._hyperparams['conditions']):
            self._world[i].init_viewer(AGENT_MUJOCO['image_width'],
                                       AGENT_MUJOCO['image_height'],
                                       cam_pos[0], cam_pos[1], cam_pos[2],
                                       cam_pos[3], cam_pos[4], cam_pos[5])

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        feature_fn = None
        if 'get_features' in dir(policy):
            feature_fn = policy.get_features



        new_sample = self._init_sample(condition, feature_fn=feature_fn)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        # Take the sample.
        for t in range(self.T): # 100 steps
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :]) # get actions from policy
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition, feature_fn=feature_fn)
        new_sample.set(ACTION, U)
        new_sample.set(NOISE, noise)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    # Initialize condition!
    def _init(self, condition):
        """
        Set the world to a given model, and run kinematics.
        Args:
            condition: Which condition to initialize.
        """

        # Initialize world/run kinematics
        self._world[condition].set_model(self._model[condition])

        x0 = self._hyperparams['x0'][condition]
        idx = len(x0) // 2
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        self._world[condition].set_data(data)
        self._world[condition].kinematics()

    def _init_sample(self, condition, feature_fn=None):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
            feature_fn: function to compute image features from the observation.
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        self._init(condition)

        # Initialize sample with stuff from _data
        data = self._world[condition].get_data()

        sample.set(JOINT_ANGLES, data['qpos'].flatten(), t=0)
        sample.set(JOINT_VELOCITIES, data['qvel'].flatten(), t=0)
        eepts = data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)

        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)

        jac = np.zeros([eepts.shape[0], self._model[condition]['nq']]) #### important: why is here the wrong_number?????????  (added -1 so that it works, but that is not a longtime solution!)
        for site in range(eepts.shape[0] // 3):
            idx = site * 3

            nnn = self._world[condition].get_jac_site(site)

            jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)

        # save initial image to meta data
        self._world[condition].plot(self._hyperparams['x0'][condition])
        # update world --> plot
        img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                      self._hyperparams['image_height'])
        # mjcpy image shape is [height, width, channels],
        # dim-shuffle it for later conv-net processing,
        # and flatten for storage
        img_data = np.transpose(img["img"], (2, 1, 0)).flatten()
        # if initial image is an observation, replicate it for each time step
        if CONTEXT_IMAGE in self.obs_data_types:
            sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        else:
            sample.set(CONTEXT_IMAGE, img_data, t=None)
        sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                self._hyperparams['image_width'],
                                                self._hyperparams['image_height']]), t=None)
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            sample.set(RGB_IMAGE, img_data, t=0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
            if IMAGE_FEAT in self.obs_data_types:
                raise ValueError('Image features should not be in observation, just state')
            if feature_fn is not None:
                obs = sample.get_obs()  # Assumes that the rest of the sample has been populated
                sample.set(IMAGE_FEAT, feature_fn(obs), t=0)
            else:
                # TODO - need better solution than setting this to 0.
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=0)
        return sample



    def _set_sample(self, sample, mj_X, t, condition, feature_fn=None):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample (joint_angle and joint_velocity)
            t: Time step to set for sample.
            condition: Which condition to set.
            feature_fn: function to compute image features from the observation.
        """

        ## Set the position or other properties of objects for one sample --> after the sample the model is reinitialized with the initial properties
        #self._model[condition] = self._world[condition].get_model().copy()
        #self._model[condition]['body_pos'][10, :] += np.array([-0.0003, 0, -0.0003])

        # m = self._world[condition].get_model().copy()
        # m['body_pos'][10, :] += np.array([-0.003, 0, -0.003])
        # data = {'body_pos': m['body_pos']}
        # self._world[condition].set_model(data)#self._model[condition])

        #data = self._world[condition].get_data()    # <-- position of object is in xpos
        #m = self._world[condition].get_model().copy()

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

        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)  # important: [0:21] is a hack just for now, because q_vel is not the same as g_pos!! (not a longtime solution!!)
        cur_eepts = self._data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, cur_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (cur_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)

        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(cur_eepts, self._hyperparams['target_idx']), t=t+1)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(eept_vels, self._hyperparams['target_idx']), t=t+1)

        jac = np.zeros([cur_eepts.shape[0], self._model[condition]['nq']]) # important: -1 is a hack just for now, because q_vel is not the same as g_pos!! (not a longtime solution!!)
        for site in range(cur_eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)
        if RGB_IMAGE in self.obs_data_types:
            img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                          self._hyperparams['image_height'])
            sample.set(RGB_IMAGE, np.transpose(img["img"], (2, 1, 0)).flatten(), t=t+1)
            if feature_fn is not None:
                obs = sample.get_obs()  # Assumes that the rest of the observation has been populated
                sample.set(IMAGE_FEAT, feature_fn(obs), t=t+1)
            else:
                # TODO - need better solution than setting this to 0.
                sample.set(IMAGE_FEAT, np.zeros((self._hyperparams['sensor_dims'][IMAGE_FEAT],)), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_width, image_height, image_channels))
        return img
