# README


This code is an implementation of our work *Graph-Structrued Visual Imitation*. The code includes experiments on a real Baxter robot, as well as an extra, but not fully extensive, Pybullet experiment to facilitate use in the simulator environment.

The policy learning is based on a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization. For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps). 

As for the object detection, we used code from [here](https://github.com/matterport/Mask_RCNN). This is reflected by using their inference API in our code. In principle, any instance segmentation algorithm can be used by making apropriate changes in the code. 

Code base for pixel-level point feature detection is included here under 'code/python/pixel-feature-learning-master'.

The code base is **a work in progress**. 

### 1. Setup
```
source code/python/pixel-feature-learning-master/config/setup_environment.sh
```
Installation of required python dependencies is assumed. We use ROS for our Baxter experiments. We also provide a Pybullet interface.

### 2. Running experiments

Experiments are run by creating a corresponding folder in 'code/experiments' with three files: hyperparams.py, config.py, and an agent.py file. One can use existing experiment files as a template to create one's own experiment.

Assuming that the current working directory is the 'code' folder, one can then run the pipeline in the following way:

```
python python/gps/gps_main.py -experiment pouring_with_grasp_vel_control_lacan
```

Paths and weight files need to be appropriately specified in the config and hyperparams file. Policy learning parameters are also declared in these files, e.g., episode length, number of iterations per update cycle, and dimension of the state space. For questions referring to the policy learning itself, refer to [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps), on which we have extended our policy learning framework.

The agent.py file declares the required interface of the used environment, e.g., the Baxter robot, PyBullet, Mujoco, or any other platform.

We have provided an experiment without pixel feature level learning for pybullet that can be run via

```
python python/gps/gps_main.py -experiment cube_and_bowl_mrcnn
```
