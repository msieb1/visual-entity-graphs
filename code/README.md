GPS
======

This code is a reimplementation of the guided policy search algorithm and LQG-based trajectory optimization, meant to help others understand, reuse, and build upon existing work.

For full documentation, see [rll.berkeley.edu/gps](http://rll.berkeley.edu/gps).

The code base is **a work in progress**. See the [FAQ](http://rll.berkeley.edu/gps/faq.html) for information on planned future additions to the code.
# gps-lfd


### 1. Setup
```
cd ~/ros_ws && ./baxter.sh
source ~/virtualenvs/pci_venv/bin/activate
source ~/git/pixel-feature-learning/config/setup_environment.sh
```

### 2. Running the main file

```
python python/gps/gps_main.py -experiment giraffe_pushing_reward_computation
python python/gps/gps_main.py -experiment reach
python python/gps/gps_main.py -experiment reach_fingers_mean
```


### Remarks to do on 03/17/2019
- Tensorflow error with core dumped: 2019-03-17 10:38:21.741202: F ./tensorflow/core/framework/tensor.h:663] Check failed: new_num_elements == NumElements() (11 vs. 7)

in agent_baxter.py:
- Display End Effector mean on figure (self.fig.gca().scatter somehow doesnt work)
Look into get_mrcnn_features and how the figure is handled
- overlay demo images (can be accessed via self._hyperparams['demo_imgs'][t])
- make a video saver function or somehting
- investigate cost with ipdb in line 371
(just run python python/gps/gps_main.py -experiment stacking_yellowhexagon_on_purplering and it will break there )

- also you can change command velocity ('max_velocity' in agent dict)

Good luck!


### Parameters to look out for
- cost weight
- taskspace delta / max velocity
- dt (the smaller the better for velocity, higher better for position control)
- T (the higher the finer the demo is dissected, overall execution time for robot is dt*T)
-