# pixel-feature-learning
This repository contains code for learning pixel feature via either of the following two data settings:
- images taken from multiple viewpoints in simulation (multi-view setting)
- images augmented from realistic food images via affine transformation, pixel intensity change, color shift, etc.. (image augmentation setting)

It is adapted from the Dense Object Net code (https://github.com/RobotLocomotion/pytorch-dense-correspondence). It learns pixel embedding via learning pixel-level correspondence.

Note: to run the code you need to clone https://github.com/warmspringwinds/pytorch-segmentation-detection.git as a subfolder in the root directory.

## 1. Setup
Under current (```pixel-feature-learning```) directory:
```
mkdir -p pdc/trained_models
sudo apt install --no-install-recommends wget libglib2.0-dev libqt4-dev libqt4-opengl libx11-dev libxext-dev libxt-dev mesa-utils libglu1-mesa-dev python-dev python-lxml python-numpy python-scipy python-yaml libyaml-dev python-tk ffmpeg

# make virtualenv
cd ~/virtualenvs
virtualenv --system-site-packages pci_venv 
source ~/virtualenvs/pci_venv/bin/activate

# Edit config/setup_environment.sh if needed
source config/setup_environment.sh

pip install matplotlib testresources pybullet==2.4.0 visdom requests scipy imageio scikit-image==0.14.2 tensorboard sklearn opencv-contrib-python tensorboard_logger tensorflow
pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl torchvision torchnet 
pip install jupyter opencv-python plyfile pandas six Shapely imgaug
```

## 2. File structure
- ```config```: this folder contains all the configuration you would need to specify for training a model.
- ```dense_correspondence```: main directory for training code.
- ```modules```: here are some utility functions required for training the model.
- ```pdc``` and ```data```: this folders are where we put data and store models.
- ```nearest-neighbour-visualization```: code for visualizing nearest neighbour pixel.

## 3. Training using multi-view setting
### 3.1 Collect Data (don't activate virtualenv for this since we use ROS python)
Turn on baxter, then start ROS.
```
cd ~/ros_ws && . baxter.sh
```
All the following roslaunch commands each needs to run inside a shell window with ROS started. 
If you see errors when starting the camera, try reconnecting it manually.
IMPORTANT: Sometimes the camera image becomes reddish. Reconnect the camera to make it normal.
```
# launch camera
roslaunch realsense2_camera rs_rgbd.launch serial_no:=826212070528
# broadcast camera to robot transformation
roslaunch hand_eye_calibration hand_eye_tf_broadcaster.launch
# open image viewer to check whether the color is reddish/normal
rosrun image_view image_view image:=/camera/color/image_raw
```
Record trajectory.
```
# untuck robot
rosrun baxter_tools tuck_arms.py -u
# Record a trajectory and store it to a csv file. Grab the cuff of baxter and move around the camera. Once you are done press Ctrl-C to stop.
rosrun baxter_examples joint_recorder.py -f ~/git/pixel-feature-learning/data_generation/trajs/traj.csv
# Process recorded trajectory to add end-effector rotation. -s scales up the duration of the traj. -c controls the duration of one cycle of camera rotation.
python ~/git/pixel-feature-learning/data_generation/process_traj.py -f ~/git/pixel-feature-learning/data_generation/trajs/traj.csv -s 5 -c 10
```
Play back the processed trajectory and collect data. Specify the scene name.
Note: 
1. you need to change the ```limit_lower``` and ```limit_upper``` to produce correct masks. These two values specify a box region in which objects are placed. Pixels outside this 3D box are considered background and masked out.
2. The minimum depth range is 20cm for the D435 camera, so do not place the camera too close to the objects.
3. Depth cameras does not work very well for reflective surface.
4. You should generate data for a few different scenes with different object configuration.
5. In your experiment, you should replace all the 'example' which your own object names for clarity.
```
rosrun baxter_examples joint_position_file_playback.py -f ~/git/pixel-feature-learning/data_generation/trajs/traj_processed.csv
python ~/git/pixel-feature-learning/data_generation/collect_data.py -n /camera -t /camera -s example_1 -f 10
```

Don't forget to tuck and then turn off robot when you are done collecting data.
```
rosrun baxter_tools tuck_arms.py -t
```

### 3.2 Training
Edit/check these before training:
- get_default_K_matrix in ```dense_correspondence/correspondence_tools/correspondence_finder.py```
- Training config file in ```config/dense_correspondence/training/```. e.g. ```example_training.yaml```
- Dataset config file in ```config/dense_correspondence/dataset/```. e.g. ```composite/example.yaml``` and ```single_object/example.yaml```
- training script in ```dense_correspondence/training```. e.g. ```run_training_example.py```

Train:
```
source ~/virtualenvs/pci_venv/bin/activate
source ~/git/pixel-feature-learning/config/setup_environment.sh
python ~/git/pixel-feature-learning/dense_correspondence/training/run_training_example.py
```

### 3.3 Testing
Edit/check these before evaluation:
- ```config/dense_correspondence/evaluation/evaluation.yaml```
- ```config/dense_correspondence/heatmap_vis/heatmap.yaml```

Visualize heatmap:
```
source ~/virtualenvs/pci_venv/bin/activate
source ~/git/pixel-feature-learning/config/setup_environment.sh
python ~/git/pixel-feature-learning/modules/user-interaction-heatmap-visualization/live_heatmap_visualization.py
```

