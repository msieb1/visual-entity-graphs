import pybullet as p
import time
import argparse
import math
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from os.path import join
import cv2
import tensorflow as tf 
import imageio
import torch
from PIL import Image
from pdb import set_trace
import json
import pybullet_data
sys.path.append('./python')
#from gps.agent.bullet.SimEnv import simEnv
sys.path.append('/home/max/projects/gps-stick')
from config_global import Config as Config, \
                    Inference_Camera_Config as Camera_Config, \
                    Demo_Config as Demo_Config, \
                    Trajectory_Config
conf = Config()
cam_conf = Camera_Config()
dconf = Demo_Config()
trjconf = Trajectory_Config()

#from tcn import define_model_depth as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
#from tcn import define_model as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
#from mftcn import define_model
from gps.agent.bullet.bullet_env import Agent, SimEnv
# from env.kuka_iiwa import kuka_iiwa
#p.connect(p.UDP,"192.168.86.100")
plt.ion()
sys.path.append('../')
#from experiments.cube_and_bowl_mrcnn.configs import Inference_Camera_Config
#camconf = Inference_Camera_Config()

TARGET_POSITION = [1.32, -0.33, 0.46]
ATTACH_DIST = 0.12
SAMPLE_RATE = 20
IMG_SIZE = 240
FOV = 30
NEARVAL = 1
FARVAL = 3
CAMERA_DISTANCE = 1.5
YAW = -91
PITCH = -48

DEMOS_DIR = '/home/msieb/projects/lang2sim/demos'
SAVE_ROOT_DIR = '/home/msieb/projects/gps-lfd/demo_data'
# Config
EXP_DIR = conf.EXP_DIR
NUM_VIEWS = 1
USE_CUDA = conf.USE_CUDA
MODE = conf.MODE
AUDIO_OFFSET = 0.6
N_FRAMES = conf.N_PREV_FRAMES + 1
HOME_PATH = conf.HOME_PATH
IMAGE_SIZE = conf.IMAGE_SIZE_RESIZED
ACTION_DIM = dconf.ACTION_DIM

# Demo Config
EXP_NAME = dconf.EXP_NAME
DEMO_NAME = dconf.DEMO_NAME
DEMO_PATH = join(EXP_DIR, 'demonstrations', DEMO_NAME)
EMBEDDING_DIM = dconf.EMBEDDING_DIM
COMPUTE_TCN = dconf.COMPUTE_TCN
IMG_H = dconf.IMG_H
IMG_W = dconf.IMG_W
OBJECT_TYPE = dconf.OBJECT_TYPE
FPS = dconf.FPS
T = dconf.T
SELECTED_VIEW = dconf.SELECTED_VIEW

# Camera Config
VIEW_PARAMS = cam_conf.VIEW_PARAMS
PROJ_PARAMS = cam_conf.PROJ_PARAMS

# Demo Config
TRAJECTORY = trjconf.ROTATE
#####




def main(args):
    save_path = join(SAVE_ROOT_DIR, args.demo)
    visual = False
    rgb_folder, depth_folder, seg_folder, info_folder, base_folder, vid_folder = create_data_folders(visual, save_path)

    print("saving to ", save_path)
    demo_num = args.num


    files = [f for f in os.listdir(rgb_folder) if not f.endswith('.npy')]
    """Creates and returns one view directory per webcam."""
    # Create and append a sequence name.
    if args.num:
        demo_num = args.num
    else:
        # If there's no video directory, this is the first sequence.
        if not files:
          demo_num = '0'
        else:
          # Otherwise, get the latest sequence name and increment it.
          seq_names = [i.split('_')[0] for i in files if not i.endswith('.mp4')]
          latest_seq = sorted(map(int, seq_names), reverse=True)[0]
          demo_num = str(latest_seq+1)

    print("storing under demo number: ", demo_num)


    agent = Agent()
    rgb_writer = imageio.get_writer('{}/{}.mp4'.format(
                    rgb_folder, demo_num))
    crop_writer = imageio.get_writer('{}/{}_cropped.mp4'.format(
                    vid_folder, demo_num))
    #orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    ad = (np.random.rand(4)-0.5)*0.3 * 0
    orn = [0.70603903128+ad[0], 0.708148792076+ad[1], 0+ad[2], 0+ad[3]]
    ad = (np.random.rand(4)-0.5)*0.5

    #orn = p.getQuaternionFromEuler([-math.pi/2,0, math.pi/2])
    pos = [1.0, -0.400000, 0.9]
    agent.simEnv.reset(pos + orn)
    # agent.simEnv.reset()
    start_time = time.time()

    object_centroid_trajectory = np.zeros((T, 7))
    poses = np.zeros((T, 4))
    rots_pred = np.zeros((T, 4))

    end_effector_trajectory = np.zeros((T, 7))

    excluded_ids = ['kuka', 'pr2_gripper','pr2_cid', 'pr2_cid2']
    relevant_ids = {val: key for key, val in agent.h.items() if key not in excluded_ids}
    # relevant_ids = {h.cubeid: 'cube1', h.bowlid: 'bowl'}
    relevant_ids_names = {val: str(key) for key, val in relevant_ids.items()}
    object_trajectories = {key: [] for key in relevant_ids.keys()}
    init_object_poses = {}

    for key, val in relevant_ids.items():
        init_object_poses[key] = getPose(key)

    t = 0
    while(t < T):

        curr_time = time.time()

        dx = TRAJECTORY[t, 0]
        dy = TRAJECTORY[t, 1]
        dz =TRAJECTORY[t, 2]
        da = TRAJECTORY[t, 3]
        # dphi = TRAJECTORY[t, 4]
        # dtheta = TRAJECTORY[t, 5]

        if not t == 0: # Skip first time step because apparently GPS sets t=0 as no action 
          # agent.step_taskspace(np.asarray([dx,dy,dz,da, dphi, dtheta]))
          agent.step_taskspace_trans(np.asarray([dx, dy, dz, da]))
        if t >= T - 18:
          agent.step_taskspace_trans(np.asarray([dx, dy, dz, da]), stay=True)

        print(np.asarray([dx, dy, dz, da]))

        # if t >= T - 10:
        #     link_state = p.getLinkState(agent.o.kukaobject.kukaId, agent.o.kukaobject.kukaEndEffectorIndex)
        #     curr_eepos = link_state[0]
        #     curr_orn = link_state[1]
        #     pos = [0,0,0]
        #     orn = [0.70603903128, 0.708148792076, 0, 0]
        #     pos[0] = curr_eepos[0]
        #     pos[1] = curr_eepos[1]
        #     pos[2] = 1.3
        #     #orn = p.getQuaternionFromEuler([-math.pi/2,0, math.pi/2])
        #     agent.o.kukaobject.moveKukaEndtoPos(pos  , orn)
        # # if t % 1000 == 0:
        # color_view_buffer = []
        # depth_view_buffer = []
        # if t >= T - 12:
        #     agent.o.kukaobject.open_gripper()

        viewMat = p.computeViewMatrixFromYawPitchRoll(getPose(agent.h.cubeid)[:3] ,   #TARGET_POSITION, #getPose(objects[-1])[:3],
                                                      distance=CAMERA_DISTANCE,
                                                      yaw=YAW,
                                                      pitch=PITCH, 
                                                      roll=0,
                                                      upAxisIndex=2)
        projMat = p.computeProjectionMatrixFOV(FOV, aspect=1, nearVal=NEARVAL, farVal=FARVAL)
        rgb_crop, depth_crop = agent.get_images(0)
        resized_image = resize_frame(rgb_crop, IMAGE_SIZE)[None, :]
        resized_depth = resize_frame(depth_crop, IMAGE_SIZE)[None, :]

        crop_writer.append_data(rgb_crop)

        results = p.getCameraImage(width=IMG_SIZE,
                                   height=IMG_SIZE,
                                   viewMatrix=viewMat,
                                   projectionMatrix=projMat)
        # Note: we can compute flow online
        imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
            rgb_folder, demo_num, t), rgb_crop)

        imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
            depth_folder, demo_num, t), depth_crop)
        imageio.imwrite('{0}/{1}_{2:05d}.png'.format(
            seg_folder, demo_num, t), results[4])
        np.save('{0}/{1}_{2:05d}_mask.npy'.format(
            seg_folder, demo_num, t), results[4])

        object_centroid_trajectory[t, :] = np.array(agent.getObjectPose(agent.h.cubeid))
        link_state = p.getLinkState(agent.o.kukaobject.kukaId, agent.o.kukaobject.kukaEndEffectorIndex)
        end_effector_trajectory[t, :] = np.array(link_state[0] + link_state[1])

        rgb_writer.append_data(results[2])
        t += 1
        run_time = time.time() - curr_time
        #print("runtime: {}".format(run_time))
        # time.sleep(max(1.0/FPS - run_time, 0.0))
        for key in relevant_ids.keys():
          object_trajectories[key].append(getPose(key))   
        current = time.time()
        if t % 5 == 0:

            # tf.logging.info('Collected %s of video, %d frames at ~%.2f fps.' % (
            #     timer(start_time, current), frame_count, frame_count/(current-start_time)))
            print('Collected %s of video, %d frames at ~%.2f fps, time since start: %f' % (
              timer(start_time, current), t, t/(current-start_time), current-start_time))
    rgb_writer.close()
    crop_writer.close()
    print("saving data into {}".format(base_folder))

    np.save('{0}/{1}.npy'.format(
            vid_folder, demo_num,), object_trajectories[agent.h.cubeid])
    np.save('{}/{}_ee.npy'.format(
        sensor_folder, demo_num), np.array(end_effector_trajectory))
    with open('{}/{}_objects.json'.format(sensor_folder, demo_num), 'w') as f:
        json.dump(object_trajectories, f)

    with open('{}/{}_init_object_poses.json'.format(sensor_folder, demo_num), 'w') as f:
        json.dump(init_object_poses, f)
    camera_params = {"cameraTargetPosition": TARGET_POSITION, "distance": CAMERA_DISTANCE, "yaw": YAW, "pitch":   PITCH, 
    "roll": 0.0, "upAxisIndex": 2, "nearPlane":  NEARVAL, "farPlane":  FARVAL, "fov":  FOV,  "aspect":  1}
    with open('{}/{}_camera_params.json'.format(sensor_folder, demo_num), 'w') as f:
        json.dump(camera_params, f)
    with open('{}/{}_init_object_poses.json'.format(sensor_folder, demo_num), 'w') as f:
        json.dump(init_object_poses, f)
    with open('{}/{}_relevant_ids_names.json'.format(sensor_folder, demo_num), 'w') as f:
        json.dump(relevant_ids_names, f)

######## UTILS ##############3


# Note depth is in OpenGL format. It is non-linear. [0, 1]
# Nonlinearity helps to preserve more details when the object is close.
# Checkout https://learnopengl.com/Advanced-OpenGL/Depth-testing
def tinyDepth_no_rescale(depthSample, near, far):
    zLinear = far * near / (far - (far - near) * depthSample) 
    return np.asarray(zLinear)

def linDepth(depthSample, zNear, zFar):
    depthSample = 2.0 * depthSample - 1.0
    zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear))
    return zLinear

def nonLinearDepth(depthSample, zNear, zFar):
    linearDepth = linDepth(depthSample, zNear, zFar)
    nonLinearDepth = (zFar + zNear - 2.0 * zNear * zFar / linearDepth) / (zFar - zNear)
    nonLinearDepth = (nonLinearDepth + 1.0) / 2.0
    return nonLinearDepth

def create_data_folders(visual ,save_path):
    rgb_folder = 'rgb'
    depth_folder = 'depth'
    seg_folder = 'masks'
    flow_folder = 'flow'
    info_folder = 'info'
    vid_folder = 'videos'
    sensor_folder = 'sensor'
    rgb_folder = join(save_path, rgb_folder)
    depth_folder = join(save_path, depth_folder)
    seg_folder = join(save_path, seg_folder)
    flow_folder = join(save_path, flow_folder)
    info_folder = join(save_path, info_folder)
    vid_folder = join(save_path, vid_folder)
    sensor_folder = join(save_path, sensor_folder)

    base_folder = save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if visual is True:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(flow_folder):
            os.makedirs(flow_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        if not os.path.isdir(vid_folder):
            os.makedirs(vid_folder)
        if not os.path.isdir(sensor_folder):
            os.makedirs(sensor_folder)
        return rgb_folder, depth_folder, seg_folder, flow_folder, info_folder, base_folder, vid_folder, sensor_folder
    else:
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)
        if not os.path.isdir(depth_folder):
            os.makedirs(depth_folder)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)
        if not os.path.isdir(info_folder):
            os.makedirs(info_folder)
        if not os.path.isdir(vid_folder):
            os.makedirs(vid_folder)
        if not os.path.isdir(sensor_folder):
            os.makedirs(sensor_folder)
        return rgb_folder, depth_folder, seg_folder, info_folder, base_folder, vid_folder, sensor_folder


def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

def load_tcn_model(model_path, use_cuda=False):
    tcn = define_model(use_cuda, ACTION_DIM)
    tcn = torch.nn.DataParallel(tcn, device_ids=range(1))

    # tcn = PosNet()

    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    tcn.load_state_dict(state_dict)

    if use_cuda:
        tcn = tcn.cuda()
    return tcn

def load_tcn_weights(model_path):
    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    tcn.load_state_dict(state_dict)

def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)


def getPose(objID):
    pos, orn = p.getBasePositionAndOrientation(objID)
    return list(pos + orn)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', type=str, default='tmp', help='demo name to save')
  parser.add_argument('--num', type=int, default=None, help='demo name to save')

  args = parser.parse_args()

  main(args)
