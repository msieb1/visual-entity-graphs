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
#import tensorflow as tf 
import imageio
import torch
from PIL import Image

import pybullet_data
sys.path.append('./python')
#from gps.agent.bullet.SimEnv import SimEnv
from gps.utility.data_logger import VideoLogger

from pdb import set_trace
sys.path.append('/home/msieb/projects/gps-lfd')
from config import Training_Config, Multi_Camera_Config as Camera_Config
from gps.agent.bullet.bullet_env import Agent, SimEnv
sys.path.append(join('../', 'general-utils'))
from rot_utils import eulerAnglesToRotationMatrix, geodesic_dist_quat
from pyquaternion import Quaternion
# CONFIG
tconf = Training_Config()
EXP_NAME = tconf.EXP_NAME
HOME_PATH = tconf.HOME_PATH
#EXP_DIR = '/media/msieb/data/tcn_data/experiments'
EXP_DIR = tconf.EXP_DIR
MODE = tconf.MODE
OBJECT_TYPE = tconf.OBJECT_TYPE
FPS = tconf.FPS
SELECTED_VIEW = None # Collect data with all views!
IMG_H = tconf.IMG_H
IMG_W = tconf.IMG_W
IMAGE_SIZE = tconf.IMAGE_SIZE_RESIZED
CONTROL_TYPE = tconf.CONTROL_TYPE
T = tconf.T
EMBEDDING_DIM = tconf.EMBEDDING_DIM

# Camera Config
cam_conf = Camera_Config()
VIEW_PARAMS = cam_conf.VIEW_PARAMS
PROJ_PARAMS = cam_conf.PROJ_PARAMS
ROT_MATRICES = cam_conf.ROT_MATRICES

NUM_VIEWS = len(VIEW_PARAMS)
print("num viws: ", NUM_VIEWS)

##########

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', dest='dataset', type=str, default=EXP_NAME, help='Name of the dataset we`re collecting.')
  parser.add_argument('--mode', dest='mode', type=str, default=MODE, 
                         help='What type of data we`re collecting. E.g.: \
                         `train`,`valid`,`test`, or `demo`')
  parser.add_argument('--seqname', dest='seqname', type=str, default='',
                        help= 'Name of this sequence. If empty, the script will use, \
                         the name seq_N+1 where seq_N is the latest \
                         integer-named sequence in the videos directory.')
  parser.add_argument('--num_views', dest='num_views', type=int, default=NUM_VIEWS,
                          help='Number of webcams.')
  parser.add_argument('--expdir', dest='expdir', type=str, default=EXP_DIR,
                         help='dir to write experimental data to.')
  parser.add_argument('--tmp_imagedir', dest='tmp_imagedir' , type=str, default='/tmp/tcn/data',
                         help='Temporary outdir to write images.')
  parser.add_argument('--viddir', dest='viddir', type=str, default='videos',
                        help= 'Base directory to write videos.')
  parser.add_argument('--depthdir', dest='depthdir', type=str, default='depth',
                         help='Base directory to write depth.')
  parser.add_argument('--auddir', dest='auddir', type=str, default='audio',
                         help='Base directory to write audio.')
  parser.add_argument('--debug_vids', dest='debug_vids', type=bool, default= True,
                          help='Whether to generate debug vids with multiple \
                          concatenated views.')
  parser.add_argument('--debug_lhs_view',dest='debug_lhs_view', type=int, default= '1',
                        help= 'Which viewpoint to use for the lhs video.')
  parser.add_argument('--debug_rhs_view', dest='dest_rhs_view', type=int, default='2',
                         help='Which viewpoint to use for the rhs video.')
  parser.add_argument('--height', dest='height', type=int, default=1080, help='Raw input height.')
  parser.add_argument('--width', dest='width', type=int, default=1920, help='Raw input width.')
  return parser.parse_args()

FLAGS = parse_args()

# lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
# upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
# joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
# restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]




def main(args):

        module = importlib.import_module('bullet.simenv.' + args.env)
    envClass = getattr(module, 'UserEnv')
    env = envClass()
    cid = -1
    if demo_path is None:
        cid = p.connect(p.SHARED_MEMORY)

    if cid<0 and not no_connect:
        cid = p.connect(p.GUI)
    p.resetSimulation()
    #disable rendering during loading makes it much faster
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # Env is all loaded up here
    h, o = env.load()
    print('Total Number:', p.getNumBodies())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.setGravity(0.000000,0.000000,0.000000)
    p.setGravity(0,0,-10)

    ##show this for 10 seconds
    #now = time.time()
    #while (time.time() < now+10):
    # p.stepSimulation()
    p.setRealTimeSimulation(1)
    agent = Agent()
    # TODO: 
    # 1. Build logger for every view
    # 2. Collect demo run
    # 3. Train network
    # 4. Run PILQR with trained network and rewards
    n_samples = 60
    for itr in range(n_samples):
      view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth, audio_path = setup_paths_w_depth()

      # rgb_path = join(HOME_PATH, "projects/gps-lfd/experiments/ltcn_experiment/videos/1_view0.mp4")
      # depth_path = join(HOME_PATH, "projects/gps-lfd/experiments/ltcn_experiment/depth/1_view0.mp4")

      # rgb_logger = VideoLogger(rgb_path, 2.0)
      # depth_logger = VideoLogger(depth_path, 2.0)
      rgb_loggers = [imageio.get_writer(vidpath, fps=FPS) for vidpath in vid_paths]
      depth_loggers = [imageio.get_writer(depthpath, fps=FPS) for depthpath in depth_paths]
      object_centroid_trajectories = np.zeros((T, 7))
      end_effector_trajectory = np.zeros((T, 7))
      t = 0
      #orn = p.getQuaternionFromEuler([0, -math.pi, 0])
      ad = (np.random.rand(4)-0.5)*0.3
      orn = [0.70603903128+ad[0], 0.708148792076+ad[1], 0+ad[2], 0+ad[3]]
      
      #orn = p.getQuaternionFromEuler([-math.pi/2,0, math.pi/2])
      pos = [0.9, -0.100000, 0.85]
      #agent.simEnv.reset(pos + orn)
      agent.simEnv.reset()
      GLOBAL_IMAGE_BUFFER = []
      GLOBAL_DEPTH_BUFFER = []
      start_time = time.time()

      # ACTION PERTURBATUON PARAMETERS
      signs = (np.random.randint(2, size=6) - 0.5)*2
      weighting = np.random.rand(10)*3
      signs = (np.random.randint(2, size=10) - 0.5)*2
      


      while(t < T):

          curr_time = time.time()

          ######
          # EXECUTED ACTIONS FOR DEMONSTRATION

          # TASK SPACE
          dx = signs[0] * (np.random.rand(1)) * 0.0 
          dy = signs[1] * (np.random.rand(1)) * 0.0
          dz = signs[2] * (np.random.rand(1)) * 0.0 
          da = signs[3] * (np.random.rand(1)) * 0
          dphi = signs[4] * (np.random.rand(1)) * 2 * 0
          # dtheta =  signs[5]*(np.random.rand(1)) * 10
          dtheta =  signs[5]*40 # PERTURB EFFECTOR AXIS MORE TO GET MORE VARIETY IN MOTION

          # JOINT SPACE
          joint_actions = weighting*(np.random.rand(10)-0.5)*2

          ######
          if t > 0:
            if CONTROL_TYPE == 'task':
              agent.step_taskspace([dx,dy,dz,da, dphi, dtheta])
            else:
                #print(joint_actions)
                joint_actions[7] *= 5
                joint_pos = [p.getJointState(agent.o.kukaobject.kukaId, i)[0] for i in range(p.getNumJoints(agent.o.kukaobject.kukaId))]
                joint_actions[8] = 1.0
                joint_actions[9] = 1.0
                agent.step_jointspace(joint_actions)
                     
            # if t % 1000 == 0:
          color_view_buffer = []
          depth_view_buffer = []
          # gt = Quaternion(matrix=ROT_MATRICES[0]).elements
          # for rot in ROT_MATRICES:
          #   print(geodesic_dist_quat(Quaternion(matrix=rot).elements,  gt , tensor=False))
          for view in range(len(rgb_loggers)):
              if SELECTED_VIEW is None:
                  sel_view = view
              else:
                  sel_view = SELECTED_VIEW
              rgb_crop, depth_crop = agent.get_images(sel_view)

              color_view_buffer.append(rgb_crop)
              depth_view_buffer.append(depth_crop)
          object_centroid_trajectories[t, :] = np.array(agent.getObjectPose(agent.h.cubeid))
          link_state = p.getLinkState(agent.o.kukaobject.kukaId, agent.o.kukaobject.kukaEndEffectorIndex)
          end_effector_trajectory[t, :] = np.array(link_state[0] + link_state[1])

          GLOBAL_IMAGE_BUFFER.append(color_view_buffer)
          GLOBAL_DEPTH_BUFFER.append(depth_view_buffer)

          t += 1
          run_time = time.time() - curr_time
          #print("runtime: {}".format(run_time))
          time.sleep(max(1.0/FPS - run_time, 0.0))
      
          current = time.time()
          if t % 5 == 0:

              print('Collected %s of video, %d frames at ~%.2f fps, time since start: %f' % (
                timer(start_time, current), t, t/(current-start_time), current-start_time))
      for view in range(len(rgb_loggers)):
        for t in range(len(GLOBAL_IMAGE_BUFFER)):
            rgb_loggers[view].append_data(GLOBAL_IMAGE_BUFFER[t][view])
            depth_loggers[view].append_data(GLOBAL_DEPTH_BUFFER[t][view])
        rgb_loggers[view].close()
        depth_loggers[view].close()
      np_path = '/'.join(vid_paths[0].split('/')[:-1]) + '/' + seqname 
      print('saving np files to {}'.format(np_path))
      np.save(np_path + '_obj', object_centroid_trajectories)
      np.save(np_path + '_ee', end_effector_trajectory)
      np.save(np_path + '_camparams', np.array(ROT_MATRICES))



def list_add(a, b):
    return [aa + bb for aa, bb in zip(a, b)]

def setup_paths_w_depth():
  """Sets up the necessary paths to collect videos."""
  assert FLAGS.dataset
  assert FLAGS.mode
  assert FLAGS.num_views
  assert FLAGS.expdir
  assert FLAGS.auddir

  # Setup directory for final images used to create videos for this sequence.
  tmp_imagedir = os.path.join(FLAGS.tmp_imagedir, FLAGS.dataset, FLAGS.mode)
  if not os.path.exists(tmp_imagedir):
    os.makedirs(tmp_imagedir)
  tmp_depthdir = os.path.join(FLAGS.tmp_imagedir,  FLAGS.dataset, 'depth', FLAGS.mode)
  if not os.path.exists(tmp_depthdir):
    os.makedirs(tmp_depthdir)
  # Create a base directory to hold all sequence videos if it doesn't exist.
  vidbase = os.path.join(FLAGS.expdir, FLAGS.dataset, FLAGS.viddir, FLAGS.mode)

  if not os.path.exists(vidbase):
    os.makedirs(vidbase)

    # Setup depth directory

  depthbase = os.path.join(FLAGS.expdir, FLAGS.dataset, FLAGS.depthdir, FLAGS.mode)
  if not os.path.exists(depthbase):
    os.makedirs(depthbase)
  # Get one directory per concurrent view and a sequence name.
  view_dirs, seqname = get_view_dirs(vidbase, tmp_imagedir)
  view_dirs_depth = get_view_dirs_depth(vidbase, tmp_depthdir)

  # Setup audio directory
  audbase = os.path.join(FLAGS.expdir, FLAGS.dataset, FLAGS.auddir, FLAGS.mode)
  if not os.path.exists(audbase):
    os.makedirs(audbase)
  audio_path = os.path.join(audbase, seqname)

  # Get an output path to each view's video.
  vid_paths = []
  for idx, _ in enumerate(view_dirs):
    vid_path = os.path.join(vidbase, '%s_view%d.mp4' % (seqname, idx))
    vid_paths.append(vid_path)
  depth_paths = []
  for idx, _ in enumerate(view_dirs_depth):
    depth_path = os.path.join(depthbase, '%s_view%d.mp4' % (seqname, idx))
    depth_paths.append(depth_path)

  # Optionally build paths to debug_videos.
  debug_path = None
  if FLAGS.debug_vids:
    debug_base = os.path.join(FLAGS.expdir, FLAGS.dataset, '%s_debug' % FLAGS.viddir, 
                              FLAGS.mode)
    if not os.path.exists(debug_base):
      os.makedirs(debug_base)
    debug_path = '%s/%s.mp4' % (debug_base, seqname)
    debug_path_depth = '%s/%s_depth.mp4' % (debug_base, seqname)


  return view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth, audio_path

def get_view_dirs(vidbase, tmp_imagedir):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if FLAGS.seqname:
    seqname = FLAGS.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(vidbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(vidbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs = [os.path.join(
      tmp_imagedir, '%s_view%d' % (seqname, v)) for v in range(FLAGS.num_views)]
  for d in view_dirs:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs, seqname


def get_view_dirs_depth(depthbase, tmp_depthdir):
  """Creates and returns one view directory per webcam."""
  # Create and append a sequence name.
  if FLAGS.seqname:
    seqname = FLAGS.seqname
  else:
    # If there's no video directory, this is the first sequence.
    if not os.listdir(depthbase):
      seqname = '0'
    else:
      # Otherwise, get the latest sequence name and increment it.
      seq_names = [i.split('_')[0] for i in os.listdir(depthbase)]
      latest_seq = sorted(map(int, seq_names), reverse=True)[0]
      seqname = str(latest_seq+1)
    print('No seqname specified, using: %s' % seqname)
  view_dirs_depth = [os.path.join(
      tmp_depthdir, '%s_view%d' % (seqname, v)) for v in range(FLAGS.num_views)]
  for d in view_dirs_depth:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs_depth

def linearDepth(depthSample, zNear=None, zFar=None):
    zNear = 0.01
    zFar = 100
    depthSample = 2.0 * depthSample - 1.0
    zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear))
    clipping_distance = 2.0
    zLinear[np.where(zLinear > clipping_distance)] = 0
    depth_rescaled = ((zLinear  - 0) / (clipping_distance - 0)) * (255 - 0) + 0

    return np.asarray(depth_rescaled, dtype=np.uint8)

def timer(start, end):
  """Returns a formatted time elapsed."""
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)


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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
