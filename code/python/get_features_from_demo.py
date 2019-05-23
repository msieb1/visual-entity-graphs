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

import pybullet_data
sys.path.append('./python')
#from gps.agent.bullet.SimEnv import simEnv
from gps.utility.data_logger import VideoLogger
from gps.agent.bullet.bullet_utils import get_view_embedding
sys.path.append('/home/max/projects/gps-lfd')
from config import Config as Config, \
                    Camera_Config as Camera_Config, \
                    Demo_Config as Demo_Config, \
                    Trajectory_Config
conf = Config()
cam_conf = Camera_Config()
dconf = Demo_Config()
trjconf = Trajectory_Config()

sys.path.append(conf.TCN_PATH)
#from tcn import define_model_depth as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
#from tcn import define_model as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
#from mftcn import define_model
from view_tcn import define_model
from gps.agent.bullet.bullet_env import Agent, SimEnv

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
MODEL_NAME = dconf.MODEL_NAME
MODEL_FOLDER = dconf.MODEL_FOLDER
MODEL_PATH = join(EXP_DIR, EXP_NAME, 'trained_models',MODEL_FOLDER, MODEL_NAME)
MODEL_PATH = dconf.MODEL_PATH
# Camera Config
VIEW_PARAMS = cam_conf.VIEW_PARAMS
PROJ_PARAMS = cam_conf.PROJ_PARAMS

# Demo Config
TRAJECTORY = trjconf.RANDOM
SELECTED_SEQ = dconf.SELECTED_SEQ_FOR_FEATURE_COMPUTATION
#####

get_embedding = get_view_embedding


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', dest='dataset', type=str, default=EXP_NAME, help='Name of the dataset we`re collecting.')
  parser.add_argument('--mode', dest='mode', type=str, default=MODE, 
                         help='What type of data we`re collecting. E.g.: \
                         `train`,`valid`,`test`, or `demo`')
  parser.add_argument('-s', '--seqname', dest='seqname', type=str, default='',
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

  print("model path: {}".format(MODEL_PATH))
  return parser.parse_args()

FLAGS = parse_args()

def main(args):

    output_folder = os.path.join(DEMO_PATH)
    input_folder = os.path.join(DEMO_PATH)
    vid_file = os.path.join(input_folder, '%s_view%d.mp4' % (SELECTED_SEQ, 0))


    if not os.path.exists(output_folder):
        print("wrong path provided")   
    
    tcn = load_tcn_model(MODEL_PATH, use_cuda=USE_CUDA)


    view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth, audio_path = setup_paths_w_depth()

    # rgb_path = join(HOME_PATH, "projects/gps-lfd/experiments/ltcn_experiment/videos/1_view0.mp4")
    # depth_path = join(HOME_PATH, "projects/gps-lfd/experiments/ltcn_experiment/depth/1_view0.mp4")

    # rgb_logger = VideoLogger(rgb_path, 2.0)
    # depth_logger = VideoLogger(depth_path, 2.0)

    rgb_vids = imageio.get_reader(vid_file) 
    rgb_writer = imageio.get_writer(vid_paths[0], fps=FPS) 



    embeddings = np.zeros((T, EMBEDDING_DIM))
    embeddings_normalized = np.zeros((T, EMBEDDING_DIM))
    object_centroid_trajectory = np.load(join(input_folder, "{}_object_pose.npy".format(SELECTED_SEQ)))
    end_effector_trajectory = np.load(join(input_folder, "{}_end_effector_pose.npy".format(SELECTED_SEQ)))

    poses = np.zeros((T, 4))
    rots_pred = np.zeros((T, 4))

    end_effector_trajectory = np.zeros((T, 7))

    #reader = imageio.get_writer(join(output_folder, seqname + '_view0.mp4'), fps=10)
    t= 0
    for im in rgb_vids:

        curr_time = time.time()
        resized_image = resize_frame(im, IMAGE_SIZE)[None, :]
        emb_unnormalized, a_pred = get_view_embedding(tcn, resized_image, resized_image,use_cuda=USE_CUDA)
        embeddings[t, :] = emb_unnormalized
        rots_pred[t, :] = a_pred

        run_time = time.time() - curr_time
        rgb_writer.append_data(im)
        #print("runtime: {}".format(run_time))
        t += 1
    print("saving data into {}".format(output_folder))
    save_np_file(output_folder, "{}_tcn_features_raw.npy".format(seqname), embeddings)
    save_np_file(output_folder, "{}_object_pose.npy".format(seqname), object_centroid_trajectory)
    save_np_file(output_folder, "{}_end_effector_pose.npy".format(seqname), end_effector_trajectory)

######################  UTILS ######################

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
  vidbase = os.path.join(DEMO_PATH)


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
  # audbase = os.path.join(FLAGS.expdir, FLAGS.dataset, FLAGS.auddir, FLAGS.mode)
  # if not os.path.exists(audbase):
  #   os.makedirs(audbase)
  # audio_path = os.path.join(audbase, seqname)
  audbase = None
  audio_path = None
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

  print('Storing under sequence name {}'.format(seqname))
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
    tf.logging.info('No seqname specified, using: %s' % seqname)
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
    tf.logging.info('No seqname specified, using: %s' % seqname)
  view_dirs_depth = [os.path.join(
      tmp_depthdir, '%s_view%d' % (seqname, v)) for v in range(FLAGS.num_views)]
  for d in view_dirs_depth:
    if not os.path.exists(d):
      os.makedirs(d)
  return view_dirs_depth



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



if __name__ == '__main__':
    args = parse_args()

    main(args)
