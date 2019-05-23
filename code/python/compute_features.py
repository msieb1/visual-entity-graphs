import sys
import os
import argparse
from os.path import join

from collections import OrderedDict
from imageio import imwrite
import imageio
import numpy as np
from PIL import Image
import torch
from pdb import set_trace

sys.path.append('/home/max/projects/gps-lfd_git')
sys.path.append('/home/msieb/projects/gps-lfd_git')
from gps.agent.bullet.bullet_utils import get_view_embedding, resize_frame, load_tcn_model

from config import Config as Config, \
                    Camera_Config as Camera_Config, \
                    Demo_Config as Demo_Config, \
                    Trajectory_Config
conf = Config()
cam_conf = Camera_Config()
dconf = Demo_Config()


# Config
EXP_DIR = conf.EXP_DIR
NUM_VIEWS = conf.NUM_VIEWS
USE_CUDA = conf.USE_CUDA
MODE = conf.MODE
AUDIO_OFFSET = 0.6
N_FRAMES = conf.N_PREV_FRAMES + 1
HOME_PATH = conf.HOME_PATH
IMAGE_SIZE = conf.IMAGE_SIZE_RESIZED


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

# Camera Config
VIEW_PARAMS = cam_conf.VIEW_PARAMS
PROJ_PARAMS = cam_conf.PROJ_PARAMS

# Demo Config
TRAJECTORY = trjconf.TRAJECTORY_PICKUP
#####

get_embedding = get_view_embedding



def main(args):
    # output_folder = join(OUTPUT_PATH, args.experiment_relative_path)
    output_folder = OUTPUT_PATH
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)     
    
    tcn = load_tcn_model(MODEL_PATH, use_cuda=USE_CUDA)
    print('convert videos to features in {}'.format(INPUT_PATH))
    # input_folder = join(INPUT_PATH, args.experiment_relative_path)
    for file in [ p for p in os.listdir(RGB_PATH) if p.endswith('.mp4') ]:
        print("Processing ", file)
        reader = imageio.get_reader(join(RGB_PATH, file))
        reader_depth = imageio.get_reader(join(DEPTH_PATH, file))

        embeddings = np.zeros((len(reader), EMBEDDING_DIM))
        embeddings_normalized = np.zeros((len(reader), EMBEDDING_DIM))

        i = 0
        for im, im_depth in zip(reader, reader_depth):
            resized_image = resize_frame(im, IMAGE_SIZE)[None, :]
            resized_depth = resize_frame(im_depth, IMAGE_SIZE)[None, :]
            # resized_depth = resize_frame(depth_rescaled[:, :, None], IMAGE_SIZE)[None, :]
            frame = np.concatenate([resized_image[0], resized_depth[0, None, 0]], axis=0)            
            if USE_CUDA:
              output_normalized, output_unnormalized, _ = tcn(torch.Tensor(frame[None, :]).cuda())
            else:
              output_normalized, output_unnormalized, _ = tcn(torch.Tensor(frame[None, :])) 
            embeddings[i, :] = output_unnormalized.detach().cpu().numpy()
            embeddings_normalized[i, :] = output_normalized.detach().cpu().numpy()
            i += 1
        print("Saving to ", output_folder)
        save_np_file(folder_path=output_folder, name=(file.split('.')[0] + '_' + 'emb').replace('video_sample_emb', 'emb_visual_ff'), file=embeddings)
        save_np_file(folder_path=output_folder, name=(file.split('.')[0] + '_' + 'emb_norm').replace('video_sample_emb', 'emb_visual'),file=embeddings_normalized)
        print("=" * 10)
    print('Exit function')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)        
