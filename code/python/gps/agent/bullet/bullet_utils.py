
import numpy as np
import torch
from pdb import set_trace
from collections import OrderedDict
from imageio import imwrite
import imageio
from PIL import Image
import torch
import sys
import os
from os.path import join
import importlib
from ipdb import set_trace
# sys.path.append('/home/max/projects/gps-lfd')
# sys.path.append('/home/msieb/projects/gps-lfd')

#from view_tcn import define_model as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
sys.path.append('/home/msieb/projects/general-utils')
from plot_utils import plot_results

import pybullet as p
import math
import numpy as np
from os.path import join
from ipdb import set_trace as st
# Standard imports
import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import os

import matplotlib.pylab as pl


def array_to_pil(img):
    # transform 0 to 1 values to PIL
    if img.shape[2] == 4:
        img = img[:, :, :-1]
    return Image.fromarray(np.uint8(img * 255))

def rgb_to_pil(img):
  # Transform 0 to 255 to PIL
    if img.shape[2] == 4:
        img = img[:, :, :-1]
    return Image.fromarray(np.uint8(img))



def get_pose_embedding(tcn, resized_image, resized_depth, use_cuda=True):
  ### SINGLE FRAME
  frame = np.concatenate([resized_image[0], resized_depth[0, None, 0]], axis=0)
  if use_cuda:
    output_normalized, output_unnormalized, pose_output = tcn(torch.Tensor(frame[None, :]).cuda())
  else:
    output_normalized, output_unnormalized, pose_output = tcn(torch.Tensor(frame[None, :]))  

          #### INSERTED FOR MULTIFRAME TCN
          # resized_image = np.squeeze(resized_image)
          # image_buffer[view].append(resized_image)
          # frames = np.asarray(image_buffer[view][-N_FRAMES-1:])
          # if len(frames < N_FRAMES + 1):
          #   for i in range(N_FRAMES - len(frames)):
          #     frames = np.concatenate([frames, frames[-1][None]], axis=0)

          # frames = frames[None]
          # if USE_CUDA:
          #   output_normalized, output_unnormalized, pose_output = tcn(torch.Tensor(frames).cuda())
          # else:
            #output_normalized, output_unnormalized, pose_output = tcn(torch.Tensor(frames))  

def get_view_embedding(tcn, resized_image, resized_image_2=None, use_cuda=True):
  #### INSERTED FOR MULTICAMERA VIEW TCN
  if resized_image_2 is not None:
    frames = np.concatenate([resized_image, resized_image_2], axis=0)
    frames = frames[None]
  else:
    frames = resized_image[None]
  if use_cuda:
    _, rot_pred, output_unnormalized = tcn(torch.Tensor(frames).cuda())
  else:
    _, rot_pred, output_unnormalized = tcn(torch.Tensor(frames))
  ##########################
  emb_raw = np.squeeze(output_unnormalized.detach().cpu().numpy())
  rot =  np.squeeze(rot_pred.detach().cpu().numpy())
  return emb_raw / np.linalg.norm(emb_raw), rot



def resize_frame(frame, out_size):
    image = Image.fromarray(frame)
    image = image.resize(out_size)
    scaled = np.array(image, dtype=np.float32) / 255
    return np.transpose(scaled, [2, 0, 1])

def load_pixel_model(model_path, use_cuda=True):
    """
    Assumes weight_files folder is within the folder of the currently executed experiment.
    Also assumes that the name of the used model in pytorch is given by the first part of the '-'separated file name, i.e. tcn_2-epoch-10.pk
    would equal the model name tcn_2
    """
    basedir = '/home/msieb/projects/pixel-correspondence-imitation/src'
    # basedir = '/home/msieb/projects/pixel/src'

    sys.path.append(basedir)
    module = importlib.import_module('correspondence_finder')
    CorrespondenceFinder = getattr(module, 'CorrespondenceFinder')
    cf = CorrespondenceFinder()
    if use_cuda:
        cf = cf
    return cf

def load_tcn_model(model_path, use_cuda=True, basedir=None):
    """
    Assumes weight_files folder is within the folder of the currently executed experiment.
    Also assumes that the name of the used model in pytorch is given by the first part of the '-'separated file name, i.e. tcn_2-epoch-10.pk
    would equal the model name tcn_2
    """
    if basedir is None:
      basedir = os.path.join('/'.join(model_path.split('/')[:-5]), 'LTCN/')
    model_name = model_path.split('/')[-1].split('-')[0]
    sys.path.append(basedir)
    module = importlib.import_module('models.' + model_name)
    define_model = getattr(module, 'define_model')

    tcn = define_model(use_cuda)
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

def save_np_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    np.save(filepath, file)

def save_image_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    imwrite(filepath, file)

def list_add(a, b):
    return [aa + bb for aa, bb in zip(a, b)]

def linearDepth(depthSample, zNear=None, zFar=None):
    if zNear is None:
      zNear = 0.01
    if zFar is None:
      zFar = 100
    depthSample = 2.0 * depthSample - 1.0
    zLinear = 2.0 * zNear * zFar / (zFar + zNear - depthSample * (zFar - zNear))
    clipping_distance = zFar
    zLinear[np.where(zLinear > clipping_distance)] = 0
    depth_rescaled = ((zLinear  - 0) / (clipping_distance - 0)) * (255 - 0) + 0
    return np.asarray(depth_rescaled, dtype=np.uint8)

def tinyDepth(depthSample, near, far):
    zLinear = far * near / (far - (far - near) * depthSample) 
    clipping_distance = far
    zLinear[np.where(zLinear > clipping_distance)] = 0
    depth_rescaled = ((zLinear  - 0) / (clipping_distance - 0)) * (255 - 0) + 0
    return np.asarray(depth_rescaled, dtype=np.uint8)

def tinyDepth_no_rescale(depthSample, near, far):
    zLinear = far * near / (far - (far - near) * depthSample) 
    return np.asarray(zLinear, dtype=np.uint8)


def pixel_normalize(input, max_x, max_y, max_z):
  # Normalizes X and Y as well Depth to unit values, max_z is 255 if rescaled depth is used, otherwise clipping value
  if len(input.shape) < 2:
    input[:3] /= np.array([max_x, max_y, max_z])
  else:
    input[:, :3] /= np.array([max_x, max_y, max_z])
  return input


def unproject_2d_point(x, y, z, cx=120, cy=120, fx=447.846, fy=447.846):
  p3d_x = (x - cx)*z / fx
  p3d_y = (y - cy)*z / fy
  p3d_z = z
  return p3d_x, p3d_y, p3d_z



def plot_costs(logdir, itr, type='np_cost'):
  filenames = [file for file in os.listdir(logdir) if type in file]
  filenames = sorted(filenames, key=lambda x: int(x.split('.')[0].split('_')[-1]))
  mean = []
  std = []
  for ii, file in enumerate(filenames):
    cur_itr_cost = np.load(join(logdir, file))
    mean.append(np.mean(np.sum(cur_itr_cost, axis=1), axis=0))
    std.append(np.std(np.sum(cur_itr_cost, axis=1), axis=0))
  plot_results(np.array(mean), np.array(std), path=join(logdir, 'plots'), name='{}_plot_itr_{}'.format(type, itr), save_figure=True)

# plot_costs('/home/msieb/projects/gps-lfd/experiments/cube_and_bowl_blob/data_files/2018-10-11_19-21-15', itr=9, type='np_gt_cost')

def compute_tcn_embedding(model, image, image_resized_size=(299, 299), use_cuda=True):
    if image.dtype == 'float32':
      image = (255*image).astype(np.uint8)
    resized_image = resize_frame(image, image_resized_size)[None, :]
    if use_cuda:
      output_normalized, output_unnormalized, _ = model(torch.Tensor(resized_image).cuda())
    else:
      output_normalized, output_unnormalized, _ = model(torch.Tensor(resized_image)) 
    embeddings = output_unnormalized.detach().cpu().numpy()
    embeddings_normalized= output_normalized.detach().cpu().numpy()
    return embeddings_normalized / 10.0



def compute_view_pose_embedding(model, image, image_resized_size=(299, 299), use_cuda=True):
    if image.dtype == 'float32':
      image = (255*image).astype(np.uint8)
    resized_image = resize_frame(image, image_resized_size)[None, :]
    if use_cuda:
      embedding, pose  = model(torch.Tensor(resized_image).cuda())
    else:
      embedding, pose  = model(torch.Tensor(resized_image)) 
    embedding = embedding.detach().cpu().numpy()
    pose= pose.detach().cpu().numpy()
    return pose


def compute_pixel_query_points(model, query_mask, n_points_final, query_image=None):

    valid_query_ind = np.zeros(query_mask.shape)
    valid_query_ind[np.where(query_mask == 3)] = 1
    valid_query_ind = np.where(valid_query_ind)

    sample_inds = np.random.choice(len(valid_query_ind[0]), len(valid_query_ind[0]), replace=False)
    points = []
    for i in sample_inds:
        add_point = True
        curr_pt = (valid_query_ind[1][i], valid_query_ind[0][i]) 
        for pt in points:
            dist = np.linalg.norm(np.array(curr_pt) - np.array(pt))
            # if dist < 1:
            #     add_point = False
            #     continue
        if add_point:
            points.append(curr_pt)
        if len(points) >= n_points_final:
            break
    assert len(points) == n_points_final


    return points


def compute_pixel_target_points(model, query_points, valid_target_ind, target_image, use_cuda=True):
    model.feed_images(array_to_pil(img1), array_to_pil(target_image))
    matches  = []
    for i, pt in enumerate(query_points):
        
        img1_uv = pt
        img2_uv_match = model.find_best_match(img1_uv)

        if valid_target_ind[img2_uv_match] != 1:
            continue
        else:
            matches.append([img1_uv, img2_uv_match])

def compute_pixel_target_distances(model, query_points, query_image, valid_target_ind, target_image, n_max_correspondences, debug=False, use_cuda=True):
    # plt.imsave('/home/msieb/misc/for_zhouxian/0_00000.png', query_image)
    # plt.imsave('/home/msieb/misc/for_zhouxian/0_00001.png', target_image)
    st()
    model.feed_images_rgb(array_to_pil(query_image), rgb_to_pil(target_image), flags=[True, True])#, img_target_mask=valid_target_ind)
    # set_trace()
    # matches  = []
    # all_diffs = []
    # for i, pt in enumerate(query_points):
        
    #     img1_uv = tuple(pt.astype(np.uint8))
    #     img2_uv_match, best_match_diff = model.find_best_match(img1_uv)
    #     if valid_target_ind[img2_uv_match] != 1 or best_match_diff > 0.08:
    #         continue
    #     else:
    #         all_diffs.append(best_match_diff)
    #         matches.append([img1_uv, img2_uv_match])


    target_points, feat_dists = model.find_best_matches(query_points)
    matches = [[a, b] for a, b in zip(query_points, target_points)]

    if len(matches) < n_max_correspondences:
      distances = [100] * n_max_correspondences
    else:
      distances = [np.linalg.norm(np.array(a[0]) - np.array(a[1])) for a in matches]
    # Sort by pixel distances
    matches = matches[:n_max_correspondences]
    distances = sorted(distances)[:n_max_correspondences]

    if debug:
      colors = pl.cm.jet(np.linspace(0,1,len(matches)))

      # DEBUG PRINT
      plt.figure(figsize=(15,15))
      plt.subplot(1,2,1), plt.imshow(query_image)
      plt.subplot(1,2,2), plt.imshow(target_image)
      for i, match in enumerate(matches):
        img1_uv = match[0]
        plt.subplot(1,2,1)
        ax  = plt.gca()
        ax.scatter(*img1_uv, s=30.0, c=colors[i])
        ax.annotate(i, (img1_uv[0], img1_uv[1]-2), color=colors[i])
        
        img2_uv_match = match[1]
        plt.subplot(1,2,2)
        ax  = plt.gca()
        ax.scatter(*img2_uv_match, s=30.0, c=colors[i])
        ax.annotate(i, (img2_uv_match[0], img2_uv_match[1]-2), color=colors[i])
      print(np.mean(distances))
      plt.show()
      ###


    # set_trace()
    return distances



class BlobDetector(object):
  def __init__(self):

    # Setup SimpleBlobDetector parameters.
    self.params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    self.params.minThreshold = 100
    self.params.maxThreshold = 500

    # Filter by Area.
    self.params.filterByArea = True
    self.params.minArea = 10
    # Filter by Color
    self.params.filterByColor = True
    self.params.blobColor = 0



    # Filter by Circularity
    self.params.filterByCircularity = False
    self.params.minCircularity = 0.2

    # Filter by Convexity
    self.params.filterByConvexity = True
    self.params.minConvexity = 0.2

    # Filter by Inertia
    self.params.filterByInertia = True
    self.params.minInertiaRatio = 0.2

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        self.detector = cv2.SimpleBlobDetector(self.params)
    else:
        self.detector = cv2.SimpleBlobDetector_create(self.params)


    # Read image

  # Set up the detector with default parameters.

  # Detect blobs.

  # Draw detected blobs as red circles.
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

  # Show keypoints
  def get_blobs(self, image, blobColor, whigh=240, wlow=0, hhigh=140, hlow=50):

    self.params.blobColor = blobColor
    gray_image = cv2.cvtColor(image[:, :], cv2.COLOR_RGB2GRAY)    

    keypoints = self.detector.detect(gray_image)
    keypoints = [k for k in keypoints if k.pt[1] > hlow and k.pt[1] < hhigh and k.pt[0] > wlow and k.pt[0] < whigh]
    im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints, np.array([]), (0, 0, 255),
                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    points = [k.pt for k in keypoints]
    return points, im_with_keypoints
