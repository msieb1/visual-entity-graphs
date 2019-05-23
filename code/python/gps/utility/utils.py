import torch
import numpy as np
import math
# import pybullet_data
import os
import numpy as np
import pybullet as p
import struct

# Standard imports
import cv2
import matplotlib.pyplot as plt
from skimage import io, transform

import numpy as np
from pdb import set_trace
from collections import OrderedDict
from imageio import imwrite
import imageio
from PIL import Image
import sys
import os
from os.path import join
from copy import deepcopy as copy
import importlib
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
import numpy as np

from pdb import set_trace as st
# sys.path.append('/home/max/projects/gps-lfd')
# sys.path.append('/home/msieb/projects/gps-lfd')



#from view_tcn import define_model as define_model # different model architectures - fix at some point because inconvenient having different loading methods for each model
from plot_utils import plot_results
import pylab as pl


CONTROLLER_ID = 0
POSITION=1
ORIENTATION=2
ANALOG=3
BUTTONS=6

BUTTON_MAPS = {'trigger':33, 'menu':1, 'side':2, 'pad':32}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        
        tsfm_labels = np.zeros((img.shape[0], img.shape[1], 2))
        for i in range(labels.shape[-1]):
            landmarks = labels[..., i]
            points = list(np.where(landmarks == 1))
            points[0] = np.array(points[0] * new_h / h, dtype=np.int32)
            points[1] = np.array(points[1] * new_w / w, dtype=np.int32)
            buff = np.zeros((img.shape[0], img.shape[1]))
            
            try:
                buff[tuple(points)] = 1
                tsfm_labels[..., i] = buff
            except:
                import ipdb; ipdb.set_trace()
        return {'image': img, 'label': tsfm_labels}

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
	"""
	Theta is given as euler angles Z-Y-X, corresponding to yaw, pitch, roll
	"""
	 
	R_x = np.array([[1,         0,                  0                   ],
					[0,         math.cos(theta[0]), -math.sin(theta[0]) ],
					[0,         math.sin(theta[0]), math.cos(theta[0])  ]
					])
		 
		 
					 
	R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
					[0,                     1,      0                   ],
					[-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
					])
				 
	R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
					[math.sin(theta[2]),    math.cos(theta[2]),     0],
					[0,                     0,                      1]
					])
					 
					 
	R = np.dot(R_z, np.dot( R_y, R_x ))
 
	return R


	# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
	assert(isRotationMatrix(R))
	 
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	 
	singular = sy < 1e-6
 
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
 
	return np.array([x, y, z])


def create_rot_from_vector(vector):
	"""
	vector should be 6 dimensional
	"""
	# random unit vectors
	u = vector[:3]
	v = vector[3:]
	u /= np.linalg.norm(u)
	v /= np.linalg.norm(v)
	# subtract (v*u)u from v and normalize
	v -= v.dot(u)*u
	v /= np.linalg.norm(v)
	# build cross product
	w = np.cross(u, v)
	w /= np.linalg.norm(w)
	R = np.hstack([u[:,None], v[:,None], w[:,None]])
	assert isRotationMatrix(R)
	return R



def sysdatapath(*filepath):
  return os.path.join(pybullet_data.getDataPath(), *filepath)


def euclidean_dist(point1, point2):
	assert len(point1) == 3
	assert len(point2) == 3
	point1 = np.array(point1)
	point2 = np.array(point2)
	return np.linalg.norm(point1 - point2)


def readLogFile(filename, verbose = True):
  f = open(filename, 'rb')

  print('Opened'),
  print(filename)

  keys = f.readline().decode('utf8').rstrip('\n').split(',')
  fmt = f.readline().decode('utf8').rstrip('\n')

  # The byte number of one record
  sz = struct.calcsize(fmt)
  # The type number of one record
  ncols = len(fmt)

  if verbose:
    print('Keys:'),
    print(keys)
    print('Format:'),
    print(fmt)
    print('Size:'),
    print(sz)
    print('Columns:'),
    print(ncols)

  # Read data
  wholeFile = f.read()
  # split by alignment word
  chunks = wholeFile.split(b'\xaa\xbb')
  log = list()
  for chunk in chunks:
    if len(chunk) == sz:
      values = struct.unpack(fmt, chunk)
      record = list()
      for i in range(ncols):
        record.append(values[i])
      log.append(record)

  return log
  
def clean_line(line):
    """
    Cleans a single line of recorded joint positions
    @param line: the line described in a list to process
    @param joint_names: joint name keys
    @return command: returns dictionary {joint: value} of valid commands
    @return line: returns list of current line values stripped of commas
    """
    def try_float(x):
        try:
            return float(x)
        except ValueError:
            return None
    #convert the line of strings to a float or None
    line = [try_float(x) for x in line.rstrip().split(' ')]
    #zip the values with the joint names
    #take out any tuples that have a none value
    #convert it to a dictionary with only valid commands

    ID = str(int(line[1]))
    return (ID, line,)



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

def load_pixel_model(basedir, use_cuda=True):
    """
    Assumes weight_files folder is within the folder of the currently executed experiment.
    Also assumes that the name of the used model in pytorch is given by the first part of the '-'separated file name, i.e. tcn_2-epoch-10.pk
    would equal the model name tcn_2
    """
    # basedir = '/home/msieb/projects/pixel/src'

    sys.path.append(basedir)
    module = importlib.import_module('correspondence_finder')
    CorrespondenceFinder = getattr(module, 'CorrespondenceFinder')
    cf = CorrespondenceFinder()
    if use_cuda:
        cf = cf
    return cf

def load_tcn_model(model_path, use_cuda=True):
    """
    Assumes weight_files folder is within the folder of the currently executed experiment.
    Also assumes that the name of the used model in pytorch is given by the first part of the '-'separated file name, i.e. tcn_2-epoch-10.pk
    would equal the model name tcn_2
    """
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

def load_eenet_model(model_path, use_cuda=True):
    """
    Assumes weight_files folder is within the folder of the currently executed experiment.
    Also assumes that the name of the used model in pytorch is given by the first part of the '-'separated file name, i.e. tcn_2-epoch-10.pk
    would equal the model name tcn_2
    """
    basedir = os.path.join('/'.join(model_path.split('/')[:-5]), 'git/eenet/')
    model_name = model_path.split('/')[-1].split('-')[0]
    sys.path.append(basedir)
    module = importlib.import_module('models.' + model_name)
    define_model = getattr(module, 'define_model')
    IMG_HEIGHT = 240 # These are the dimensions used as input for the ConvNet architecture, so these are independent of actual image size
    IMG_WIDTH = 320
    model = define_model(IMG_HEIGHT, IMG_WIDTH, use_cuda)
    # tcn = torch.nn.DataParallel(tcn, device_ids=range(1))
    # Change dict names if model was created with nn.DataParallel
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    # for k, v in state_dict.items():
    #     name = k[7:] # remove module.
    #     new_state_dict[name] = v
    # map_location allows us to load models trained on cuda to cpu.
    model.load_state_dict(state_dict)
    if use_cuda:
      model = model.cuda()
    return model

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
  return mean, std
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
    return embeddings_normalized



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
    valid_query_ind[np.where(query_mask == 1)] = 1
    valid_query_ind = np.where(valid_query_ind)

    sample_inds = np.random.choice(len(valid_query_ind[0]), len(valid_query_ind[0]), replace=False)
    points = []
    for i in sample_inds:
        add_point = True
        curr_pt = (valid_query_ind[1][i], valid_query_ind[0][i]) 
        # for pt in points:
        #     dist = np.linalg.norm(np.array(curr_pt) - np.array(pt))
            # if dist < 1:
            #     add_point = False
            #     continue
        if add_point:
            points.append(curr_pt)
        if len(points) >= n_points_final:
            break
    assert len(points) == n_points_final


    return points

def get_sample_points(query_mask, n_samples):

    valid_query_ind = np.zeros(query_mask.shape)
    valid_query_ind[np.where(query_mask == 1)] = 1
    valid_query_ind = np.where(valid_query_ind)

    sample_inds = np.random.choice(len(valid_query_ind[0]), n_samples, replace=False)
    points = np.array(valid_query_ind).T
    points = points[:, [1, 0]]
    return points[sample_inds]


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

    model.feed_images_rgb(rgb_to_pil(query_image), rgb_to_pil(target_image), flags=[True, True], img_target_mask=valid_target_ind)
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
      # plt.figure(figsize=(15,15))
      # plt.subplot(1,2,1)
      # ax  = plt.gca()
      # ax.clear()
      # plt.subplot(1,2,2)
      # ax  = plt.gca()
      # ax.clear()
      # plt.subplot(1,2,1), plt.imshow(query_image)
      # plt.subplot(1,2,2), plt.imshow(target_image)

      # for i, match in enumerate(matches):
      #   img1_uv = match[0]
      #   plt.subplot(1,2,1)
      #   ax  = plt.gca()
      #   ax.scatter(*img1_uv, s=30.0, c=colors[i])
      #   ax.annotate(i, (img1_uv[0], img1_uv[1]-2), color=colors[i])
        
      #   img2_uv_match = match[1]
      #   plt.subplot(1,2,2)
      #   ax  = plt.gca()
      #   ax.scatter(*img2_uv_match, s=30.0, c=colors[i])
      #   ax.annotate(i, (img2_uv_match[0], img2_uv_match[1]-2), color=colors[i])
      # print(np.mean(distances))
      # plt.show()
      # plt.pause(.001)
      # import IPython;IPython.embed()
      mask_img = np.ceil(valid_target_ind * 255)
      target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
      query_bgr = cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR)
      for i, match in enumerate(matches):
        query_uv = match[0]
        target_uv = match[1]
        color = colors[i]
        cv2.circle(target_bgr, tuple(target_uv), 3, np.ceil(color[:-1]*255), -1)
        cv2.circle(query_bgr, tuple(query_uv), 3,np.ceil(color[:-1]*255), -1)
      cv2.imshow('target', target_bgr)
      cv2.imshow('query', query_bgr)
      # cv2.imshow('mask1', mask_img)
      k = cv2.waitKey(1)
      ###

    # set_trace()
    return distances

def compute_pixel_centroid_and_rotation_difference(model, query_points, query_image, valid_target_ind, target_image, n_max_correspondences, query_centroid=None, target_centroid=None, debug=False, use_cuda=True, alpha=3.0/4):
    # plt.imsave('/home/msieb/misc/for_zhouxian/0_00000.png', query_image)
    # plt.imsave('/home/msieb/misc/for_zhouxian/0_00001.png', target_image)
    model.feed_images_rgb(rgb_to_pil(query_image), rgb_to_pil(target_image), flags=[True, True], img_target_mask=valid_target_ind)

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

    target_points = np.array(target_points)
    query_points = np.array(query_points)

    if query_centroid is None or target_centroid is None:
      query_centroid = np.mean(query_points, axis=0)
      target_centroid = np.mean(target_points, axis=0)
    else:
      pass
    

    query_points_local_frame = query_points - query_centroid
    target_points_local_frame = target_points - target_centroid
    # alpha = 1.0
    # centroid_diff = np.linalg.norm(query_centroid - target_centroid)
    # distances = (1 - alpha)*centroid_diff + alpha*rotation_diff
    rotation_diff = np.array([np.sum(np.linalg.norm(query_points_local_frame - target_points_local_frame, axis=1))]) / len(query_points_local_frame)
    
    # rotation_diff = np.sum(query_points_local_frame - target_points_local_frame, axis=0)

    if debug:
      colors = pl.cm.jet(np.linspace(0,1,len(matches)))

      mask_img = np.ceil(valid_target_ind * 255)
      target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
      query_bgr = cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR)
      for i, match in enumerate(matches):
        query_uv = match[0]
        target_uv = match[1]
        color = colors[i]
        cv2.circle(target_bgr, tuple(target_uv), 3, np.ceil(color[:-1]*255), -1)
        cv2.circle(query_bgr, tuple(query_uv), 3,np.ceil(color[:-1]*255), -1)
      cv2.imshow('target', target_bgr)
      cv2.imshow('query', query_bgr)
      # cv2.imshow('mask1', mask_img)
      k = cv2.waitKey(1)
      ###
      ###
    # set_trace()
    # print(distance)
    return rotation_diff


  

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




def grabcut(img, valid_depth):

    mask = np.zeros(img.shape[:2],np.uint8)
    # Make mask generation easier> 3 means possibly foreground, 1 means definitely foreground
    mask[np.where(valid_depth == 1)] = 3 
    # mask[np.where(valid_depth * valid_rgb)] = 1 

    # mask[np.where(valid == False)] = 2
    # mask[:100, 0:30] = 0
    # mask[-100:, 0:30] = 0
    # mask[:200, 0:50] = 0
    # mask[-100:, -30:] = 0
    mask[600:, :] = 0
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    rps = sorted((regionprops(label(mask > 0.5, background=0))), key=lambda x: x.area, reverse=True)
    mask_clean = np.zeros(mask.shape)
    mask_clean[rps[0].coords[:, 0], rps[0].coords[:,1]] = 1
    # mask_clean = ndi.binary_fill_holes(mask_clean, structure=np.ones((2,2))).astype(np.uint8)
    mask_clean_ = copy(mask_clean)
    # mask_clean = ndi.binary_fill_holes(mask_clean_).astype(np.uint8)
    img_masked = img*mask_clean[:,:,np.newaxis]
    # cv2.imshow('frame',rgb.astype(np.uint8))
    # vis_mask = ndi.binary_erosion(mask)
    # cv2.imshow('mask', mask_clean*255)
    # cv2.imshow('img',img_masked.astype(np.uint8)[:, :, :])
    # plt.imshow(img_masked)
    # plt.show()
    return img_masked, mask_clean

def depth_to_pc(depth_img, T_camera_ref=np.eye(4), scale=1000.0, K=None):
    if K is None:
        K = np.array([
            [615.3900146484375, 0.0, 326.35467529296875],
            [0.0, 615.323974609375, 240.33250427246094],
            [0.0, 0.0, 1.0]])
    img_H, img_W = depth_img.shape
    depth_vec = depth_img/float(scale)
    zero_depth_ids = np.where(depth_vec == 0)
    depth_vec = depth_vec.ravel()
    u_vec, v_vec = np.meshgrid(np.arange(img_W), np.arange(img_H))
    u_vec = u_vec.ravel() * depth_vec
    v_vec = v_vec.ravel() * depth_vec
    full_vec = np.vstack((u_vec, v_vec, depth_vec))

    pc_camera = np.linalg.inv(K).dot(full_vec)
    pc_camera = np.vstack([pc_camera, np.ones(img_H*img_W)])
    pc_ref = T_camera_ref.dot(pc_camera)[:3].T

    return pc_ref.reshape((img_H, img_W, -1)), zero_depth_ids

def depth_to_mask(depth_img, limit_lower, limit_upper, T_camera_world, scale=1000.0, K=None):
    # import IPython;IPython.embed()
    pc_world, zero_depth_ids = depth_to_pc(depth_img, T_camera_world, scale, K)

    mask = np.logical_and(pc_world<limit_upper, 
                          pc_world>limit_lower).all(axis=2)
    mask[zero_depth_ids[0], zero_depth_ids[1]] = False

    return mask

def create_writer(root_path, classifier='', fps=10):
    if not os.path.exists(join(root_path, 'vids')):
        os.makedirs(join(root_path, 'vids'))
    rgb_path = join(root_path, 'vids', "rgb_sample_{}.mp4".format(classifier))
    rgb_writer = imageio.get_writer(rgb_path, fps=fps)
    return rgb_writer
