import argparse
import os
from os.path import join
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

# Mask RCNN imports
sys.path.append('/home/msieb/projects/Mask_RCNN')
FROM_DATASET = True
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from samples.bullet.bullet import BulletConfig, BulletDataset
from samples.bullet.config import GeneralConfig
gconf = GeneralConfig()
DATASET_DIR, MODEL_DIR, OLD_MODEL_PATH, COCO_MODEL_PATH, WEIGHTS_FILE_PATH, EXP_DIR  \
        = gconf.DATASET_DIR, gconf.MODEL_DIR, gconf.OLD_MODEL_PATH, gconf.COCO_MODEL_PATH, gconf.WEIGHTS_FILE_PATH, gconf.EXP_DIR

from gps.agent.bullet.bullet_utils import tinyDepth, unproject_2d_point, tinyDepth_no_rescale
from ipdb import set_trace

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
plt.ion()

class InferenceConfig(BulletConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig, ax

def main(args):
    dataset_dir = args.dataset_dir
    seqname = args.seqname
    fig, ax = visualize.get_ax()
    inference_config = InferenceConfig()
    target_ids = gconf.CLASS_IDS
    target_ids = [1,3]
    class_names = gconf.CLASS_NAMES_W_BG
    colors = visualize.random_colors(7)
    with open('{}/{}_relevant_ids_names.json'.format(dataset_dir, seqname), 'r') as f:
        NAME_TO_BULLET_ID = json.load(f)
    with open('{}/{}_camera_params.json'.format(dataset_dir, seqname), 'r') as f:
        CAMERA_PARAMS = json.load(f)

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                                  config=inference_config)

    model.load_weights(WEIGHTS_FILE_PATH, by_name=True)

    dataset = BulletDataset()
    dataset.load_bullet(DATASET_DIR, "train")
    dataset.prepare()
    colors = visualize.random_colors(10)

    objects_centroids_mrcnn = {str(key): [] for key in NAME_TO_BULLET_ID.values()}


    if FROM_DATASET:

        dataset_dir_rgb = join(dataset_dir, 'rgb')
        dataset_dir_depth = join(dataset_dir, 'depth')
        save_path = os.path.join(dataset_dir_rgb, 'mrcnn_output')

        filenames = os.listdir(dataset_dir_rgb)
        filenames = [file for file in filenames if '.jpg' in file or '.png' in file]
        filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        for ii, file in enumerate(filenames):
            if file.endswith('.jpg'):
                image = plt.imread(os.path.join(dataset_dir_rgb, file))
            else:
                image = (plt.imread(os.path.join(dataset_dir_rgb, file))[:, :, :-1]*255).astype(np.uint8)

            depth_raw = plt.imread(os.path.join(dataset_dir_depth, file))
            depth_rescaled = tinyDepth(np.repeat(depth_raw[:,:,None], 3, axis=2), CAMERA_PARAMS["nearPlane"], CAMERA_PARAMS["farPlane"])
            depth_image = depth_rescaled[:, :, 0]
            results = model.detect([image], verbose=1)
            res = results[0]
            encountered_ids = []
            filtered_inds = []
            for i, box in enumerate(res['rois']):
                # ROIs are sorted after confidence, if one was registered discard lower confidence detections to avoid double counting
                class_id = res['class_ids'][i]
                if class_id not in target_ids or class_id in encountered_ids:
                    continue
                encountered_ids.append(class_id)
                cropped = utils.crop_box(image, box, y_offset=20, x_offset=20)
                # cropped = utils.resize_image(cropped, max_dim=299)[0]
                cropped = cv2.resize(cropped, (299, 299))
                # all_cropped_boxes.append(cropped)
                masked_depth = depth_image * res['masks'][:, :, i]
                masked_depth = masked_depth[np.where(masked_depth > 0)]
                x, y = utils.get_box_center(box)
                z = np.median(np.sort(masked_depth.flatten()))
                centroid = [x, y , z]
                # all_centroids_unordered.append(centroid)
                objects_centroids_mrcnn[NAME_TO_BULLET_ID[class_names[class_id]]].append(centroid)

                # all_visual_features_unordered.append(res['roi_features'][i])
                filtered_inds.append(i)
            filtered_inds = np.array(filtered_inds)
            # set_trace()
            # fig, ax = visualize.get_ax()
            fig.clf()
            # only plot highest scoring one per class
            ax = visualize.display_instances(image, res['rois'][filtered_inds], res['masks'][:, :, filtered_inds], res['class_ids'][filtered_inds],            
                        class_names, res['scores'][filtered_inds], ax=fig.gca(), colors=colors)
            # plot all detections
            # ax = visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'],            
            #             self.class_names, res['scores'], ax=self.fig.gca(), colors=self.colors)
            plt.pause(0.001)
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            canvas.draw()       # draw the canvas, cache the renderer
            rcnn_output = np.array(fig.canvas.renderer._renderer)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(os.path.join(save_path, '{}.png'.format(file.split('.')[0])))
            with open('{}/{}_objects_centroid_mrcnn.json'.format(dataset_dir, seqname), 'w') as f:
                json.dump(objects_centroids_mrcnn, f)
            plt.close()
        # Create video
        filenames = [file for file in os.listdir(save_path) if '.jpg' in file or '.png' in file]
        filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        vidwriter = imageio.get_writer(join(save_path, str(seqname)+'.mp4'))
        for ii, file in enumerate(filenames):
            vidwriter.append_data(plt.imread(join(save_path, file)))
        vidwriter.close()

    else:
        # for seq_name in os.listdir('/home/msieb/projects/Mask_RCNN/datasets/bullet/test'):
        #     print("Processing ", seq_name)
        # # seq_name = args.seqname
        #     dataset_dir = os.path.join(DATASET_DIR, "test", seq_name)
        #     filenames = os.listdir(dataset_dir)
        #     filenames = [file for file in filenames if '.jpg' in file]
        #     filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        dataset_dir = '/home/msieb/projects/Mask_RCNN/datasets/bullet/test'
        filenames = os.listdir(dataset_dir)
        filenames = [file for file in filenames if '.jpg' in file]
        filenames = sorted(filenames, key=lambda x: x.split('.')[0])
        for ii, file in enumerate(filenames):
            # if not ii % 1 == 0:
            #     continue
            # # Load image and ground truth data
            # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            #     modellib.load_image_gt(dataset, inference_config,
            #                            image_id, use_mini_mask=False)
            # molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # # Run object detection
            # results = model.detect([image], verbose=0)
            # res = results[0]
            # # Compute AP
            # AP, precisions, recalls, overlaps =\
            #     utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
            #                      res["rois"], res["class_ids"], res["scores"], res['masks'])
            # APs.append(AP)

            image = plt.imread(os.path.join(dataset_dir, file))

            results = model.detect([image], verbose=1)
            fig, ax = get_ax()
            res = results[0]
            ax = visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], 
                                    dataset.class_names, res['scores'], ax=ax, colors=colors)
            save_path = os.path.join(EXP_DIR, 'runs', args.runname)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(os.path.join(save_path, '{}.png'.format(file.strip('.jpg'))))
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seqname', type=str, default=16)
    parser.add_argument('-d', '--dataset-dir', type=str, default='test')
    parser.add_argument('-r', '--runname', type=str, default='test')
    args = parser.parse_args()
    main(args)

