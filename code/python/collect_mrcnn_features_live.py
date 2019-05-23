import sys
import rospy
import os
import argparse
import imageio
import time
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_ROS
# from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
#         policy_to_msg, tf_policy_to_action_msg, tf_obs_msg_to_numpy
# from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
# from gps_agent_pkg.msg import TrialCommand, SampleResult, PositionCommand, \
#         RelaxCommand, DataRequest, TfActionCommand, TfObsData
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, NOISE, TCN_EMBEDDING, RGB_IMAGE

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None

from gps.sample.sample import Sample

from imageio import imwrite
import numpy as np
from pygame import mixer
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from gps.agent.baxter.baxter_utils import img_subscriber, depth_subscriber
from ipdb import set_trace


# Mask RCNN imports
sys.path.append('/home/msieb/projects/Mask_RCNN/samples')
import tensorflow as tf
from baxter.baxter import BaxterConfig
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
print(tf.__version__)

# TCN relevant imports
sys.path.append('/home/msieb/projects/LTCN')
from tcn import define_model


EMBEDDING_DIM = 3
T = 10
IMAGE_SIZE = (299, 299)
dt = 0.5
FPS = 1/dt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"

SAMPLE_NR=''
OUTPUT_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/rotating_rcnn_reward_eval/embeddings_live'
MODEL_PATH = "/home/msieb/projects/Mask_RCNN/weights/baxter20180707T1715/mask_rcnn_baxter_0019.h5"
# MODEL_PATH = "/home/msieb/projects/Mask_RCNN/weights/baxter20180707T1715/mask_rcnn_baxter_0019.h5"   # BEST WEIGHTS SO FAR
MODEL_DIR = '/home/msieb/projects/gps/experiments/baxter_reaching/data_files'

VIEW = 0


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

def save_image_file(folder_path, name, file):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name)
    imwrite(filepath, file)

def create_video_of_sample(folder_path, name, images):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    vidpath = os.path.join(folder_path, name + '.mp4')
    writer = imageio.get_writer(vidpath, fps=FPS)
    for i in range(len(images)):
        writer.append_data(images[i])
    writer.close()

def get_depth_img(depth_subs_obj):
    depth_scale = 0.001 # not precisely, but up to e-8
    clipping_distance_in_meters = 1.5 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    depth_image = depth_subs_obj.img
    depth_image[np.where(depth_image > clipping_distance)] = 0
    depth_rescaled = (((depth_image  - 0) / (clipping_distance - 0)) * (255 - 0) + 0).astype(np.uint8)
    return depth_rescaled

def load_tcn_model(model_path, use_cuda=True):
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

class Embedder(object):
    def __init__(self, output_folder):
        class InferenceConfig(BaxterConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        self.tcn = load_tcn_model('/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/ltcn/trained_models/tcn-no-depth-sv/ltcn-epoch-8.pk')
        inference_config = InferenceConfig()

        with tf.device('/device:GPU:1'):
            self.rcnn = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
                                      config=inference_config)
            self.rcnn.load_weights(MODEL_PATH, by_name=True)
        # with tf.device('/device:GPU:0'):
        #     self.feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            # feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.class_names = ['BG', 'blue_ring', 'green_ring', 'yellow_ring', 'tower', 'hand', 'robot']
        self.target_ids = [1, 4]
        self.colors = visualize.random_colors(7)
        self.plot_mode = True
        self.output_folder = output_folder

    def get_rcnn_features(self, image, depth_rescaled):
        results = self.rcnn.detect([image], verbose=0)
        r = results[0]
        encountered_ids = []
        all_cropped_boxes = []
        all_centroids_unordered = [] # X Y Z
        all_centroids = dict()
        all_features_unordered = []
        all_features = dict()
        for i, box in enumerate(r['rois']):
            class_id = r['class_ids'][i]
            if class_id not in self.target_ids or class_id in encountered_ids:
                continue
            encountered_ids.append(class_id)
            cropped = utils.crop_box(image, box, y_offset=0, x_offset=0)
            # cropped = utils.resize_image(cropped, max_dim=299)[0]
            cropped = cv2.resize(cropped, (299, 299))
            all_cropped_boxes.append(cropped)
            masked_depth = depth_rescaled * r['masks'][:, :, i]
            masked_depth = masked_depth[np.where(masked_depth > 0)]
            z = np.median(np.sort(masked_depth.flatten()))
            x, y = utils.get_box_center(box)
            all_centroids_unordered.append([x, y, z])
            all_features_unordered.append(r['roi_features'][i])
        all_cropped_boxes = np.asarray(all_cropped_boxes)
        all_centroids_unordered = np.asarray(all_centroids_unordered)
        for i in range(all_cropped_boxes.shape[0]):
            all_features[encountered_ids[i]] = all_features_unordered[i]
            all_centroids[encountered_ids[i]] = all_centroids_unordered[i]
        all_centroids = np.asarray([val for key, val in all_centroids.items()])
        all_features = np.asarray([val for key, val in all_features.items()])
        if self.plot_mode:
            fig, ax = visualize.get_ax()
            ax = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],            
                        self.class_names, r['scores'], ax=ax, colors=self.colors)
        else:
            fig = None
        return all_features, all_centroids, fig


def main(args):

    EMBEDDING_DIM = 3

    embedder = Embedder(args.output_folder)

    rospy.init_node('embeddings_collector', anonymous=True)
    # pub = rospy.Publisher('/tcn/embedding', numpy_msg(Floats), queue_size=3)
    img_subs_obj = img_subscriber(topic="/camera" + str(args.view + 1) + "/color/image_raw")
    depth_subs_obj = depth_subscriber(topic="/camera" + str(args.view + 1) + "/aligned_depth_to_color/image_raw")
    # try:
    #     image = img_subs_obj.img
    #     depth_image = depth_subs_obj.img
    # except:
    #     print("Camera not working / connected, check connection")
    #     print("Exit function")
    #     return
    rospy.sleep(0.5)
    for i in range(100):
        print('Taking ramp image %d.' % i)
        image = img_subs_obj.img
        depth_image = depth_subs_obj.img
    results = embedder.rcnn.detect([image], verbose=0)

    while True:
        try:
            len_recording = 1*T
            embeddings = np.zeros((len_recording, EMBEDDING_DIM))
            embeddings_visual = np.zeros((len_recording, 2, 7, 7, 256))
            image_buffer = []
            rcnn_image_buffer= []

            print("="*20)
            print('Starting recording...')
            for t in range(len_recording):
                curr_time = rospy.get_time()
                # print(curr_time)
                image = img_subs_obj.img
                depth_rescaled = get_depth_img(depth_subs_obj)
                resized_image = resize_frame(image, IMAGE_SIZE)[None, :]
                resized_depth = resize_frame(np.tile(depth_rescaled[:, :, None], (1,1,3)), IMAGE_SIZE)[None, :]
                # resized_depth = resize_frame(depth_rescaled[:, :, None], IMAGE_SIZE)[None, :]
                frame = np.concatenate([resized_image[0], resized_depth[0, None, 0]], axis=0)

                all_features, all_centroids, fig = embedder.get_rcnn_features(image, depth_rescaled)
                delta_centroid = all_centroids[0] - all_centroids[1]
                embeddings[t, :] = delta_centroid
                embeddings_visual[t, :, :, :, :] = all_features
                inference_time = rospy.get_time() - curr_time
                image_buffer.append(image)
                if fig is not None:
                    canvas = FigureCanvas(fig)
                    ax = fig.gca()
                    canvas.draw()       # draw the canvas, cache the renderer
                    rcnn_image_buffer.append(np.array(fig.canvas.renderer._renderer))
                rospy.sleep(dt - inference_time)
                print("step: ", t, "runtime: ", rospy.get_time() - curr_time)
              # Create and append a sequence name.
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            if args.sample_nr:
                seqname = args.sample_nr
            else:
                if not os.listdir(args.output_folder):
                  seqname = '0'
                else:
                  # Otherwise, get the latest sequence name and increment it.
                  seq_names = [int(i.split('_')[0]) for i in os.listdir(args.output_folder)]
                  latest_seq = sorted(seq_names, reverse=True)[0]
                  seqname = str(latest_seq+1)
                print('No seqname specified, using: %s' % seqname)
            print("Finished recording")


            save_np_file(folder_path=args.output_folder, name='{}_view{}_emb'.format(seqname, args.view), file=embeddings)
            save_np_file(folder_path=args.output_folder, name='{}_view{}_emb_visual'.format(seqname, args.view), file=embeddings_visual)

            create_video_of_sample(os.path.join(args.output_folder), "{}_view{}_video_sample".format(seqname, args.view), image_buffer)
            create_video_of_sample(os.path.join(args.output_folder), "{}_view{}_video_sample_rcnn".format(seqname, args.view), rcnn_image_buffer)
            if not args.collect_multiple:
                break
            rospy.sleep(5.0)
            # for t, img in enumerate(image_buffer):
            #     save_image_file(folder_path=os.path.join(args.output_folder, 'images'), name='{0:05d}.png'.format(t), file=img)
        except KeyboardInterrupt:
            return
    print('Exit function')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=MODEL_PATH)
    parser.add_argument('--output-folder', type=str, default=OUTPUT_PATH)
    parser.add_argument('--view', type=str, default=VIEW)
    parser.add_argument('--sample-nr', type=str, default=SAMPLE_NR)
    parser.add_argument('-c', '--collect-multiple', type=bool, default=False)

    args = parser.parse_args()
    main(args)        
