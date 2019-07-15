from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType

import os
import cv2
import numpy as np
import logging
import glob
import random

import torch
import torch.utils.data as data

from PIL import Image

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision
from torchvision import transforms

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

import dense_correspondence_manipulation.utils.constants as constants


utils.add_dense_correspondence_to_python_path()
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation

import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


class SaladDataset(data.Dataset):

    PADDED_STRING_WIDTH = 6

    def __init__(self, config, debug=False, mode="train"):
        self.debug = debug
        self.mode = mode

        self.setup_dataset(config)
        self.setup_data_augmenter()

        print "Using SaladDataset:"
        print "   - in", self.mode, "mode"
        print "   - total images:    ", self.num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
 
        img_orig_PIL = self.get_PIL_img(index)
        img_orig_rgb = np.array(img_orig_PIL)

        # sample pixels
        uv_orig = self.get_random_pixels(width=img_orig_rgb.shape[1], height=img_orig_rgb.shape[0], num_samples=self.num_matching_pixels*4)
        keypoints_on_images_orig = [ia.KeypointsOnImage.from_coords_array(coords=uv_orig, shape=img_orig_rgb.shape)]
        
        # augment image to generate a pair of images for training and find correspondences
        aug_seq_det_a = self.aug_seq.to_deterministic()
        img_a_rgb = aug_seq_det_a.augment_image(img_orig_rgb)
        keypoints_a = aug_seq_det_a.augment_keypoints(keypoints_on_images_orig)[0]
        uv_a = np.rint(keypoints_a.get_coords_array()).astype(int)
        
        aug_seq_det_b = self.aug_seq.to_deterministic()
        img_b_rgb = aug_seq_det_b.augment_image(img_orig_rgb)
        keypoints_b = aug_seq_det_b.augment_keypoints(keypoints_on_images_orig)[0]
        uv_b = np.rint(keypoints_b.get_coords_array()).astype(int)
        
        # remove pixels outside frame
        within_a = np.logical_and(uv_a >= [0, 0], uv_a < [img_orig_rgb.shape[1], img_orig_rgb.shape[0]])
        within_b = np.logical_and(uv_b >= [0, 0], uv_b < [img_orig_rgb.shape[1], img_orig_rgb.shape[0]])
        valid_ids = np.where(np.logical_and(within_a, within_b).all(axis=1))[0]
        uv_orig = uv_orig[valid_ids][:self.num_matching_pixels]
        uv_a = uv_a[valid_ids][:self.num_matching_pixels]
        uv_b = uv_b[valid_ids][:self.num_matching_pixels]
        
        if self.debug:
            self.aug_seq.show_grid(img_orig_rgb, cols=4, rows=4)
            img_a_cv = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2BGR)
            img_b_cv = cv2.cvtColor(img_b_rgb, cv2.COLOR_RGB2BGR)
            for pix_a, pix_b in zip(uv_a, uv_b):
                color = cv2.cvtColor(np.array([[[np.random.randint(256), 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0][0].astype(int)
                cv2.circle(img_a_cv, tuple(pix_a), 5, color, -1)
                cv2.circle(img_b_cv, tuple(pix_b), 5, color, -1)

            cv2.imshow('a', img_a_cv)
            cv2.imshow('b', img_b_cv)
            k = cv2.waitKey(0)

        # convert to torch Tensor
        uv_a = (torch.from_numpy(uv_a[:, 0]).type(torch.LongTensor), torch.from_numpy(uv_a[:, 1]).type(torch.LongTensor))
        uv_b = (torch.from_numpy(uv_b[:, 0]).type(torch.FloatTensor), torch.from_numpy(uv_b[:, 1]).type(torch.FloatTensor))

        # find non_correspondences
        uv_b_non_matches = correspondence_finder.create_non_correspondences(uv_b, img_b_rgb.shape, num_non_matches_per_match=self.num_non_matches_per_match)

        # convert PIL.Image to torch.FloatTensor
        image_a_rgb = self.rgb_image_to_tensor(img_a_rgb)
        image_b_rgb = self.rgb_image_to_tensor(img_b_rgb)

        image_width = img_orig_rgb.shape[1]
        image_height = img_orig_rgb.shape[0]
        matches_a = SaladDataset.flatten_uv_tensor(uv_a, image_width)
        matches_b = SaladDataset.flatten_uv_tensor(uv_b, image_width)

        uv_a_long, uv_b_non_matches_long = self.create_non_matches(uv_a, uv_b_non_matches, self.num_non_matches_per_match)

        non_matches_a = SaladDataset.flatten_uv_tensor(uv_a_long, image_width).squeeze(1)
        non_matches_b = SaladDataset.flatten_uv_tensor(uv_b_non_matches_long, image_width).squeeze(1)

        # 5 is custom data type for distortion image training
        return 5, image_a_rgb, image_b_rgb, matches_a, matches_b, non_matches_a, non_matches_b

    def get_random_pixels(self, width, height, num_samples):
        rand_us = np.random.rand(num_samples) * width
        rand_vs = np.random.rand(num_samples) * height
        rand_uvs = np.floor(np.vstack((rand_us, rand_vs))).astype(int).T
        return rand_uvs

    def setup_dataset(self, config):
        if self.mode == 'train':
            self.dataset_dir = config['trainset_dir']
        elif self.mode == 'test':
            self.dataset_dir = config['testset_dir']
        self._initialize_rgb_image_to_tensor()

        self.img_width = config['img_width']
        self.img_height = config['img_height']

        self._config = config

        images_regex = os.path.join(self.dataset_dir, config['img_name_format'])
        self.image_paths = glob.glob(images_regex)
        self.num_images = len(self.image_paths)

    def setup_data_augmenter(self):
        aug_sometimes_80 = lambda aug: iaa.Sometimes(0.8, aug)
        aug_sometimes_50 = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_seq = iaa.Sequential([
            iaa.Fliplr(0.3), 
            iaa.Flipud(0.3), 
            aug_sometimes_80(iaa.Affine(
                scale=(0.3, 3.0),
                translate_percent=({"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), # translate by -20 to +20 percent (per axis)
                rotate=(-60, 60),
                shear=(-30, 30),
                order=1, # use nearest neighbour or bilinear interpolation (fast)
                mode='reflect' 
            )),
            aug_sometimes_80(iaa.Add((-20, 20), per_channel=0.5)), # change brightness of images
            aug_sometimes_50(iaa.AddToHueAndSaturation((-10, 10))), # change hue and saturation
        ])

    def get_PIL_img(self, index):
        img_file = self.image_paths[index]
        img_PIL = Image.open(img_file).convert('RGB')
        img_PIL = img_PIL.resize((self.img_width, self.img_height), Image.ANTIALIAS)
        return img_PIL

    def set_parameters_from_training_config(self, training_config):
        """
        Some parameters that are really associated only with training, for example
        those associated with random sampling during the training process,
        should be passed in from a training.yaml config file.

        :param training_config: a dict() holding params
        """

        self.num_matching_pixels = int(training_config['training']['num_matching_pixels'])
        self.num_non_matches_per_match = training_config['training']["num_non_matches_per_match"]


    def scene_generator(self, mode=None):
        """
        Returns a generator that traverses all the scenes
        :return:
        :rtype:
        """
        if mode is None:
            mode = self.mode

        for object_id, single_object_scene_dict in self._single_object_scene_dict.iteritems():
            for scene_name in single_object_scene_dict[mode]:
                yield scene_name

        for scene_name in self._multi_object_scene_dict[mode]:
            yield scene_name

    def _initialize_rgb_image_to_tensor(self):
        """
        Sets up the RGB PIL.Image --> torch.FloatTensor transform
        :return: None
        :rtype:
        """
        norm_transform = transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        self._rgb_image_to_tensor = transforms.Compose([transforms.ToTensor(), norm_transform])


    def get_within_scene_data(self, scene_name, metadata, for_synthetic_multi_object=False):
        """
        The method through which the dataset is accessed for training.

        Each call is is the result of
        a random sampling over:
        - random scene
        - random rgbd frame from that scene
        - random rgbd frame (different enough pose) from that scene
        - various randomization in the match generation and non-match generation procedure

        returns a large amount of variables, separated by commas.

        0th return arg: the type of data sampled (this can be used as a flag for different loss functions)
        0th rtype: string

        1st, 2nd return args: image_a_rgb, image_b_rgb
        1st, 2nd rtype: 3-dimensional torch.FloatTensor of shape (image_height, image_width, 3)

        3rd, 4th return args: matches_a, matches_b
        3rd, 4th rtype: 1-dimensional torch.LongTensor of shape (num_matches)

        5th, 6th return args: masked_non_matches_a, masked_non_matches_b
        5th, 6th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        7th, 8th return args: non_masked_non_matches_a, non_masked_non_matches_b
        7th, 8th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        9th, 10th return args: blind_non_matches_a, blind_non_matches_b
        9th, 10th rtype: 1-dimensional torch.LongTensor of shape (num_non_matches)

        11th return arg: metadata useful for plotting, and-or other flags for loss functions
        11th rtype: dict

        Return values 3,4,5,6,7,8,9,10 are all in the "single index" format for pixels. That is

        (u,v) --> n = u + image_width * v

        If no datapoints were found for some type of match or non-match then we return
        our "special" empty tensor. Note that due to the way the pytorch data loader
        functions you cannot return an empty tensor like torch.FloatTensor([]). So we
        return SpartanDataset.empty_tensor()

        """



    def create_non_matches(self, uv_a, uv_b_non_matches, multiplier):
        """
        Simple wrapper for repeated code
        :param uv_a:
        :type uv_a:
        :param uv_b_non_matches:
        :type uv_b_non_matches:
        :param multiplier:
        :type multiplier:
        :return:
        :rtype:
        """
        uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                     torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1))

        uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

        return uv_a_long, uv_b_non_matches_long

    def get_image_mean(self):
        """
        Returns dataset image_mean
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_MEAN

        # return self.config["image_normalization"]["mean"]

        
        return constants.DEFAULT_IMAGE_MEAN

    def get_image_std_dev(self):
        """
        Returns dataset image std_dev
        :return: list
        :rtype:
        """

        # if "image_normalization" not in self.config:
        #     return constants.DEFAULT_IMAGE_STD_DEV

        # return self.config["image_normalization"]["std_dev"]

        return constants.DEFAULT_IMAGE_STD_DEV

        

    def rgb_image_to_tensor(self, img):
        """
        Transforms a PIL.Image to a torch.FloatTensor.
        Performs normalization of mean and std dev
        :param img: input image
        :type img: PIL.Image
        :return:
        :rtype:
        """

        return self._rgb_image_to_tensor(img)

    @property
    def config(self):
        return self._config

    @staticmethod
    def flatten_uv_tensor(uv_tensor, image_width):
        """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
        """
        return uv_tensor[1].long() * image_width + uv_tensor[0].long()

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]. It has a 1 exactly
        at the indices specified by uv_flat_tensor
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width*image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat


    @staticmethod
    def subsample_tuple(uv, num_samples):
        """
        Subsamples a tuple of (torch.Tensor, torch.Tensor)
        """
        indexes_to_keep = (torch.rand(num_samples) * len(uv[0])).floor().type(torch.LongTensor)
        return (torch.index_select(uv[0], 0, indexes_to_keep), torch.index_select(uv[1], 0, indexes_to_keep))

    @staticmethod
    def subsample_tuple_pair(uv_a, uv_b, num_samples):
        """
        Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor), (torch.Tensor, torch.Tensor)
        """
        assert len(uv_a[0]) == len(uv_b[0])
        indexes_to_keep = (torch.rand(num_samples) * len(uv_a[0])).floor().type(torch.LongTensor)
        uv_a_downsampled = (torch.index_select(uv_a[0], 0, indexes_to_keep), torch.index_select(uv_a[1], 0, indexes_to_keep))
        uv_b_downsampled = (torch.index_select(uv_b[0], 0, indexes_to_keep), torch.index_select(uv_b[1], 0, indexes_to_keep))
        return uv_a_downsampled, uv_b_downsampled


    @staticmethod
    def make_default_10_scenes_drill():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                   'dataset',
                                   '10_drill_scenes.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset

    @staticmethod
    def make_default_caterpillar():
        """
        Makes a default SpartanDatase from the 10_scenes_drill data
        :return:
        :rtype:
        """
        config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                   'dataset', 'composite',
                                   'caterpillar_only.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        dataset = SpartanDataset(mode="train", config=config)
        return dataset
