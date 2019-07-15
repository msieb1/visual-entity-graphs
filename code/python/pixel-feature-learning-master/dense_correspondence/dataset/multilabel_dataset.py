from dense_correspondence_dataset_masked import DenseCorrespondenceDataset, ImageType

import os
import cv2
import numpy as np
import logging
import glob
import random

import torch
import torch.utils.data as data
import torchvision.new_transforms.transforms as new_transforms
from PIL import Image

# note that this is the torchvision provided by the warmspringwinds
# pytorch-segmentation-detection repo. It is a fork of pytorch/vision

import dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence_manipulation.utils.utils import CameraIntrinsics

import dense_correspondence_manipulation.utils.constants as constants


utils.add_dense_correspondence_to_python_path()
import dense_correspondence.correspondence_tools.correspondence_finder as correspondence_finder
import dense_correspondence.correspondence_tools.correspondence_augmentation as correspondence_augmentation

import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


class MultilabelDataset(data.Dataset):
    def __init__(self, config, split):
        data_dir = config['multilabel_dir']
        self.CLASS_NAMES = config['CLASS_NAMES']
        self.image_names = np.loadtxt(data_dir + "/ing_list/" + self.CLASS_NAMES[0] + ".txt").astype(np.int)[:, 0]
        self.image_dir = data_dir + "/images/"
        self.labels = np.zeros((self.image_names.shape[0], len(self.CLASS_NAMES)), dtype=np.float32)
        self.weights = np.zeros((self.image_names.shape[0], len(self.CLASS_NAMES)), dtype=np.float32)
        if split == 'train':
            self.transform = new_transforms.Compose([
                new_transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                new_transforms.RandomHorizontalFlip(),
                new_transforms.ToTensor(),
                new_transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
            ])
        elif split == 'test': 
            self.transform = new_transforms.Compose([
            new_transforms.Resize(224),
            new_transforms.ToTensor(),
            new_transforms.Normalize(self.get_image_mean(), self.get_image_std_dev())
        ])

        for i, c in enumerate(self.CLASS_NAMES):
            self.labels[:, i] = (np.loadtxt(data_dir + "/ing_list/" + c + ".txt") > -0.5).astype(np.int)[:, 1]
            self.weights[:, i] = (np.abs(np.loadtxt(data_dir + "/ing_list/" + c + ".txt")) > 0.5).astype(np.int)[:, 1]

        print "Using MultilabelDataset:"
        print "   - in", split, "mode"
        print "   - total images:    ", len(self.image_names)


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.image_dir + str(self.image_names[idx]) + ".jpg").convert('RGB')

        image = self.transform(image)

        return image, self.labels[idx], self.weights[idx]

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

def compute_ap(gt, pred, valid, average=None):
    nclasses = gt.shape[1]
    AP = []
    for cid in range(nclasses):
        gt_cls = gt[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls = pred[:, cid][valid[:, cid] > 0].astype('float32')
        pred_cls -= 1e-5 * gt_cls
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=average)
        AP.append(ap)
    return AP


def eval_dataset_map(model, dataset, device):
    model.eval()
    for batch, (images, labels, weights) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        output = model(images)
        if batch == 0:
            gt = labels.cpu().detach().numpy()
            pred = output.cpu().detach().numpy()
            valid = weights.cpu().detach().numpy()
        else:
            gt = np.append(gt, labels.cpu().detach().numpy(), axis=0)
            pred = np.append(pred, output.cpu().detach().numpy(), axis=0)
            valid = np.append(valid, weights.cpu().detach().numpy(), axis=0)

    AP = compute_ap(gt, pred, valid)
    return AP, np.mean(AP)