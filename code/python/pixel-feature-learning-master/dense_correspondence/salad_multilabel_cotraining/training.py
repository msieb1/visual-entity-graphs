# system
import numpy as np
import sys
import os
import fnmatch
import gc
import logging
import time
import shutil
import subprocess
import copy
from datetime import timedelta

# torch
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import visdom
import tensorboard_logger
from torchnet.logger import VisdomPlotLogger, VisdomLogger



# dense correspondence
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from pytorch_segmentation_detection.transforms import (ComposeJoint,
                                                       RandomHorizontalFlipJoint,
                                                       RandomScaleJoint,
                                                       CropOrPad,
                                                       ResizeAspectRatioPreserve,
                                                       RandomCropJoint,
                                                       Split2D)

from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, SpartanDatasetDataType
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork

from dense_correspondence.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import dense_correspondence.loss_functions.loss_composer as loss_composer
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation


class DenseCorrespondenceTraining(object):

    def __init__(self, config=None, dataset=None, dataset_test=None, multilabel_collection=None):
        if config is None:
            config = DenseCorrespondenceTraining.load_default_config()

        self._config = config
        self._dataset = dataset
        self._dataset_test = dataset_test

        self._dcn = None
        self._optimizer = None

        self.multilabel_trainset, self.multilabel_testset, self.multilabel_dataloader_train, self.multilabel_dataloader_test, self.num_classes = multilabel_collection

        self.multilabel_train_iterator = iter(self.multilabel_dataloader_train)
        self.multilabel_test_iterator = iter(self.multilabel_dataloader_test)
        self.multilabel_criterion = nn.BCELoss()
        self.multilabel_train_batch_i = 0

    def setup(self):
        self.load_dataset()
        self.setup_logging_dir()
        #self.setup_visdom()
        self.setup_tensorboard()

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def load_dataset(self):
        """
        Loads a dataset, construct a trainloader.
        Additionally creates a dataset and DataLoader for the test data
        :return:
        :rtype:
        """

        batch_size = self._config['training']['batch_size']
        num_workers = self._config['training']['num_workers']
        logging.info("Number of data loading workers: %d" % (num_workers))
        
        self._dataset.set_parameters_from_training_config(self._config)

        self._data_loader = torch.utils.data.DataLoader(self._dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

        # create a test dataset
        if self._config["training"]["compute_test_loss"]:
            self._dataset_test.set_parameters_from_training_config(self._config)
            self._data_loader_test = torch.utils.data.DataLoader(self._dataset_test, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)

    def load_dataset_from_config(self, config):
        """
        Loads train and test datasets from the given config
        :param config: Dict gotten from a YAML file
        :type config:
        :return: None
        :rtype:
        """
        self._dataset = SpartanDataset(mode="train", config=config)
        self._dataset_test = SpartanDataset(mode="test", config=config)
        self.load_dataset()

    def build_network(self):
        """
        Builds the DenseCorrespondenceNetwork
        :return:
        :rtype: DenseCorrespondenceNetwork
        """
        return DenseCorrespondenceNetwork.from_config(self._config['dense_correspondence_network'], load_stored_params=False, cotrain=True, cotrain_num_classes=self.num_classes)

    def _construct_optimizer(self, parameters):
        """
        Constructs the optimizer
        :param parameters: Parameters to adjust in the optimizer
        :type parameters:
        :return: Adam Optimizer with params from the config
        :rtype: torch.optim
        """

        learning_rate = float(self._config['training']['learning_rate'])
        weight_decay = float(self._config['training']['weight_decay'])
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _get_current_loss(self, logging_dict):
        """
        Gets the current loss for both test and train
        :return:
        :rtype: dict
        """
        d = dict()
        d['train'] = dict()
        d['test'] = dict()

        for key, val in d.iteritems():
            for field in logging_dict[key].keys():
                vec = logging_dict[key][field]

                if len(vec) > 0:
                    val[field] = vec[-1]
                else:
                    val[field] = -1 # placeholder


        return d

    def load_pretrained(self, model_folder, iteration=None):
        """
        Loads network and optimizer parameters from a previous training run.

        Note: It is up to the user to ensure that the model parameters match.
        e.g. width, height, descriptor dimension etc.

        :param model_folder: location of the folder containing the param files 001000.pth. Can be absolute or relative path. If relative then it is relative to pdc/trained_models/
        :type model_folder:
        :param iteration: which index to use, e.g. 3500, if None it loads the latest one
        :type iteration:
        :return: iteration
        :rtype:
        """

        if not os.path.isdir(model_folder):
            pdc_path = utils.getPdcPath()
            model_folder = os.path.join(pdc_path, "trained_models", model_folder)

        # find idx.pth and idx.pth.opt files
        if iteration is None:
            files = os.listdir(model_folder)
            model_param_file = sorted(fnmatch.filter(files, '*.pth'))[-1]
            iteration = int(model_param_file.split(".")[0])
            optim_param_file = sorted(fnmatch.filter(files, '*.pth.opt'))[-1]
        else:
            prefix = utils.getPaddedString(iteration, width=6)
            model_param_file = prefix + ".pth"
            optim_param_file = prefix + ".pth.opt"

        print "model_param_file", model_param_file
        model_param_file = os.path.join(model_folder, model_param_file)
        optim_param_file = os.path.join(model_folder, optim_param_file)


        self._dcn = self.build_network()
        self._dcn.load_state_dict(torch.load(model_param_file))
        self._dcn.cuda()
        self._dcn.train()

        self._optimizer = self._construct_optimizer(self._dcn.parameters())
        self._optimizer.load_state_dict(torch.load(optim_param_file))

        return iteration

    def run_from_pretrained(self, model_folder, iteration=None, learning_rate=None):
        """
        Wrapper for load_pretrained(), then run()
        """
        iteration = self.load_pretrained(model_folder, iteration)
        if iteration is None:
            iteration = 0

        if learning_rate is not None:
            self._config["training"]["learning_rate_starting_from_pretrained"] = learning_rate
            self.set_learning_rate(self._optimizer, learning_rate)

        self.run(loss_current_iteration=iteration, use_pretrained=True)

    def run(self, loss_current_iteration=0, use_pretrained=False):
        """
        Runs the training
        :return:
        :rtype:
        """

        start_iteration = copy.copy(loss_current_iteration)

        DCE = DenseCorrespondenceEvaluation

        self.setup()
        self.save_configs()

        if not use_pretrained:
            # create new network and optimizer
            self._dcn = self.build_network()
            self._optimizer = self._construct_optimizer(self._dcn.parameters())
        else:
            logging.info("using pretrained model")
            if (self._dcn is None):
                raise ValueError("you must set self._dcn if use_pretrained=True")
            if (self._optimizer is None):
                raise ValueError("you must set self._optimizer if use_pretrained=True")

        # make sure network is using cuda and is in train mode
        dcn = self._dcn
        dcn.cuda()
        dcn.train()

        optimizer = self._optimizer
        batch_size = self._data_loader.batch_size

        pixelwise_contrastive_loss = PixelwiseContrastiveLoss(image_shape=dcn.image_shape, config=self._config['loss_function'])
        pixelwise_contrastive_loss.debug = True

        # Repeat M for background and masked
        pixelwise_contrastive_loss._config['M_background'] = pixelwise_contrastive_loss._config['M_descriptor']
        pixelwise_contrastive_loss._config['M_masked'] = pixelwise_contrastive_loss._config['M_descriptor']

        loss = match_loss = non_match_loss = 0

        num_epochs = self._config['training']['num_epochs']
        logging_rate = self._config['training']['logging_rate']
        save_rate = self._config['training']['save_rate']
        compute_test_loss_rate = self._config['training']['compute_test_loss_rate']

        # logging
        self._logging_dict = dict()
        self._logging_dict['train'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "masked_non_match_loss": [], 
                                           "background_non_match_loss": [],
                                           "blind_non_match_loss": [],
                                           "learning_rate": [],
                                           "different_object_non_match_loss": []}

        self._logging_dict['test'] = {"iteration": [], "loss": [], "match_loss": [],
                                           "non_match_loss": []}

        # save network before starting
        if not use_pretrained:
            self.save_network(dcn, optimizer, 0)

        t_start = time.time()
        loss_vec = []
        match_loss_vec = []
        non_match_loss_vec = []
        multilabel_loss_vec = []
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            for i, data in enumerate(self._data_loader, 0):
                loss_current_iteration += 1
                start_iter = time.time()

                ############### pixel correspondence ###############
                match_type, img_a, img_b, matches_a, matches_b, non_matches_a, non_matches_b = data
                
                img_a = Variable(img_a.cuda(), requires_grad=False)
                img_b = Variable(img_b.cuda(), requires_grad=False)

                # Note: repeat non_matches for both masked and background, and fake blind nonmatches using empty tensor, for compatibility in loss computation
                matches_a = Variable(matches_a.cuda().squeeze(0), requires_grad=False)
                matches_b = Variable(matches_b.cuda().squeeze(0), requires_grad=False)
                non_matches_a = Variable(non_matches_a.cuda().squeeze(0), requires_grad=False)
                non_matches_b = Variable(non_matches_b.cuda().squeeze(0), requires_grad=False)
                blind_non_matches_a = Variable(SpartanDataset.empty_tensor().cuda().squeeze(0), requires_grad=False)
                blind_non_matches_b = Variable(SpartanDataset.empty_tensor().cuda().squeeze(0), requires_grad=False)

                optimizer.zero_grad()
                self.adjust_learning_rate(optimizer, loss_current_iteration)

                # run both images through the network
                image_a_pred = dcn.forward(img_a)
                image_a_pred = dcn.process_network_output(image_a_pred, batch_size)

                image_b_pred = dcn.forward(img_b)
                image_b_pred = dcn.process_network_output(image_b_pred, batch_size)

                # get loss.
                loss, match_loss, non_match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss \
                    = loss_composer.get_loss(pixelwise_contrastive_loss, match_type,
                                            image_a_pred, image_b_pred,
                                            matches_a,     matches_b,
                                            non_matches_a, non_matches_b,
                                            non_matches_a, non_matches_b,
                                            blind_non_matches_a, blind_non_matches_b)


                ############### multilabel ###############
                try:
                    imgs, labels, weights = next(self.multilabel_train_iterator)
                    self.multilabel_train_batch_i += 1
                except StopIteration:
                    self.multilabel_train_iterator = iter(self.multilabel_dataloader_train)
                    self.multilabel_train_batch_i = 1
                    imgs, labels, weights  = next(self.multilabel_train_iterator)
                imgs = Variable(imgs.cuda(), requires_grad=False)
                labels = Variable(labels.cuda(), requires_grad=False)

                outputs = torch.sigmoid(dcn.forward_multilabel(imgs))
                multilabel_loss = self.multilabel_criterion(outputs, labels)


                ############# combine together #############
                loss_weight = {'pixel':1.0, 'multilabel':3.0}
                total_loss = loss*loss_weight['pixel'] + multilabel_loss*loss_weight['multilabel']

                total_loss.backward()
                optimizer.step()
                elapsed = time.time() - start_iter

                # print "single iteration took %.3f seconds" %(elapsed)

                if loss_current_iteration % save_rate == 0:
                    self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)

                sys.stdout.write('Epoch %d/%d, total_itr: %d, \n \t -- Pixel Correspondence: image %d/%d, loss: %.4f, match_loss: %.4f, non_match_loss: %.4f, total_time: %s \n \t -- Multilabel classification: batch %d/%d, loss: %.4f \033[F\033[F' % \
                    (epoch+1, num_epochs, loss_current_iteration, i+1, len(self._dataset), loss.data[0],  match_loss.data[0], non_match_loss.data[0], str(timedelta(seconds=time.time()-t_start))[:-4], self.multilabel_train_batch_i, len(self.multilabel_dataloader_train), multilabel_loss.data[0])); sys.stdout.flush()

                loss_vec.append(loss.data[0])
                match_loss_vec.append(match_loss.data[0])
                non_match_loss_vec.append(non_match_loss.data[0])
                multilabel_loss_vec.append(multilabel_loss.data[0])

                if self._config["training"]["compute_test_loss"] and (loss_current_iteration % compute_test_loss_rate == 0):
                    print
                    # logging.info("Computing test loss")

                    # delete the loss, match_loss, non_match_loss variables so that
                    # pytorch can use that GPU memory
                    del loss, match_loss, non_match_loss, masked_non_match_loss, background_non_match_loss, blind_non_match_loss, multilabel_loss
                    gc.collect()
                    print '\n'
                    print '\tTraining average: pixel_loss: %.4f, match_loss: %.4f, non_match_loss: %.4f, multilabel_loss: %.4f' % \
                            (np.mean(loss_vec), np.mean(match_loss_vec), np.mean(non_match_loss_vec), np.mean(multilabel_loss_vec))
                    loss_vec = []
                    match_loss_vec = []
                    non_match_loss_vec = []
                    multilabel_loss_vec = []

                    if self._config['training']['batch_size'] == 1:
                        # with batch size 1, testing should use train mode since batchnorm doesn't work
                        test_loss, test_match_loss, test_non_match_loss = \
                            DCE.compute_loss_on_salad_dataset(
                                dcn, self._data_loader_test, 
                                self._config['loss_function'], 
                                num_iterations=self._config['training']['test_loss_num_iterations'],
                                evalmode=False)
                    else:
                        assert False # we only allow batch size = 1 now
                        dcn.eval()
                        test_loss, test_match_loss, test_non_match_loss = \
                            DCE.compute_loss_on_salad_dataset(
                                dcn, self._data_loader_test, 
                                self._config['loss_function'], 
                                num_iterations=self._config['training']['test_loss_num_iterations'])
                    # make sure to set the network back to train mode
                    dcn.train()

                    print '\tTesting results: loss: %.4f, match_loss: %.4f, non_match_loss: %.4f' % \
                        (test_loss,  test_match_loss, test_non_match_loss)

                    # delete these variables so we can free GPU memory
                    del test_loss, test_match_loss, test_non_match_loss


                if loss_current_iteration % self._config['training']['garbage_collect_rate'] == 0:
                    logging.debug("running garbage collection")
                    gc_start = time.time()
                    gc.collect()
                    gc_elapsed = time.time() - gc_start
                    logging.debug("garbage collection took %.2d seconds" %(gc_elapsed))

        logging.info("Finished training.")
        self.save_network(dcn, optimizer, loss_current_iteration, logging_dict=self._logging_dict)
        return


    def setup_logging_dir(self):
        """
        Sets up the directory where logs will be stored and config
        files written
        :return: full path of logging dir
        :rtype: str
        """

        if 'logging_dir_name' in self._config['training']:
            dir_name = self._config['training']['logging_dir_name']
        else:
            dir_name = utils.get_current_time_unique_name() +"_" + str(self._config['dense_correspondence_network']['descriptor_dimension']) + "d"

        self._logging_dir_name = dir_name

        self._logging_dir = os.path.join(utils.convert_to_absolute_path(self._config['training']['logging_dir']), dir_name)



        if os.path.isdir(self._logging_dir):
            shutil.rmtree(self._logging_dir)

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        # make the tensorboard log directory
        self._tensorboard_log_dir = os.path.join(self._logging_dir, "tensorboard")
        if not os.path.isdir(self._tensorboard_log_dir):
            os.makedirs(self._tensorboard_log_dir)

        return self._logging_dir

    def save_network(self, dcn, optimizer, iteration, logging_dict=None):
        """
        Saves network parameters to logging directory
        :return:
        :rtype: None
        """

        network_param_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + ".pth")
        optimizer_param_file = network_param_file + ".opt"
        torch.save(dcn.state_dict(), network_param_file)
        torch.save(optimizer.state_dict(), optimizer_param_file)

        # also save loss history stuff
        if logging_dict is not None:
            log_history_file = os.path.join(self._logging_dir, utils.getPaddedString(iteration, width=6) + "_log_history.yaml")
            utils.saveToYaml(logging_dict, log_history_file)

            current_loss_file = os.path.join(self._logging_dir, 'loss.yaml')
            current_loss_data = self._get_current_loss(logging_dict)

            utils.saveToYaml(current_loss_data, current_loss_file)



    def save_configs(self):
        """
        Saves config files to the logging directory
        :return:
        :rtype: None
        """
        training_params_file = os.path.join(self._logging_dir, 'training.yaml')
        utils.saveToYaml(self._config, training_params_file)

        dataset_params_file = os.path.join(self._logging_dir, 'dataset.yaml')
        utils.saveToYaml(self._dataset.config, dataset_params_file)        

    def adjust_learning_rate(self, optimizer, iteration):
        """
        Adjusts the learning rate according to the schedule
        :param optimizer:
        :type optimizer:
        :param iteration:
        :type iteration:
        :return:
        :rtype:
        """

        steps_between_learning_rate_decay = self._config['training']['steps_between_learning_rate_decay']
        if iteration % steps_between_learning_rate_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self._config["training"]["learning_rate_decay"]

    @staticmethod
    def set_learning_rate(optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    @staticmethod
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        return lr

    def setup_visdom(self):
        """
        Sets up visdom visualizer
        :return:
        :rtype:
        """
        self.start_visdom()
        self._visdom_env = self._logging_dir_name
        self._vis = visdom.Visdom(env=self._visdom_env)

        self._port = 8097
        self._visdom_plots = dict()

        self._visdom_plots["train"] = dict()
        self._visdom_plots['train']['loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Train Loss'}, env=self._visdom_env)

        self._visdom_plots['learning_rate'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Learning Rate'}, env=self._visdom_env)

        self._visdom_plots['train']['match_loss'] = VisdomPlotLogger(
        'line', port=self._port, opts={'title': 'Train Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['masked_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Masked Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['background_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Background Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['train']['blind_non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Train Blind Non Match Loss'}, env=self._visdom_env)


        self._visdom_plots["test"] = dict()
        self._visdom_plots['test']['loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Loss'}, env=self._visdom_env)

        self._visdom_plots['test']['match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Match Loss'}, env=self._visdom_env)

        self._visdom_plots['test']['non_match_loss'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Test Non Match Loss'}, env=self._visdom_env)

        self._visdom_plots['masked_hard_negative_rate'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Masked Matches Hard Negative Rate'}, env=self._visdom_env)

        self._visdom_plots['non_masked_hard_negative_rate'] = VisdomPlotLogger(
            'line', port=self._port, opts={'title': 'Non-Masked Hard Negative Rate'}, env=self._visdom_env)

    def setup_tensorboard(self):
        """
        Starts the tensorboard server and sets up the plotting
        :return:
        :rtype:
        """

        # start tensorboard
        # cmd = "python -m tensorboard.main"
        logging.info("setting up tensorboard_logger")
        cmd = "tensorboard --logdir=%s" %(self._tensorboard_log_dir)
        self._tensorboard_logger = tensorboard_logger.Logger(self._tensorboard_log_dir)
        logging.info("tensorboard logger started")


    @staticmethod
    def load_default_config():
        dc_source_dir = utils.getDenseCorrespondenceSourceDir()
        config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence',
                                   'training', 'training.yaml')

        config = utils.getDictFromYamlFilename(config_file)
        return config

    @staticmethod
    def make_default():
        dataset = SpartanDataset.make_default_caterpillar()
        return DenseCorrespondenceTraining(dataset=dataset)


    @staticmethod
    def start_visdom():
        """
        Starts visdom if it's not already running
        :return:
        :rtype:
        """

        vis = visdom.Visdom()

        if vis.check_connection():
            logging.info("Visdom already running, returning")
            return


        logging.info("Starting visdom")
        cmd = "python -m visdom.server"
        subprocess.Popen([cmd], shell=True)
