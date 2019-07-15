import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.salad_training.training import *
import sys
import logging

#utils.set_default_cuda_visible_devices()
# utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.salad_training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.salad_dataset import SaladDataset
logging.basicConfig(level=logging.INFO)

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'dataset', 'salad', 'salad.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'training', 'salad_training.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
trainset = SaladDataset(config=config, mode='train')
testset = SaladDataset(config=config, mode='test')

d = train_config["dense_correspondence_network"]["descriptor_dimension"] # the descriptor dimension
name = "salad_%d" %(d)
train_config["training"]["logging_dir_name"] = name

TRAIN = True
EVALUATE = True

if TRAIN:
    print "training descriptor of dimension %d" %(d)
    train = DenseCorrespondenceTraining(dataset=trainset, dataset_test=testset, config=train_config)
    train.run()
    # train.run_from_pretrained('/home/zhouxian/git/pixel-feature-learning/pdc/trained_models/salad_32_backup')
    print "finished training descriptor of dimension %d" %(d)