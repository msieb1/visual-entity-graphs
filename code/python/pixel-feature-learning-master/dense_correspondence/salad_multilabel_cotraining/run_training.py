import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
from dense_correspondence.salad_training.training import *
import sys
import logging
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#utils.set_default_cuda_visible_devices()
# utils.set_cuda_visible_devices([0]) # use this to manually set CUDA_VISIBLE_DEVICES

from dense_correspondence.salad_multilabel_cotraining.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.salad_dataset import SaladDataset
from dense_correspondence.dataset.multilabel_dataset import MultilabelDataset
logging.basicConfig(level=logging.INFO)

from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation

config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'dataset', 'salad', 'salad_multilabel.yaml')
config = utils.getDictFromYamlFilename(config_filename)

train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                               'training', 'salad_multilabel_cotraining.yaml')

train_config = utils.getDictFromYamlFilename(train_config_file)
trainset = SaladDataset(config=config, mode='train')
testset = SaladDataset(config=config, mode='test')

# for multilabel
multilabel_trainset = MultilabelDataset(config=config, split='train')
multilabel_testset = MultilabelDataset(config=config, split='test')
test_split = .2
indices = list(range(len(multilabel_trainset)))
split = int(np.floor(test_split * len(multilabel_trainset)))
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
multilabel_train_sampler = SubsetRandomSampler(train_indices)
multilabel_test_sampler = SubsetRandomSampler(test_indices)
multilabel_dataloader_train = DataLoader(multilabel_trainset, batch_size=32, sampler=multilabel_train_sampler, num_workers=train_config['training']['num_workers'])
multilabel_dataloader_test = DataLoader(multilabel_testset, batch_size=32, sampler=multilabel_test_sampler, num_workers=train_config['training']['num_workers'])



d = train_config["dense_correspondence_network"]["descriptor_dimension"] # the descriptor dimension
name = "salad_multilabel_%d" %(d)
train_config["training"]["logging_dir_name"] = name
TRAIN = True
EVALUATE = True

if TRAIN:
    print "training descriptor of dimension %d" %(d)
    train = DenseCorrespondenceTraining(dataset=trainset, dataset_test=testset, config=train_config, multilabel_collection=[multilabel_trainset, multilabel_testset, multilabel_dataloader_train, multilabel_dataloader_test, len(config['CLASS_NAMES'])])
    train.run()
    # train.run_from_pretrained('/home/zhouxian/git/pixel-feature-learning/pdc/trained_models/salad_32_backup')
    print "finished training descriptor of dimension %d" %(d)