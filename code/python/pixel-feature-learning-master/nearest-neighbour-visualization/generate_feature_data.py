import os
import glob
import pickle
import numpy as np
from PIL import Image

PRETRAIN = True

if not PRETRAIN:
    from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork


    src_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/src'
    tar_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/tar'

    src_img_paths = glob.glob(os.path.join(src_dir, '*.jpg'))
    tar_img_paths = glob.glob(os.path.join(tar_dir, '*.png'))

    model_folder = '/home/zhouxian/git/pixel-feature-learning/pdc/trained_models/salad_32'
    model_filename = model_filename='000000.pth'
    path_to_network_params = os.path.join(model_folder, model_filename)
    dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
    dataset = dcn.load_salad_training_dataset()

    # for src dir
    src_feat_array = np.zeros((len(src_img_paths), 60, 80, 32))
    for i, img_path in enumerate(src_img_paths):
        img_PIL = Image.open(img_path).convert('RGB').resize((640, 480), Image.ANTIALIAS)
        img = np.array(img_PIL)
        img_tensor = dataset.rgb_image_to_tensor(img)
        src_feat_array[i] = dcn.forward_single_image_tensor_no_upsample(img_tensor).data.cpu().numpy()
    np.save(os.path.join(src_dir, 'features.npy'), src_feat_array)
    pickle.dump(src_img_paths, open(os.path.join(src_dir, 'image_files.pkl'), 'wb' ) )

    # for tar dir
    tar_feat_array = np.zeros((len(tar_img_paths), 60, 80, 32))
    for i, img_path in enumerate(tar_img_paths):
        print i, len(tar_img_paths)
        img_PIL = Image.open(img_path).convert('RGB').resize((640, 480), Image.ANTIALIAS)
        img = np.array(img_PIL)
        img_tensor = dataset.rgb_image_to_tensor(img)
        tar_feat_array[i] = dcn.forward_single_image_tensor_no_upsample(img_tensor).data.cpu().numpy()
    np.save(os.path.join(tar_dir, 'features.npy'), tar_feat_array)
    pickle.dump(tar_img_paths, open(os.path.join(tar_dir, 'image_files.pkl'), 'wb' ) )

else:
    from dense_correspondence.network.dense_correspondence_network_pretrained import DenseCorrespondenceNetwork

    src_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/src'
    tar_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/tar'

    src_img_paths = glob.glob(os.path.join(src_dir, '*.jpg'))
    tar_img_paths = glob.glob(os.path.join(tar_dir, '*.png'))

    model_folder = '/home/zhouxian/git/pixel-feature-learning/pdc/trained_models/salad_pretrained'
    model_filename = model_filename='000000.pth'
    path_to_network_params = os.path.join(model_folder, model_filename)
    dcn = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
    dataset = dcn.load_salad_training_dataset()

    # for src dir
    src_feat_array = np.zeros((len(src_img_paths), 60, 80, 512))
    for i, img_path in enumerate(src_img_paths):
        img_PIL = Image.open(img_path).convert('RGB').resize((640, 480), Image.ANTIALIAS)
        img = np.array(img_PIL)
        img_tensor = dataset.rgb_image_to_tensor(img)
        src_feat_array[i] = dcn.forward_single_image_tensor_no_upsample(img_tensor).data.cpu().numpy()
    np.save(os.path.join(src_dir, 'features_pretrain.npy'), src_feat_array)
    pickle.dump(src_img_paths, open(os.path.join(src_dir, 'image_files_pretrain.pkl'), 'wb' ) )

    # for tar dir
    tar_feat_array = np.zeros((len(tar_img_paths), 60, 80, 512))
    for i, img_path in enumerate(tar_img_paths):
        print i, len(tar_img_paths)
        img_PIL = Image.open(img_path).convert('RGB').resize((640, 480), Image.ANTIALIAS)
        img = np.array(img_PIL)
        img_tensor = dataset.rgb_image_to_tensor(img)
        tar_feat_array[i] = dcn.forward_single_image_tensor_no_upsample(img_tensor).data.cpu().numpy()
    np.save(os.path.join(tar_dir, 'features_pretrain.npy'), tar_feat_array)
    pickle.dump(tar_img_paths, open(os.path.join(tar_dir, 'image_files_pretrain.pkl'), 'wb' ) )


