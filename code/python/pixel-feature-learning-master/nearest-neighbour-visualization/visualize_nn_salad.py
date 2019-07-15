import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
from PIL import Image
import json
import cv2
from scipy.spatial import cKDTree
import time
import pickle
import math
from sklearn import decomposition
#from utils.loss import savePickle, readPickle
import sys

TESTING = True
SAVING_VAL = False
EVA_VAL_ACC = False
EVA_MAJOR_ACC = False
MERGE_PICKLE = False
shard = 0
interval = 20
bigend = 500

start_id = shard * interval
end_id = min((shard+1) * interval, bigend)

ENLARGE_RATE = 6
MAJORITY_THRESHOLD = 28
PERCENT = 3
# save_dir = '../temp'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

source_num = 0
source_h = 0
source_w = 0
source_dim = 0

target_h = 0
target_w = 0
target_dim = 0
target_num = 0
target_tree = None
source_tree = None
source_feature = None
target_feature = None
source_names = None
target_names = None
target_feature_raw = None
source_labels = None
target_labels = None
target_reshaped = None
source_reshaped = None
#loading dataset features of two. build tree for cityscape.
"""
id_to_name = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "light",
        "sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motocycle",
        "bicycle"]
name_to_id = {name : ind for (ind, name) in enumerate(id_to_name)}
"""

def saveOrLoadPCA(src_dir, dim=128):
    global target_dim, source_dim
    picklepath = os.path.join(src_dir, 'pca_%d.pkl'%(dim))
    if os.path.exists(picklepath):
        print('loading pca.')
        pca = pickle.load(open(picklepath, 'rb'))
        #pca = readPickle(picklepath)
        return pca
    print('building pca.')
    PCA_sample_num = 100000
    target_dim = target_feature.shape[-1]
    source_dim = source_feature.shape[-1]
    target_reshaped = target_feature.reshape((-1, target_dim))
    source_reshaped = source_feature.reshape((-1, source_dim))

    target_list = range(target_reshaped.shape[0])
    source_list = range(source_reshaped.shape[0])

    pca_data = np.concatenate([target_reshaped[target_list, ...], source_reshaped[source_list, ...]], axis=0)
    pca = decomposition.PCA(n_components=dim)
    pca.fit(pca_data)
    pickle.dump(pca, open(picklepath, 'wb'))
    #savePickle(picklepath, pca)
    return pca

def loadDataBuildTree():
    global target_feature_raw, source_feature, source_names, target_names, target_feature
    global target_num, target_h, target_w, target_dim
    global source_num, source_h, source_w, source_dim
    global target_tree, source_tree, source_labels, target_labels, target_reshaped, source_reshaped

    src_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/tar'
    tar_dir = '/home/zhouxian/git/pixel-feature-learning/data/NN/src'

    #source_labels = np.load(os.path.join(src_dir, 'labels_syn.npy'))
    #target_labels = np.load(os.path.join(tar_dir, 'labels_city.npy'))
    source_reshaped_path = os.path.join(src_dir, 'features_proj128.npy')#'features_syn_proj128.npy')
    target_reshaped_path = os.path.join(tar_dir, 'features_proj128.npy')#'features_city_proj128.npy')
    with open(os.path.join(src_dir, 'image_files.pkl'), 'rb') as f:
        source_names = pickle.load(f)
    with open(os.path.join(tar_dir, 'image_files.pkl'), 'rb') as f:
        target_names = pickle.load(f)

    # if os.path.exists(source_reshaped_path):
    #     print('loading source projected 128 features')
    #     source_reshaped = np.load(source_reshaped_path)
    #     source_num, source_h, source_w, source_dim = (36864, 7, 7, 128)
    # else:
    #     source_reshaped = None
    source_reshaped = None

    # if os.path.exists(target_reshaped_path):
    #     print('loading tar projected 128 features')
    #     target_reshaped = np.load(target_reshaped_path)
    #     target_num, target_h, target_w, target_dim = (4096, 7, 7, 128)
    #     #if 'val' in tar_dir:
    #     # target_num, target_h, target_w, target_dim = (500, 16, 32, 128)
    #     #else:
    #     # target_num, target_h, target_w, target_dim = (2975, 16, 32, 128)
    # else:
    target_reshaped = None

    if target_reshaped is None:
        #import ipdb; ipdb.set_trace()
        print('reading target features...')
        target_feature_raw = np.load(os.path.join(tar_dir, 'features.npy'))
        print('using all target imgs. as testing.')
        target_feature = target_feature_raw#[:10, ...]
        if source_feature is None:
            source_feature = np.load(os.path.join(src_dir, 'features.npy'))
        # pca = saveOrLoadPCA(src_dir, dim=32)
        target_num, target_h, target_w, target_dim = target_feature.shape
        target_reshaped = target_feature.reshape((-1, target_dim))

        # target_reshaped = pca.transform(target_reshaped)
        # target_norm = np.linalg.norm(target_reshaped, ord=2, axis=1, keepdims=True)
        # target_reshaped /= target_norm
        target_dim = 32
        # np.save(target_reshaped_path, target_reshaped)

    if source_reshaped is None:
        if source_feature is None:
            print('reading source features...')
            source_feature = np.load(os.path.join(src_dir, 'features.npy'))
        # pca = saveOrLoadPCA(src_dir)
        #target_reshaped = target_feature.reshape((-1, target_dim))
        source_num, source_h, source_w, source_dim = source_feature.shape

        source_reshaped = source_feature.reshape((-1, source_dim))
        # source_reshaped = pca.transform(source_reshaped)

        # source_norm = np.linalg.norm(source_reshaped, ord=2, axis=1, keepdims=True)
        # source_reshaped /= source_norm

        source_dim = 32
        # np.save(source_reshaped_path, source_reshaped)

    target_feature = target_reshaped.reshape((target_num, target_h, target_w, target_dim))
    source_feature = source_reshaped.reshape((source_num, source_h, source_w, source_dim))

    #import ipdb; ipdb.set_trace()
    start = time.time()
    print('start building kd tree.', start)
    if TESTING or SAVING_VAL:
        source_tree = cKDTree(source_reshaped)
    else:
        target_tree = cKDTree(target_reshaped)
    print('buidling done, %f s', time.time() - start)

#from downsampled image coord of source, find nearest neighbor on target, give downsampled coordinate.

def findNearestNeighborFromSource(id, x, y, return_dd=False, k=30):
    dd, ii = source_tree.query(target_feature[id, y, x, :], k=k)
    ids, ys, xs =  np.unravel_index(ii, (source_num, source_h, source_w))
    if return_dd:
        return ids, xs, ys, dd
    else:
        return ids, xs, ys

def findNearestNeighborFromTarget(id, x, y):
    dd, ii = target_tree.query(source_feature[id, y, x, :], k=30)
    ids, ys, xs =  np.unravel_index(ii, (target_num, target_h, target_w))
    return ids, xs, ys

def fromDownsampledCoordToImagePatch(source, id, x, y):


    #print('find index for x, y ', (x, y))
    #import ipdb; ipdb.set_trace()
    actual_w = source_w * int(224/7)
    actual_h = source_h * int(224/7)
    #each patch is 32 x 16, but crop 6*8 x 6*8 for context.
    if source:
        img_x = math.floor(float(x+0.5) / source_w * (actual_w))
        img_y = math.floor(float(y+0.5) / source_h * (actual_h))
        img_x_left = max(0, img_x - ENLARGE_RATE * 8)
        img_x_right = min(actual_w, img_x + ENLARGE_RATE * 8)
        img_y_up = max(0, img_y - ENLARGE_RATE * 8)
        img_y_down = min(actual_h, img_y + ENLARGE_RATE * 8)
        #print('cropping at src ', (img_x_left, img_y_up, img_x_right, img_y_down))
        return Image.open(source_names[id]).resize((224, 224)) \
                .crop((img_x_left, img_y_up, img_x_right, img_y_down))
    else:
        img_x = math.floor(float(x+0.5) / target_w * (actual_w))
        img_y = math.floor(float(y+0.5) / target_h * (actual_h))
        img_x_left = max(0, img_x - ENLARGE_RATE * 8)
        img_x_right = min(actual_w, img_x + ENLARGE_RATE * 8)
        img_y_up = max(0, img_y - ENLARGE_RATE * 8)
        img_y_down = min(actual_h, img_y + ENLARGE_RATE * 8)
        #print('cropping at tar ', (img_x_left, img_y_up, img_x_right, img_y_down))
        return Image.open(target_names[id]).resize((224, 224)) \
                .crop((img_x_left, img_y_up, img_x_right, img_y_down))

def fromDownsampledCoordToSrcLabels(ids, xs, ys):
    labels = []
    names = []
    for id, x, y in zip(ids, xs, ys):
        label = source_labels[id, y, x]
        labels.append(label)
        names.append(id_to_name[label])
    unique, unique_counts = np.unique(np.array(labels), return_counts=True)

    dist = []

    for label, count in zip(unique, unique_counts):
        dist.append((count, id_to_name[label]))

    dist.sort(key= lambda x: x[0], reverse=True)

    return labels, names, dist

#if TESTING or SAVING_VAL:
loadDataBuildTree()


if TESTING:
    for save_id in range(1):

        # tar_id = random.randint(0, target_num-1)
        # tar_x = random.randint(0, target_w-1)
        # tar_y = random.randint(0, target_h-1)
        tar_id = 1
        # tar_x = 149
        # tar_y = 206
        tar_x = 533
        tar_y = 335
        #import ipdb; ipdb.set_trace()
        #print('tar label:', id_to_name[target_labels[tar_id, tar_y, tar_x]])
        im_bgr = cv2.cvtColor(np.array(Image.open(target_names[tar_id]).convert('RGB').resize((640, 480), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)
        cv2.circle(im_bgr, (tar_x, tar_y), 5, (255,0,0), -1)
        cv2.imshow('tar', im_bgr)
        k = cv2.waitKey(1)
        
        src_ids, src_xs, src_ys = findNearestNeighborFromSource(tar_id, tar_x/8, tar_y/8, k=4)
        window_id = 1
        for src_id, x, y in zip(src_ids, src_xs, src_ys):
            im_bgr = cv2.cvtColor(np.array(Image.open(source_names[src_id]).convert('RGB').resize((640, 480), Image.ANTIALIAS)), cv2.COLOR_RGB2BGR)
            cv2.circle(im_bgr, (x*8+4, y*8+4), 5, (255,0,0), -1)
            cv2.imshow(str(window_id), im_bgr)
            k = cv2.waitKey(1)
            window_id += 1
        cv2.waitKey(0)

        #labels, names, dist = fromDownsampledCoordToSrcLabels(src_ids, src_xs, src_ys)
        #print(dist)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # for neighbor_id, (src_id, src_x, src_y) in enumerate(zip(src_ids, src_xs, src_ys)):
        #     patch = fromDownsampledCoordToImagePatch(True, src_id, src_x, src_y)
        #     patch.save(osp.join(save_dir, str(save_id)+'_' + str(neighbor_id) + '_s.png'))


        # patch = fromDownsampledCoordToImagePatch(False, tar_id, tar_x, tar_y)
        # patch.save(osp.join(save_dir, str(save_id)+'_t.png'))

        # if save_id % 1 == 0:
        #     print('saved %d pairs at %s' % (save_id, save_dir))


elif SAVING_VAL:
    if sys.argv[1] is not None:
        shard = int(sys.argv[1])
        print('sharding at %d from commoandline.', shard)
        start_id = shard * interval
        end_id = min((shard+1) * interval, bigend)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('will save at ', save_dir)
    tarid_to_neighbors = {}

    for tar_id in range(start_id, end_id):
        if tar_id % 10 == 0:
            print('calculating %d val images.[range %d %d]' % (tar_id, start_id, end_id))

        tarxs = {}
        for tar_x in range(32):
            #if tar_x == 1:
            #	break
            tarys = {}
            print_y = random.randint(0, 15)
            for tar_y in range(16):
                src_ids, src_xs, src_ys, src_dd = findNearestNeighborFromSource(tar_id, tar_x, tar_y, return_dd=True)

                _, _, dist = fromDownsampledCoordToSrcLabels(src_ids, src_xs, src_ys)
                if tar_x % 5 == 0 and tar_y == print_y:
                    print(dist)
                    print('[%d %d sd %d] tar label:' % (start_id, end_id, shard),
                            id_to_name[target_labels[tar_id, tar_y, tar_x]],
                            'at id %d x %d y: %d' % (tar_id, tar_x, tar_y))

                    tarys[tar_y] = (dist, src_ids, src_xs, src_ys, src_dd)
                #import ipdb; ipdb.set_trace()
            tarxs[tar_x] = tarys

        tarid_to_neighbors[tar_id] = tarxs
        #break #FOR DEBUGGGGGGGG

    #savePickle(osp.join(save_dir, 'neighbors_' + str(start_id) + '_' + str(end_id) + '.pkl'), tarid_to_neighbors)
    pickle.dump(tarid_to_neighbors, open(osp.join(save_dir, 'neighbors_' + str(start_id) + '_' + str(end_id) + '.pkl'), 'wb'))

    print('saving done. at ', save_dir)

elif EVA_VAL_ACC:
    loadpath = osp.join(save_dir, 'neighbors_' + str(0) + '_' + str(bigend) + '.pkl')
    print('loading NN results at ', loadpath)

    #tarid_to_neighbors = readPickle(loadpath)

    EPS = 1e-6
    cor_counter = np.zeros(15, np.float32)
    total_counter = np.zeros(15, np.float32)
    pred_counter = np.zeros(15, np.float32)
    start_time = time.time()
    for tar_id in range(0, bigend):
        if tar_id % 50 == 0:
            now_time = time.time()
            print('start processing %d val images.[range %d %d major %d] took %f seconds' \
                    % (tar_id, 0, bigend, MAJORITY_THRESHOLD, now_time - start_time))
            start_time = now_time
        for tar_x in range(32):
            print_y = random.randint(0, 15)
            for tar_y in range(16):
                true_label = target_labels[tar_id, tar_y, tar_x]
                total_counter[true_label] += 1

                dist = tarid_to_neighbors[tar_id][tar_x][tar_y][0]
                major_num = dist[0][0]
                pred_label = name_to_id[dist[0][1]]

                if major_num >= MAJORITY_THRESHOLD:
                    pred_counter[pred_label] += 1
                    if pred_label == true_label:
                        cor_counter[true_label] += 1

        if tar_id % 100 == 0 or tar_id == bigend-1:
            print('calculating %d val images.[range %d %d]' % (tar_id, start_id, end_id))

            recall = cor_counter / (total_counter + EPS)
            acc = cor_counter / (pred_counter + EPS)

            print('progress %d / %d for threhold %d' %(tar_id, bigend, MAJORITY_THRESHOLD))
            for i in range(15):
                print('class %s - acc: %.4f recall: %.4f cor: %.0f pred: %.0f total %.0f' \
                        % (id_to_name[i], acc[i], recall[i], cor_counter[i], pred_counter[i], total_counter[i]))

elif EVA_MAJOR_ACC:
    global target_reshaped
    loadpath = osp.join(save_dir, 'neighbors_' + str(0) + '_' + str(bigend) + '.pkl')
    print('loading NN results at ', loadpath)

    #tarid_to_neighbors = readPickle(loadpath)

    EPS = 1e-6

    pred_collect = {}
    def getEntropyfromDist(dist):
        entropy = 0.
        for count, name in dist:
            prob = (count / 30.)
            entropy += - math.log(prob) * prob
        return entropy
    total_counter = np.zeros(15, np.float32)

    start_time = time.time()
    for tar_id in range(0, bigend):
        if tar_id % 50 == 0:
            now_time = time.time()
            print('start processing %d val images.[range %d %d percent %d] took %f seconds' \
                    % (tar_id, 0, bigend, PERCENT, now_time - start_time))
            start_time = now_time
        for tar_x in range(32):
            print_y = random.randint(0, 15)
            for tar_y in range(16):
                true_label = target_labels[tar_id, tar_y, tar_x]
                total_counter[true_label] += 1

                dist = tarid_to_neighbors[tar_id][tar_x][tar_y][0]
                major_num = dist[0][0]
                pred_label = name_to_id[dist[0][1]]
                if len(dist) > 1:
                    if dist[1][0] == major_num:
                        continue #give up tied pixel NN
                if true_label == 14:
                    continue #invalid region
                class_collect = pred_collect.get(pred_label, [])
                entropy = getEntropyfromDist(dist)
                class_collect.append((entropy, true_label,
                    tar_id, tar_x, tar_y))
                pred_collect[pred_label] = class_collect
                #pred_counter[pred_label] += 1

                #if pred_label == true_label:
                #	cor_counter[true_label] += 1

        if tar_id % 100 == 0 or tar_id == bigend-1:
            print('calculating %d val images.[range %d %d]' % (tar_id, start_id, end_id))

            cor_counter = np.zeros(15, np.float32)

            pred_counter = np.zeros(15, np.float32)

            for pred_label, class_collect in pred_collect.items():
                class_collect.sort(key=lambda x: x[0])
                total = len(class_collect)
                keep_num = min(100, total) #total * PERCENT // 100
                #total_counter[pred_label] = keep_num

                for ind in range(keep_num):
                    true_label = class_collect[ind][1]
                    #total_counter[true_label] += 1
                    pred_counter[pred_label] += 1
                    if true_label == pred_label:
                        cor_counter[pred_label] += 1


            recall = cor_counter / (total_counter + EPS)
            acc = cor_counter / (pred_counter + EPS)

            print('progress %d / %d for percent %d' %(tar_id, bigend, PERCENT))
            for i in range(15):
                print('class %s - acc: %.4f recall: %.4f cor: %.0f pred: %.0f total %.0f' \
                        % (id_to_name[i], acc[i], recall[i], cor_counter[i], pred_counter[i], total_counter[i]))
                print('mean_acc:', acc[acc > 0].mean())
    class_collect = pred_collect[1]
    ids = np.array([x[2] for x in class_collect])
    xs = np.array([x[3] for x in class_collect])
    ys = np.array([x[4] for x in class_collect])
    target_indices = list(np.ravel_multi_index((ids, ys, xs), (target_num, target_h, target_w)))
    target_set = set(target_indices)
    candidate_set = set()
    import ipdb;ipdb.set_trace()
    _, ii = target_tree.query(target_reshaped[target_indices], k=5)
    for x in ii.reshape(-1):
        if x not in target_set:
            candidate_set.add(x)
    print('added %d candidates' % len(candidate_set))
    min_dis = 500
    official = None
    for x in candidate_set:
        dis = np.absolute(np.dot(target_reshaped[target_indices], x)).sum()
        if min_dis > dis:
            min_dis = dis
            official = x
    target_indices.append(official)
    target_set.add(official)
    id, y, x = np.unravel_index(official, (target_num, target_h, target_w))
    class_collect.append((0., target_labels[id,y,x], id, x, y))
    print('adding label: ', target_labels[id,y,x])
    import ipdb; ipdb.set_trace()

elif MERGE_PICKLE:
    #be careful, copy them first!
    tomerge = {}
    print('start merging...')
    for shard in range(25):
        print('merging shard %d', shard)
        start_id = shard * interval
        end_id = min((shard+1) * interval, bigend)
        #current = readPickle(osp.join(save_dir, 'neighbors_' + str(start_id) + '_' + str(end_id) + '.pkl'))
        for i in range(start_id, end_id):
            tomerge[i] = current[i]

    #savePickle(osp.join(save_dir, 'neighbors_' + str(0) + '_' + str(bigend) + '.pkl'), tomerge)
    print('done.')
else:
    print('NOT IMPLEMENTED OPTION...')
    print('loading VALresults and processing..')
