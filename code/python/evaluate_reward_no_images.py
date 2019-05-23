import argparse
import os
import sys
import numpy as np
from os.path import join
import time
import imageio

import matplotlib.pyplot as plt
from ipdb import set_trace

from plot_utils import plot_multiple_mean, concat_frames

OUTPUT_FOLDER = 'reward_plots__visual'
INPUT_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/reward_evaluation/mugs'
TCN_MODEL_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mugs/trained_models/tcn-no-depth-sv/ltcn-epoch-8.pk'
FEAT_DIM = 16

def main(args):
  set_trace()
  output_folder = join(INPUT_PATH, args.output_path)
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  input_folder = join(INPUT_PATH, args.target_path)
        output_normalized, output_unnormalized = tcn(torch.Tensor(resized_image).cuda())
        embeddings[i, :] = output_unnormalized.detach().cpu().numpy()
        embeddings_normalized[i, :] = output_normalized.detach().cpu().numpy()
        
  query_embs = np.load(join(INPUT_PATH, args.query_path))
  query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual.npy'))
  query_embs_vis = query_embs_vis_raw.flatten()

  sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_vis = apply_feature_selection_episode(query_embs_vis_raw, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)
  query_embs = np.concatenate([query_embs, query_embs_vis], axis=1)  
  
  all_rewards = []
  all_labels = []
  input_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    if not file.endswith('emb.npy'):
      continue
    # if args.query_path.split('/')[-1] == file:
    #   continue

    target_embs = np.load(join(input_folder, file))
    target_embs_vis_raw = np.load(join(input_folder, file.strip('.npy') + '_visual.npy'))
    # target_embs_vis = apply_feature_selection_episode(target_embs_vis_raw, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)
    target_emb_vis = target_embs_vis_raw.flatten()
    target_embs = np.concatenate([target_embs, target_embs_vis], axis=1)
    all_rewards.append(MakeRewardPlot(output_folder, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0.0))
    all_labels.append(file)
    concat_frames_and_rewards( join(input_folder, file.split('emb')[0] + 'video_sample.mp4'), \
                                  all_rewards[ii], output_folder)
    ii += 1
  plot_multiple_mean(np.asarray(all_rewards)[:,:], output_folder, all_labels, name='rewards_plot')

def MakeRewardPlot(outdir, name, query_embs, target_embs, nearest=False, centroid_weighting=0.01):
  if nearest:
    knn_indices = [KNNIds(q, target_embs, k=1)[0] for q in query_embs]
    assert knn_indices
    rewards = []
    knn_embs = [target_embs[k] for k in knn_indices]
    for i in range(query_embs.shape[0]):
      target_emb = knn_embs[i]
      query_emb = query_embs[i]
      # knn_embs[i][:3] *= 0.0001
      target_emb[:3] *= centroid_weighting
      query_emb[:3] *= centroid_weighting
      target_emb_norm = target_emb / np.linalg.norm(target_emb)
      query_embs_norm = query_emb / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_embs_norm, target_emb_norm))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.set_ylim([-1.0, 0.05])

    plt.plot(rewards)
    fig.savefig(os.path.join(outdir, name + '.png'))
    plt.close(fig)
  else:
    rewards = []
    for i in range(query_embs.shape[0]):
      target_emb = target_embs[i]
      query_emb = query_embs[i]
      # knn_embs[i][:3] *= 0.0001
      target_emb[:3] *= centroid_weighting
      query_emb[:3] *= centroid_weighting
      target_emb_norm = target_emb / np.linalg.norm(target_emb)
      query_embs_norm = query_emb / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_embs_norm, target_emb_norm))
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_ylim([-1.0, 0.05])    
    plt.plot(rewards)
    fig.savefig(os.path.join(outdir, name + '.png'))
    plt.close(fig)
  return rewards

def concat_frames_and_rewards(video_file, rewards, outdir):
  reader = imageio.get_reader(video_file)

  fig = plt.figure(figsize=(25, 5))
  ax = fig.add_subplot(211)
  ax2 = fig.add_subplot(212)  
  for i, img in enumerate(reader):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=1)
  ax.imshow(concat_img)
  ax.axis('tight')
  # ax.axis('off')
  # plt.subplot(212)
  ax2.plot(rewards, linewidth=3.0)
  plt.tick_params(labelsize=5)
  ax2.set_ylim([-1.0, 0.05])
  # ax2.autoscale(False)
  fig.subplots_adjust(hspace=0, wspace=0)
  ax.set_axis_off()
  fig.savefig(join(outdir, (video_file.split('/')[-1]).split('.mp4')[0] + '_rewards.jpg'))
  # plt.show()
  # set_trace()

def KNNIds(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  sorted_distances = KNNIdsWithDistances(query_vec, target_seq, k)
  return [i[0] for i in sorted_distances]

def reward_fn(x, y):
  return -np.linalg.norm(np.abs(x) - np.abs(y))


def KNNIdsWithDistances(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  if not isinstance(np.array(target_seq), np.ndarray):
    target_seq = np.array(target_seq)
  assert np.shape(query_vec) == np.shape(target_seq[0])
  distances = [(i, np.linalg.norm(query_vec-target_vec)) for (
      i, target_vec) in enumerate(target_seq)]
  sorted_distances = sorted(distances, key=lambda x: x[1])
  return sorted_distances[:k]

