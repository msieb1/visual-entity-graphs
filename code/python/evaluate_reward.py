import matplotlib
matplotlib.use('Agg')
import argparse
import os
import sys
import numpy as np
from os.path import join
import time
import imageio
import random
import matplotlib.pyplot as plt
from pdb import set_trace

from plot_utils import plot_multiple_mean, random_colors
random.seed(2)

sys.path.append('/home/max/projects/gps-lfd')
from config import Config
conf = Config()

# TCN_MODEL_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mugs/trained_models/tcn-no-depth-sv/ltcn-epoch-8.pk'
EXP_DIR = conf.EXP_DIR
EXP_NAME = conf.EXP_NAME
MODE = conf.MODE
MODEL_FOLDER = conf.MODEL_FOLDER
INPUT_PATH = join(EXP_DIR, EXP_NAME, "videos_features", MODEL_FOLDER, MODE)
OUTPUT_FOLDER = join(EXP_DIR, EXP_NAME, 'videos_features', MODEL_FOLDER, MODE)
if not os.path.exists(OUTPUT_FOLDER):
  os.makedirs(OUTPUT_FOLDER)

QUERY_EMB_PATH = join(INPUT_PATH, '0_view0_emb_norm.npy')
TARGETS_FOLDER = INPUT_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--target-path', type=str, default=TARGETS_FOLDER)
parser.add_argument('--query-path', type=str, default=QUERY_EMB_PATH)
parser.add_argument('--output-path', type=str, default=OUTPUT_FOLDER)


class PlotConfigToy(object):
  ALL_LABELS = ['demo', 'idle', 'distractor', 'robot correct', 'robot wrong', 'robot angled', 'bla']
  EXCLUDED_LABELS = []
  # EXCLUDED_LABELS = ['open', 'robot correct']
  # EXCLUDED_LABELS = ['fake']
  # EXCLUDED_LABELS = ['open',  'robot wrong']
  COLORS = random_colors(len(ALL_LABELS))
  START = 7
  END = 10
  MODE = 'FF'
  LOWER_BOUND_PLOT = -20
  # Y_LIM = [LOWER_BOUND_PLOT, 0.05]
  Y_LIM = None
  PLOT_SIZE = 2

class PlotConfigDuck(object):
  ALL_LABELS = [str(i) for i in range(20)]
  EXCLUDED_LABELS = []
  # EXCLUDED_LABELS = ['open', 'robot correct']
  # EXCLUDED_LABELS = ['fake']
  # EXCLUDED_LABELS = ['open',  'robot wrong']
  COLORS = random_colors(len(ALL_LABELS))
  START = 0
  END = 1000
  MODE = 'FF'
  LOWER_BOUND_PLOT = -20
  # Y_LIM = [LOWER_BOUND_PLOT, 0.05]
  Y_LIM = None
  PLOT_SIZE = 2
plot_config = PlotConfigDuck()

def main_all(args):
  colors = plot_config.COLORS

  query_embs_centroid = np.abs(np.load(join(INPUT_PATH, args.query_path)))
  if plot_config.MODE == 'INC':
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual_inc.npy'))
    sel_idx_1, sel_idx_2 = feature_selection_inc(query_embs_vis_raw)
    query_embs_vis = apply_feature_selection_episode_inc(query_embs_vis_raw, sel_idx_1, sel_idx_2)  
  elif plot_config.MODE == 'FF':
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '.npy'))
    query_embs_vis = query_embs_vis_raw
  else:
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual.npy'))
    query_embs_vis = np.reshape(query_embs_vis_raw, [10, 32*query_embs_vis_raw.shape[1]])[:, :]

  # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
  # sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)

  query_embs = np.concatenate([query_embs_centroid, query_embs_vis], axis=1)  
  query_embs = query_embs_vis

  all_rewards = []
  all_labels = []
  all_files = []
  input_files = sorted(os.listdir(INPUT_PATH), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    if not file.endswith('emb.npy'):
      continue
    # if args.query_path.split('/')[-1] == file:
    #   continue

    target_embs_centroid = np.abs(np.load(join(INPUT_PATH, file)))
    if plot_config.MODE == 'INC':
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '_visual_inc.npy'))
      target_embs_vis = apply_feature_selection_episode_inc(target_embs_vis_raw, sel_idx_1, sel_idx_2)
    elif plot_config.MODE == 'FF':
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '.npy'))
      target_embs_vis = target_embs_vis_raw
    else:
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '_visual.npy'))
      target_embs_vis = np.reshape(target_embs_vis_raw, [10, 32*query_embs_vis_raw.shape[1]])[:, :]
    
    # target_embs_roi = np.load(join(INPUT_PATH, file.strip('.npy') + '_roi.npy'))
    # target_embs_roi = apply_feature_selection_episode(target_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)


    target_embs = np.concatenate([target_embs_centroid, target_embs_vis], axis=1)
    target_embs = target_embs_vis
    all_rewards.append(MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0.0))
    all_labels.append(file)

    vid_path = join(join(('/'.join(str.split(join(INPUT_PATH, file), '/')[:-3])), 'videos', MODE), file.split('_emb')[0] + '.mp4')
    all_files.append(vid_path)
    ii += 1
  all_files = sorted(all_files, key=lambda x: int(x.split('/')[-1].split('_')[0]))
  # include_idx = [0, 4, 5, 6]
  # include_idx = [0, 1, 2, 3]


  # all_labels = ['Demo', 'Idle', 'Front' ,'Half Way', 'Robot Correct Jittery', 'Robot Wrong', 'Robot Correct']
  # all_labels = ['Demo', 'Wrong', 'Wrong', 'Wrong','Wrong', 'Correct End Pose', 'Correct Start Pose']
  all_labels = plot_config.ALL_LABELS
  ConcatFramesAndRewardsMultiple(  all_files, \
                                  all_rewards , all_labels, colors, OUTPUT_FOLDER, y_lim=plot_config.Y_LIM)
  
  # ConcatFramesAndRewardsMultiple(  all_files, \
  #                                 all_rewards , ['Demo', 'Different Ring', 'Robot Correct', 'Robot Wrong Behind', 'Robot Wrong Front'], colors, OUTPUT_FOLDER)
  
  # ConcatFramesAndRewardsMultiple(  all_files, \
  #                                 all_rewards , ['Demo', 'Different Ring', 'Other Side', 'Robot Correct', 'Robot Other Side', 'Robot Wrong Other Side', 'Robot Wrong Behind', 'Robot Wrong Front'], colors, OUTPUT_FOLDER)
  
  plot_multiple_mean(np.asarray(all_rewards)[:,:], OUTPUT_FOLDER, all_labels, name='rewards_plot', y_lim=plot_config.LOWER_BOUND_PLOT)

def main_all_no_images(args):
  colors = plot_config.COLORS

  query_embs_centroid = np.abs(np.load(join(INPUT_PATH, args.query_path)))
  if plot_config.MODE == 'INC':
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual_inc.npy'))
    sel_idx_1, sel_idx_2 = feature_selection_inc(query_embs_vis_raw)
    query_embs_vis = apply_feature_selection_episode_inc(query_embs_vis_raw, sel_idx_1, sel_idx_2)  
  elif plot_config.MODE == 'FF':
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '.npy'))
    query_embs_vis = query_embs_vis_raw
  else:
    query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual.npy'))
    query_embs_vis = np.reshape(query_embs_vis_raw, [10, 32*query_embs_vis_raw.shape[1]])[:, :]

  # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
  # sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)

  query_embs = np.concatenate([query_embs_centroid, query_embs_vis], axis=1)  
  query_embs = query_embs_vis

  all_rewards = []
  all_labels = []
  all_files = []
  input_files = sorted(os.listdir(INPUT_PATH), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    if not file.endswith('emb_norm.npy'):
      continue
    # if args.query_path.split('/')[-1] == file:
    #   continue

    target_embs_centroid = np.abs(np.load(join(INPUT_PATH, file)))
    if plot_config.MODE == 'INC':
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '_visual_inc.npy'))
      target_embs_vis = apply_feature_selection_episode_inc(target_embs_vis_raw, sel_idx_1, sel_idx_2)
    elif plot_config.MODE == 'FF':
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '.npy'))
      target_embs_vis = target_embs_vis_raw
    else:
      target_embs_vis_raw = np.load(join(INPUT_PATH, file.strip('.npy') + '_visual.npy'))
      target_embs_vis = np.reshape(target_embs_vis_raw, [10, 32*query_embs_vis_raw.shape[1]])[:, :]
    
    # target_embs_roi = np.load(join(INPUT_PATH, file.strip('.npy') + '_roi.npy'))
    # target_embs_roi = apply_feature_selection_episode(target_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)


    target_embs = np.concatenate([target_embs_centroid, target_embs_vis], axis=1)
    target_embs = target_embs_vis
    all_rewards.append(MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0.0))
    all_labels.append(file)

    vid_path = join(join(('/'.join(str.split(join(INPUT_PATH, file), '/')[:-3])), 'videos', MODE), file.split('_emb')[0] + '.mp4')
    all_files.append(vid_path)
    ii += 1
  all_files = sorted(all_files, key=lambda x: int(x.split('/')[-1].split('_')[0]))
  # include_idx = [0, 4, 5, 6]
  # include_idx = [0, 1, 2, 3]


  # all_labels = ['Demo', 'Idle', 'Front' ,'Half Way', 'Robot Correct Jittery', 'Robot Wrong', 'Robot Correct']
  # all_labels = ['Demo', 'Wrong', 'Wrong', 'Wrong','Wrong', 'Correct End Pose', 'Correct Start Pose']
  all_labels = plot_config.ALL_LABELS

  #plot_multiple_mean(np.asarray(all_rewards)[:,:], OUTPUT_FOLDER, all_labels, name='rewards_plot', y_lim=plot_config.LOWER_BOUND_PLOT)


def MakeRewardPlot(outdir, name, query_embs, target_embs, nearest=True, centroid_weighting=0.0001, plot_figure=True):

  if nearest and False:
    rewards = []
    window = 1
    for i in range(query_embs.shape[0]):
      query_emb = query_embs[i]
      if i < window:
        knn_ind = KNNIds(query_emb, target_embs[0:min(len(query_embs)-window, i+window)], k=1)[0]
        target_emb = target_embs[knn_ind + i ]
      elif i >= len(query_embs)-window:
        knn_ind = KNNIds(query_emb, target_embs[max(0, i-window):len(query_embs)-window], k=1)[0]
        target_emb = target_embs[knn_ind + i]           
      else:
        knn_ind = KNNIds(query_emb, target_embs[max(0, i-window):min(len(query_embs)-window, i+window)], k=1)[0]
        target_emb = target_embs[knn_ind + i - window]        
      # knn_embs[i][:3] *= 0.0001
      target_emb[:3] *= centroid_weighting
      query_emb[:3] *= centroid_weighting
      target_emb_norm = target_emb / np.linalg.norm(target_emb)
      query_embs_norm = query_emb / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_embs_norm, target_emb_norm))
  elif nearest:
    knn_indices = [KNNIds(q, target_embs, k=1)[0] for q in query_embs]
    assert knn_indices
    rewards = []
    knn_embs = [target_embs[k] for k in knn_indices]
    for i in range(query_embs.shape[0]):
      target_emb = knn_embs[i]
      query_emb = query_embs[i]
      # knn_embs[i][:3] *= 0.0001
      # target_emb[:3] *= centroid_weighting
      # query_emb[:3] *= centroid_weighting
      target_emb_norm = target_emb / np.linalg.norm(target_emb)
      query_embs_norm = query_emb / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_embs_norm, target_emb_norm))
  else:
    rewards = []
    for i in range(query_embs.shape[0]):
      target_emb = target_embs[i]
      query_emb = query_embs[i]
      # knn_embs[i][:3] *= 0.0001
      # target_emb[:3] *= centroid_weighting
      # query_emb[:3] *= centroid_weighting
      # target_emb[3:] *= 0
      # query_emb[3:] *= 0

      target_emb_norm = target_emb / np.linalg.norm(target_emb)
      query_embs_norm = query_emb  / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_embs_norm, target_emb_norm))
  if plot_figure:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_ylim([plot_config.LOWER_BOUND_PLOT, 0.05])    
    plt.plot(rewards)
    fig.savefig(os.path.join(outdir, name + '.png'))
    plt.close(fig)
  return rewards

def ConcatFramesAndRewardsMultiple(video_files, rewards, labels, colors, outdir, y_lim=None):
  n_vids = len([v for i, v in enumerate(video_files) if labels[i] not in plot_config.EXCLUDED_LABELS])
  n_vids =  -plot_config.START + plot_config.END
  plot_size = plot_config.PLOT_SIZE
  fig = plt.figure(figsize=(15, n_vids*1+2))
  concat_img = None
  vid_cnt = 0
  for i, vid in enumerate(video_files):
    if labels[i] in plot_config.EXCLUDED_LABELS:
      continue
    reader = imageio.get_reader(vid)
    cax = plt.subplot2grid((n_vids + plot_size, 1), (vid_cnt, 0))
    for j, img in enumerate(reader):
        if j < plot_config.START or j > plot_config.END:
          continue
        if concat_img is None:
          concat_img = img
          continue
        else:
          concat_img = np.concatenate([concat_img, img], axis=1)
    vid_cnt += 1
    cax.imshow(concat_img)
    cax.axis('tight')
    # cax.set_axis_off()
    cax.get_xaxis().set_visible(False)
    cax.set_yticklabels([])
    cax.set_ylabel( labels[i], fontsize=12,rotation=0, color=colors[i]) # r"$\bf{" + labels[i] + "}$"
    concat_img = None
  cax = plt.subplot2grid((n_vids + plot_size, 1), (vid_cnt, 0), rowspan=3)
    
    # ax.axis('off')
    # plt.subplot(212)

  rewards = np.asarray(rewards)
  #rewards_ff = np.load(join(outdir, 'all_rewards_ff.npy'))[:, plot_config.START:plot_config.END]
  iii = 0

  for reward, label, color in zip(rewards[:, plot_config.START:plot_config.END], labels, colors):
    if label in plot_config.EXCLUDED_LABELS or label == 'demo':
      iii +=1 
      continue
    cax.plot(reward, color=color, label=label+ ', object_centric', linewidth=2.0)
    #cax.plot(rewards_ff[iii], color=color, label=label+', full_frame', linewidth=2.0, linestyle='--')
  cax.legend(fontsize=10, loc='lower left')
  cax.tick_params(labelsize=12)
  if y_lim is not None:
    cax.set_ylim(y_lim)
  # ax2.autoscale(False)
  fig.subplots_adjust(hspace=0, wspace=0)
  fig.savefig(join(outdir, 'all_rewards.jpg'))
  # fig.savefig(join(outdir, 'all_rewards.fig'))
  np.save(join(outdir, 'all_rewards.npy'), rewards)
  # plt.show()
  # set_trace()

def KNNIds(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  sorted_distances = KNNIdsWithDistances(query_vec, target_seq, k)
  return [i[0] for i in sorted_distances]

def reward_fn(x, y):
  # return -np.linalg.norm(np.abs(x) - np.abs(y))
  alpha= 100
  beta = 0.1
  return -alpha*((np.abs(x) - np.abs(y)).dot((np.abs(x) - np.abs(y)))**2 + beta * np.sqrt((np.abs(x) - np.abs(y)).dot((np.abs(x) - np.abs(y))) + 0.0001))

def KNNIdsWithDistances(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  if not isinstance(np.array(target_seq), np.ndarray):
    target_seq = np.array(target_seq)
  assert np.shape(query_vec) == np.shape(target_seq[0])
  distances = [(i, np.linalg.norm(query_vec-target_vec)) for (
      i, target_vec) in enumerate(target_seq)]
  sorted_distances = sorted(distances, key=lambda x: x[1])
  return sorted_distances[:k]



def ConcatFrames(video_file, outdir):
  reader = imageio.get_reader(video_file)
  for i, img in enumerate(reader):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=1)
  plt.imsave(join(outdir, video_file.split('.mp4')[0] + '.jpg'), concat_img)

def ConcatFramesAndRewards(video_file, rewards, labels, colors, outdir, y_lim=None):
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
  for reward, label, color in zip(rewards, labels, colors):
    ax2.plot(reward, color=color, label=label, linewidth=3.0)
  ax2.legend()
  plt.tick_params(labelsize=5)
  if y_lim is not None:
    ax2.set_ylim(y_lim)
  # ax2.autoscale(False)
  fig.subplots_adjust(hspace=0, wspace=0)
  ax.set_axis_off()
  fig.savefig(join(outdir, (video_file.split('/')[-1]).split('.mp4')[0] + '_rewards.jpg'))
  # plt.show()
  # set_trace()


def main_multiple(args):
  colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
  colors = random_colors(10)

  OUTPUT_FOLDER = join(INPUT_PATH, args.output_path)
  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
  input_folder = join(INPUT_PATH, args.target_path)

  query_embs_centroid = np.load(join(INPUT_PATH, args.query_path))
  query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual.npy'))
  query_embs_vis = np.reshape(query_embs_vis_raw, [10, 64])
  # query_embs_vis = query_embs_vis_raw

  # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
  # sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)

  query_embs = np.concatenate([query_embs_centroid, query_embs_vis], axis=1)  
  
  all_rewards = []
  all_labels = []
  input_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    if not file.endswith('emb.npy'):
      continue
    # if args.query_path.split('/')[-1] == file:
    #   continue

    target_embs_centroid = np.load(join(input_folder, file))
    target_embs_vis_raw = np.load(join(input_folder, file.strip('.npy') + '_visual.npy'))
    target_embs_vis = np.reshape(target_embs_vis_raw, [10, 64])
    # target_embs_vis = target_embs_vis_raw
    
    # target_embs_roi = np.load(join(INPUT_PATH, file.strip('.npy') + '_roi.npy'))
    # target_embs_roi = apply_feature_selection_episode(target_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)


    target_embs = np.concatenate([target_embs_centroid, target_embs_vis], axis=1)
    all_rewards.append(MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0))
    all_labels.append(file)


    # Make GT fullframe reward plot
    target_embs_vis_ff = np.load(join(input_folder, file.strip('.npy') + '_visual_ff.npy'))
    target_embs_ff = np.concatenate([target_embs_centroid, target_embs_vis_ff], axis=1)
    query_embs_vis_ff = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual_ff.npy'))
    # query_embs_vis = query_embs_vis_raw
    # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
    # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)
    query_embs_ff = np.concatenate([query_embs_centroid, query_embs_vis_ff], axis=1)  
    reward_ff = MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs_ff, target_embs_ff, nearest=False, centroid_weighting=0.0, plot_figure=False)


    ConcatFramesAndRewards( join(input_folder, file.split('emb')[0] + 'video_sample.mp4'), \
                                  [all_rewards[ii], reward_ff] , ['Object-Centric', 'Full-Frame'], colors, OUTPUT_FOLDER)
    ii += 1
  plot_multiple_mean(np.asarray(all_rewards)[:,:], OUTPUT_FOLDER, all_labels, name='rewards_plot')

def main_single(args):
  colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
  colors = random_colors(10)

  OUTPUT_FOLDER = join(INPUT_PATH, args.output_path)
  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
  input_folder = join(INPUT_PATH, args.target_path)

  query_embs_centroid = np.load(join(INPUT_PATH, args.query_path))
  query_embs_vis_raw = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_visual.npy'))
  query_embs_vis = np.reshape(query_embs_vis_raw, [10, 64])
  # query_embs_vis = query_embs_vis_raw

  # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
  # sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)

  query_embs = np.concatenate([query_embs_centroid, query_embs_vis], axis=1)  
  
  all_rewards = []
  all_labels = []
  input_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    if not file.endswith('emb.npy'):
      continue
    # if args.query_path.split('/')[-1] == file:
    #   continue

    target_embs_centroid = np.load(join(input_folder, file))
    target_embs_vis_raw = np.load(join(input_folder, file.strip('.npy') + '_visual.npy'))
    target_embs_vis = np.reshape(target_embs_vis_raw, [10, 64])
    # target_embs_vis = target_embs_vis_raw
    
    # target_embs_roi = np.load(join(INPUT_PATH, file.strip('.npy') + '_roi.npy'))
    # target_embs_roi = apply_feature_selection_episode(target_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)


    target_embs = np.concatenate([target_embs_centroid, target_embs_vis], axis=1)
    all_rewards.append(MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0.0))
    all_labels.append(file)


    ConcatFramesAndRewards( join(input_folder, file.split('emb')[0] + 'video_sample.mp4'), \
                                  [all_rewards[ii]] , ['Object-Centric'], colors, OUTPUT_FOLDER)
    ii += 1
  plot_multiple_mean(np.asarray(all_rewards)[:,:], OUTPUT_FOLDER, all_labels, name='rewards_plot', y_lim=[plot_config.LOWER_BOUND_PLOT, 0.05])


if __name__ == '__main__':

    args = parser.parse_args()
    # main_combined(args)
    # main_single(args)
    main_all_no_images(args)        
