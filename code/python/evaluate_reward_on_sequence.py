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
from ipdb import set_trace as st

sys.path.append('/home/zhouxian/projects/gps-lfd')
from gps.utility.plot_utils import plot_multiple_mean
random.seed(2)


# TCN_MODEL_PATH = '/media/msieb/1e2e903d-5929-40bd-a22a-a94fd9e5bcce/tcn_data/experiments/mugs/trained_models/tcn-no-depth-sv/ltcn-epoch-8.pk'
EXP_DIR = '/home/zhouxian/cvpr_visualizations'
EXP_NAME = 'bottle' 
FEATURE_FOLDER = 'features'
IMAGE_FOLDER = join(EXP_DIR, EXP_NAME, 'videos', 'train')
INPUT_FOLDER = join(EXP_DIR, EXP_NAME, FEATURE_FOLDER)
OUTPUT_FOLDER = join(EXP_DIR, EXP_NAME, 'plots')
if not os.path.exists(OUTPUT_FOLDER):
  os.makedirs(OUTPUT_FOLDER)


##### FIX QUERY EMB NAME (Its a sequence of (EMB_DIM, T)
TARGETS_FOLDER = INPUT_FOLDER




class GiraffePlotConfig(object):
  # ALL_LABELS = ['demo', 'robot correct', 'robot wrong', 'robot correct w/ distractors']
  EXCLUDED_LABELS = []
  # EXCLUDED_LABELS = ['open', 'robot correct']
  # EXCLUDED_LABELS = ['fake']
  # EXCLUDED_LABELS = ['open',  'robot wrong']
  START = 0
  END = 100
  MODE = 'tcn'
  LOWER_BOUND_PLOT = -20
  # Y_LIM = [LOWER_BOUND_PLOT, 0.05]
  Y_LIM = None
  PLOT_SIZE = 1

plot_config = GiraffePlotConfig()
QUERY_EMB_PATH = join(INPUT_FOLDER, 'demo_{}.npy'.format(plot_config.MODE))

def main(args):
  colors =  plt.cm.hsv(np.linspace(0, 1, 6)).tolist()
  colors = ['g', 'r', 'b', 'purple'] * 3

  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
  input_folder = join(INPUT_FOLDER, args.target_path)
  input_folder = INPUT_FOLDER
  output_folder = OUTPUT_FOLDER

  # query_embs_vis = np.load(QUERY_EMB_PATH)
  # query_embs_vis = query_embs_vis_raw

  # query_embs_roi = np.load(join(INPUT_PATH, args.query_path.strip('.npy') + '_roi.npy'))
  # sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2 = feature_selection(query_embs_vis_raw)
  # query_embs_roi = apply_feature_selection_episode(query_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)

  
  all_rewards = []
  all_labels = []
  all_files = []
  input_files = os.listdir(input_folder)
  # input_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[0]))
  ii = 0
  for file in input_files:
    # if file[:4] == 'demo':
    #   continue
    # if not file.endswith(plot_config.MODE + '.npy'):
    #   continue
    # if file[:4] == 'demo':
      # continue
    # if args.query_path.split('/')[-1] == file:
    #   continue


    query_embs_path = join(input_folder, 'demo_' + file.split('_')[-1])
    query_embs = np.load(query_embs_path)
    target_embs = np.load(join(input_folder, file))
    # target_embs_vis = target_embs_vis_raw
    
    # target_embs_roi = np.load(join(INPUT_PATH, file.strip('.npy') + '_roi.npy'))
    # target_embs_roi = apply_feature_selection_episode(target_embs_roi, sel_idx_1, sel_idx_2, sel_idx_max_1, sel_idx_max_2)
    all_rewards.append(MakeRewardPlot(OUTPUT_FOLDER, file.split('.')[0], query_embs, target_embs, nearest=False, centroid_weighting=0.0))
    all_labels.append(file.split('.npy')[0])

    image_folder_path = join(IMAGE_FOLDER, file.split('_')[0])
    all_files.append(image_folder_path)
    ii += 1

  # all_files = sorted(all_files, key=lambda x: int(x.split('/')[-1].split('_')[0]))
  # include_idx = [0, 4, 5, 6]
  # include_idx = [0, 1, 2, 3]

  # all_labels = ['Demo', 'Idle', 'Front' ,'Half Way', 'Robot Correct Jittery', 'Robot Wrong', 'Robot Correct']
  # all_labels = ['Demo', 'Wrong', 'Wrong', 'Wrong','Wrong', 'Correct End Pose', 'Correct Start Pose']
  # all_labels = plot_config.ALL_LABELS
  st()
  ConcatFramesAndRewardsFromImagesMultiple(  all_files, \
                                  all_rewards , all_labels, colors, output_folder, y_lim=plot_config.Y_LIM)
  
  # ConcatFramesAndRewardsMultiple(  all_files, \
  #                                 all_rewards , ['Demo', 'Different Ring', 'Robot Correct', 'Robot Wrong Behind', 'Robot Wrong Front'], colors, OUTPUT_FOL$
  
  # ConcatFramesAndRewardsMultiple(  all_files, \
  #                                 all_rewards , ['Demo', 'Different Ring', 'Other Side', 'Robot Correct', 'Robot Other Side', 'Robot Wrong Other Side', 'R$
  
  plot_multiple_mean(np.asarray(all_rewards)[:,:], output_folder, all_labels, name='rewards_plot')#, y_lim=plot_config.LOWER_BOUND_PLOT)



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
      # target_emb_norm = target_emb / np.linalg.norm(target_emb)
      # query_embs_norm = query_emb / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_emb, target_emb))
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

      # target_emb_norm = target_emb / np.linalg.norm(target_emb)
      # query_embs_norm = query_emb  / np.linalg.norm(query_emb)
      # rewards.append(reward_fn(query_embs[i], knn_embs[i]))
      rewards.append(reward_fn(query_emb, target_emb))
  if plot_figure:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.set_ylim([plot_config.LOWER_BOUND_PLOT, 0.05])    
    plt.plot(rewards)
    fig.savefig(os.path.join(outdir, name + '.png'))
    plt.close(fig)
  return rewards

def ConcatFramesAndRewardsFromImagesMultiple(image_folder_paths, rewards, labels, colors, outdir, y_lim=None):
  n_folders = len(image_folder_paths)
  plot_size = plot_config.PLOT_SIZE
  fig = plt.figure(figsize=(15, n_folders/2.0+5))
  concat_img = None
  folder_cnt = 0
  processed = []
  processed_folders = []
  for i, folder_path in enumerate(image_folder_paths):
    folder = folder_path.split('/')[-1]
    if folder in processed_folders:
      processed.append(i)
      continue
    processed_folders.append(folder)
    cax = plt.subplot2grid((n_folders + plot_size, 1), (folder_cnt, 0))
    image_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
    for j, img_file in enumerate(image_files):
        img = plt.imread(join(folder_path, img_file))
        if j < plot_config.START or j > plot_config.END:
          continue
        if concat_img is None:
          concat_img = img
          continue
        else:
          concat_img = np.concatenate([concat_img, img], axis=1)

    folder_cnt += 1
    cax.imshow(concat_img)
    cax.axis('tight')
    # cax.set_axis_off()
    cax.get_xaxis().set_visible(False)
    cax.set_yticklabels([])
    # cax.set_ylabel( label, fontsize=12,rotation=0, color=colors[i]) # r"$\bf{" + labels[i] + "}$"
    # cax.set_ylabel( label, fontsize=12,rotation=0)
    concat_img = None

  cax = plt.subplot2grid((n_folders + plot_size, 1), (folder_cnt, 0), rowspan=3)
    
    # ax.axis('off')
    # plt.subplot(212)
  rewards = np.asarray(rewards)

  tcn_labels = np.where(np.array([l[-4:] for l in labels]) == '_tcn')
  don_labels = np.where(np.array([l[-4:] for l in labels]) == '_don')
  tcn_rewards = rewards[tcn_labels]
  don_rewards = rewards[don_labels]
  rewards[tcn_labels] /= np.max(np.abs(tcn_rewards))
  rewards[don_labels] /= np.max(np.abs(don_rewards))
  #rewards_ff = np.load(join(outdir, 'all_rewards_ff.npy'))[:, plot_config.START:plot_config.END]
  iii = 0
  tcn_line = '--'
  wwg_line = '-'
  plotted_i = 0
  plotted_j = 0
  for reward, label, color in zip(rewards[:, plot_config.START:plot_config.END], labels, colors):
    if label in plot_config.EXCLUDED_LABELS or label[:4] == 'demo':
      iii +=1 
      continue

    # reward = (reward - np.mean(reward)) / np.std(reward)
    color = colors[plotted_j % 2] 
    print(plotted_j)
    if label.endswith('_tcn'):

      cax.plot(reward, color=color, label=label, linestyle=tcn_line, linewidth=2.0)
    else:
      lab = label[:-4] + '_wwg'
      cax.plot(reward, color=color, label=lab, linestyle=wwg_line, linewidth=2.0)
    if plotted_i % 2:
      plotted_j += 1
    plotted_i += 1
    #cax.plot(rewards_ff[iii], color=color, label=label+', full_frame', linewidth=2.0, linestyle='--')
  cax.legend(fontsize=10, loc='lower left')
  cax.tick_params(labelsize=12)
  if y_lim is not None:
    cax.set_ylim(y_lim)
  # ax2.autoscale(False)
  fig.subplots_adjust(hspace=0, wspace=0)
  fig.savefig(join(outdir, 'all_rewards'))
  # fig.savefig(join(outdir, 'all_rewards.fig'))
  np.save(join(outdir, 'all_rewards.npy'), rewards)
  # plt.show()
  # set_trace()


def KNNIds(query_vec, target_seq, k=1):
  """Gets the knn ids to the query vec from the target sequence."""
  sorted_distances = KNNIdsWithDistances(query_vec, target_seq, k)
  return [i[0] for i in sorted_distances]

def reward_fn_(x, y):
  # return -np.linalg.norm(np.abs(x) - np.abs(y))
  alpha= 100
  beta = 0.1
  return -alpha*((np.abs(x) - np.abs(y)).dot((np.abs(x) - np.abs(y)))**2 + beta * np.sqrt((np.abs(x) - np.abs(y)).dot((np.abs(x) - np.abs(y))) + 0.0001))

def reward_fn(x, y):
  # return -np.linalg.norm(np.abs(x) - np.abs(y))
  alpha= 1
  beta = 0.1
  return -alpha*np.sqrt((np.abs(x) - np.abs(y)).dot((np.abs(x) - np.abs(y))) )

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-path', type=str, default=TARGETS_FOLDER)
    parser.add_argument('--query-path', type=str, default=QUERY_EMB_PATH)
    parser.add_argument('--output-path', type=str, default=OUTPUT_FOLDER)
    args = parser.parse_args()
    main(args)        
