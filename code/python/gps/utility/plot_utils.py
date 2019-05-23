import datetime
import itertools
import os
import numpy as np
from pdb import set_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdb import set_trace as st

def plot_split_count(counts_train, counts_test, path, name='split_count', save_figure=True, overwrite=True):


    N = len(counts_train)
    train_means = [val for key, val in counts_train.items()]

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.barh(ind, train_means, width, color='r')

    test_means = [val for key, val in counts_test.items()]
    rects2 = ax.barh(ind + width, test_means, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_xlabel('Counts')
    ax.set_title('Number of label occurences')
    ax.set_yticks(ind + width / 2)
    ax.set_yticklabels([key for key, val in counts_train.items()], fontsize=8)

    ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)

        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.jpg'.format(date=datetime.datetime.now()))
            fig.savefig(figure_path)
    plt.close()
    return

    def _autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    _autolabel(rects1)
    _autolabel(rects2)
    return

def plot_mean(mean, path, name='mean', ylabel='loss', save_figure=True, overwrite=True):
    # plots the mean and 1 sigma interval of given mean and std array.
    # path is where to store and name is unique indicate of figure
    fig = plt.figure()
    n = len(mean)
    epochs = np.arange(1, n+1, dtype=np.int32)
    plt.plot(epochs, mean)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)

        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.jpg'.format(date=datetime.datetime.now()))
            fig.savefig(figure_path)
    plt.close()
    return

def plot_multiple_mean(mean, path, labels, name='multiple_mean', ylabel='accuracy', save_figure=True, overwrite=True):
    # plots multiple results in one figue. 
    # mean and std:
    #    given as (M, N) array where m indicates
    #    the current experiment and n the epoch of the experiment
    # name: 
    #   list of names for each experiment 
    plt.figure()
    M = mean.shape[0]
    N = mean.shape[1]
    epochs = np.arange(1, N+1)
    colors = plt.cm.hsv(np.linspace(0, 1, N)).tolist()
    
    for m in range(M):
        plt.plot(epochs, mean[m, :], color=colors[m], label=labels[m])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name )
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            plt.savefig(figure_path)

            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def plot_confusion_matrix(cm, classes, path, name='confusion_matrix',
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, save_figure=True, overwrite=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # cm = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis], out=np.zeros_like(cm.astype('float')), where=cm.sum(axis=1)[:, np.newaxis]!=0) 
        cm = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])

    else:
        pass

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, rotation=90)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            plt.savefig(figure_path)

            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}.jpg'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def plot_results(mean, std, path, name, save_figure=True, color='g', overwrite=True):
    # plots the mean and 1 sigma interval of given mean and std array.
    # path is where to store and name is unique indicate of figure
    plt.ioff()
    fig = plt.figure()
    n = len(mean)
    epochs = np.arange(1, n+1, dtype=np.int32)
    plt.plot(epochs, mean, color=color)
    plt.fill_between(epochs, mean + std, mean - std, color=color, alpha=0.2)
    plt.xlabel('Iterations')
    plt.ylabel('Average cost')
    plt.grid(True)
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)

        figure_path = os.path.join(path, name + '.png')
        if not os.path.isfile(figure_path) or overwrite:
            fig.savefig(figure_path, format='png')
        else:
            print("Figure already existed under given name. Saved with current time stamp")
            figure_path = os.path.join(path, name + '_{date:%Y-%m-%d_%H-%M-%S}.png'.format(date=datetime.datetime.now()))

            fig.savefig(figure_path, format='png')
    plt.close()
    return

def plot_multiple(mean, std, path, labels, name, save_figure=True, overwrite=True):
    # plots multiple results in one figue. 
    # mean and std:
    #    given as (M, N) array where m indicates
    #    the current experiment and n the epoch of the experiment
    # name: 
    #   list of names for each experiment 
    plt.figure()
    M = mean.shape[0]
    N = mean.shape[1]
    epochs = np.arange(1, N+1)
    colors = plt.cm.hsv(np.linspace(0, 1, N)).tolist()
    
    for m in range(M):
        plt.errorbar(epochs, mean[m, :],  std[m, :], color=colors[m], label=labels[m])
        plt.xlabel('Epoch')
        plt.ylabel('Return')
    # save figure, if already exists then save under same name with current time stamp
    if save_figure:
        if not os.path.exists(path):
            os.makedirs(path)
        figure_path = os.path.join(path, name + '.jpg')
        if not os.path.isfile(figure_path) or overwrite:
            plt.savefig(figure_path)
        else:
            print("Figure already existed under given name. Saved with ccurrent time stamp")
            figure_path = os.path.join(path, name + '{date:%Y-%m-%d_%H:%M:%S}.jpg'.format(date=datetime.datetime.now()))
            plt.savefig(figure_path)
    plt.close()
    return

def save_statistics(mean, std, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    # save mean
    mean = np.asarray(mean)
    np.save(os.path.join(path, name + '_mean'), mean)
    # save std
    std = np.asarray(std)
    np.save(os.path.join(path, name + '_std'), std)




def concat_frames(video_file, outdir):
  reader = imageio.get_reader(video_file)
  for i, img in enumerate(reader):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=1)
  plt.imsave(join(outdir, video_file.split('.mp4')[0] + '.jpg'), concat_img)


def concat_frames_nosave(frames):
  for i, img in enumerate(frames):
      if i == 0:
        concat_img = img
        continue
      else:
        concat_img = np.concatenate([concat_img, img], axis=2)
  return concat_img

