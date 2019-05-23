import os
from os.path import join
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.decomposition import PCA
from pdb import set_trace

sys.path.append('/home/max/projects/gps-lfd')
sys.path.append('/home/msieb/projects/gps-lfd')
#from config import Config as Config # Import approriate config
from config import Training_Config as Config # Import approriate config
conf = Config()

EXP_DIR = conf.EXP_DIR
GPS_EXP_DIR = conf.GPS_EXP_DIR
EXP_NAME = conf.EXP_NAME
EMBEDDING_DIM = conf.EMBEDDING_DIM

DATA_PATH =join(GPS_EXP_DIR, EXP_NAME, 'data_files')

DEMO_FILE = np.load(join(EXP_DIR, "demonstrations/duck","{}_tcn_features_raw.npy".format(conf.SEQNAME)))
DEMO_FILE /= np.linalg.norm(DEMO_FILE, axis=1)[:,None]


def main():
    print('Visualizing PILQR iterations')
    print('Loaded Demo File {}: '.format(join(EXP_DIR, "demonstrations/duck","{}_tcn_features_raw.npy".format(conf.SEQNAME))))
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
    files = sorted(files, key=lambda f: int(f.split('.npy')[0].split('_')[-1]))
    files = files[:] 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 component PCA', fontsize = 20)
    targets = [] 
    all_features = []
    for itr, file in enumerate(files[:10]):
        X = np.load(join(DATA_PATH, file))
        # set_trace() 
        features = X[:,:,-32:]

        features = np.reshape(features, [features.shape[0] * \
                                  features.shape[1], features.shape[2]])
        all_features.append(features / np.linalg.norm(features, axis=1)[:, None])

    all_features = np.array(all_features)
    X = np.reshape(all_features,[all_features.shape[0]*all_features.shape[1], -1])
    set_trace()
    pca = PCA(n_components=3)
    pca.fit(X)
            
    for itr, file in enumerate(files):
        #features = np.load(join(DATA_PATH, file))[:, :, 23:26]
        #features = np.load(join(DATA_PATH, file))[:, :, -EMBEDDING_DIM:]
        X = np.load(join(DATA_PATH, file))
        features = X[:,:,-32:]

        cmap = get_cmap(10)     
        mean, std, demo = apply_pca(features, pca)
        alpha=1.0 - np.exp(-itr*0.2-0.05) 
        alpha = np.arange(0,1,0.05)
        ax.plot(mean[:, 0]
                , mean[:,1]
                , mean[:,2]
                   , c = cmap(0)
                   ,alpha=alpha[itr]
                   , markersize=1 
                   , linestyle='-'
                   #, linewidth=np.linalg.norm(std, axis=1).tolist()
                   , marker='o')
        ax.scatter(mean[0, 0]
            , mean[0,1]
            ,mean[0,2]
                   , c = cmap(0)
                   ,alpha=alpha[itr])
               #, linewidth=np.linalg.norm(std, axis=1).tolist()
        targets.append('itr_{}'.format(itr))

    ax.scatter(demo[0, 0]
            , demo[0,1]
            ,demo[0,2]
               , c = 'g'
               ,alpha=1.0)
               #, linewidth=np.linalg.norm(std, axis=1).tolist()
    ax.plot(demo[:, 0]
            , demo[:,1]
            ,demo[:,2]
               , c = 'g'
               ,alpha=1.0
               , markersize=1 
               , linestyle='-'
               #, linewidth=np.linalg.norm(std, axis=1).tolist()
               , marker='o')
    targets.append('demo')
    ax.legend(targets)
    ax.grid()
        
    
    plt.show()

def apply_pca(X, pca):
    cmap = get_cmap(len(X))

    
    traces = []
  
    #fig = plt.figure(figsize = (8,8))
    #ax = fig.add_subplot(111, projection='3d') 
    #ax.set_xlabel('Principal Component 1', fontsize = 15)
    #ax.set_ylabel('Principal Component 2', fontsize = 15)
    #ax.set_title('3 component PCA', fontsize = 20)
    targets = [] 
    projected_trajectories = []
    for itr in range(X.shape[0]):
        X_fit = pca.transform(X[itr])
        X_std = np.std(X_fit, axis=0)
        X_mean = np.mean(X_fit, axis=0)
        projected_trajectories.append(X_fit)
   #     ax.plot(X_fit[:, 0]
   #                , X_fit[:, 1]
   #                , X_fit[:, 2]
   #                , c = cmap(itr)
   #                , markerSize =5 
   #                , linestyle='-'
   #                , marker='o')
   #     targets.append('itr_{}'.format(itr))
   # ax.legend(targets)
   # ax.grid()
    #set_trace()
    demo = pca.transform(DEMO_FILE)
    projected_trajectories = np.array(projected_trajectories)
    X_mean = np.mean(projected_trajectories, axis=0)
    X_std = np.std(projected_trajectories, axis=0)
    return X_mean, X_std, demo

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



if __name__ == '__main__':
    main()
