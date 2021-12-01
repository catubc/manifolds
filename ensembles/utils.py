import matplotlib

import os
os.chdir('/home/cat/code/manifolds/')

#

import scipy
import numpy as np
import pandas as pd

from calcium import calcium
from wheel import wheel
from visualize import visualize
from tqdm import trange

from scipy.io import loadmat

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#
np.set_printoptions(suppress=True)


############### SSM FUNCTIONS ##########################
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import ssm
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap


###############################################################
###############################################################
##############################################################
def load_UMAP(fname):
    data = np.load(fname, allow_pickle=True)

    X_umap = data['X_umap']
    n_components = data['n_components']
    min_dist = data['min_dist']
    n_neighbors = data['n_neighbors']
    metric = data['metric']

    return X_umap


#
def load_binarized_traces(root_dir,
                          animal_id,
                          session,
                          bintype='upphase'  # 'spikes', 'upphase', 'oasis'

                          ):
    fname = os.path.join(root_dir,
                         animal_id,
                         session,
                         'suite2p',
                         'plane0',
                         'binarized_traces.npz'
                         )

    #
    data = np.load(fname, allow_pickle=True)
    spikes = data['spks']
    upphase = data['F_upphase']
    onphase = data['F_onphase']
    spikes_smooth = data['spks_smooth_upphase']

    #
    if bintype == 'spikes':
        return spikes
    elif bintype == 'spikes_smooth':
        return spikes_smooth
    elif bintype == 'upphase':
        return upphase
    elif bintype == 'onphase':
        return onphase


def find_ensemble_order(clusters_without_noise,
                        times,
                        ):
    #
    ensemble_frs = []
    ctr = 0
    for c in np.unique(clusters_without_noise):

        # find times of cluster
        idx = np.where(clusters_without_noise == c)[0]
        times_original = times[idx]
        if ctr == 0:
            print("ensemble order: ", c, times_original)

        # compute average timebin firing rate:
        # ave_fr = (rasters[:,times_original].T).sum(axis=1).mean(0)
        ave_fr = (rasters[:, times_original].T).mean()
        ensemble_frs.append(ave_fr)
        ctr += 1

    ensemble_frs = np.array(ensemble_frs)
    idx = np.argsort(ensemble_frs)[::-1]

    print("Sorted clusters: ", ensemble_frs[idx[:20]])
    print("sorted cluster ids: ", idx[:20])

    id3 = np.where(clusters_without_noise == idx[0])[0]
    ave_fr = (rasters[:, id3].T).sum(axis=1).mean(0)
    print("highest firing rate time bins: ", id3, " with frate: ", ave_fr)

    return idx


def load_data(root_dir, animal_id,
              session,
              dim_type,
              bin_type,
              binarize_flag=False):

    # 
    data = np.load(os.path.join(
        root_dir,
        animal_id,
        session,
        'suite2p',
        'plane0',
        'res_dbscan_' + dim_type + '_' + bin_type + '.npz'))

    rasters = data['rasters'].T
    #print("Rasters: ", rasters.shape)

    X_pca = data['X_pca']
    #print("X pca: ", X_pca.shape)

    clusters = data['labels']
    #print("# clusters: ", np.unique(clusters).shape)

    # B
    if binarize_flag:
        print("rasters: ", rasters.shape)
        idx1 = np.where(rasters >= 1)
        idx2 = np.where(rasters < 1)
        rasters[idx1] = 1
        rasters[idx2] = 0

    return X_pca, rasters
