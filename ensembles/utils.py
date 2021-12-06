import matplotlib
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from tqdm import tqdm, trange
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


class HMM():

    def __init__(self):
        pass

    def get_hmm_stats(self, hmm_z):
        lens = []
        ids = []
        starts = []
        ends = []
        start = 0
        current_state = hmm_z[0]
        #print("# unique ids starting ", np.unique(hmm_z))

        starts.append(0)
        #############################
        k = 0
        while k < hmm_z.shape[0]:
            # for k in range(hmm_z.shape[0]):

            if hmm_z[k] != current_state:
                lens.append(k - start)
                ids.append(hmm_z[k - 1])
                start = k
                # k+=1
                current_state = hmm_z[k]
                ends.append(k-1)
                starts.append(k)
                # continue
            k += 1
        ids.append(hmm_z[hmm_z.shape[0]-1])
        ends.append(hmm_z.shape[0]-1)
        lens.append(hmm_z.shape[0]-1-start)

        windows = np.vstack((starts, ends)).T

        ids = np.hstack(ids)
        #print("# unique ids found    ", np.unique(ids))
        lens = np.hstack(lens) / 30.

        # reorder
        lens_per = []
        n_occurance = []
        total_durations = []
        for id_ in np.unique(ids):
            idx = np.where(ids == id_)[0]
            lens_per.append(np.median(lens[idx]))
            n_occurance.append(idx.shape[0])
            total_durations.append(np.sum(lens[idx]))

        self.lens = lens
        self.ids = ids
        self.lens_per = lens_per
        self.n_occurance = n_occurance
        self.total_durations = total_durations
        self.windows = windows

    def computing_ensemble_loadings_per_occurance(self,
                                                  state_id,
                                                  rasters):
        #
        print ("state id: ", state_id)
        idx = np.where(self.ids == state_id)

        #
        segs = self.windows[idx]

        # grab the segmented raster
        cell_sums = []
        state_durations = []
        for s in tqdm(segs):
            raster_state = []
            temp = rasters[s[0]:s[1] + 1]
            raster_state.append(temp)

            raster_state = np.vstack(raster_state).T
            #
            cell_sums.append(raster_state.sum(1))

            state_durations.append(s[1]+1-s[0])

        state_durations = np.array(state_durations)/30
        cell_sums = np.array(cell_sums)
        print ("cell sums longitudinal: ", cell_sums.shape)

        #
        return cell_sums, state_durations



def get_footprint_contour(c, cell_id):
    points = np.vstack((c.stat[cell_id]['xpix'],
                        c.stat[cell_id]['ypix'])).T

    #
    hull = ConvexHull(points)
    hull_ids = hull.vertices
    hull_points = points[hull.vertices]
    hull_points = np.vstack((hull_points, hull_points[0]))

    return hull_points


#
def load_footprints(c):
    dims = [512, 512]

    img_all = np.zeros((dims[0], dims[1]))
    imgs = []
    contours = []
    for k in range(len(c.stat)):
        x = c.stat[k]['xpix']
        y = c.stat[k]['ypix']
        img_all[x, y] = c.stat[k]['lam']

        # save footprint
        img_temp = np.zeros((dims[0], dims[1]))
        img_temp[x, y] = c.stat[k]['lam']

        img_temp_norm = (img_temp - np.min(img_temp)) / (np.max(img_temp) - np.min(img_temp))
        imgs.append(img_temp_norm)

        contours.append(get_footprint_contour(c, k))

    imgs = np.array(imgs)

    # binarize footprints
    imgs_bin = imgs.copy() * 1E5
    imgs_bin = np.clip(imgs_bin, 0, 1)

    return imgs, img_all, imgs_bin, contours


#
def computing_ensemble_loadings(state_id,
                                h,
                                rasters):
    #
    print ("state id: ", state_id)
    idx = np.where(h.ids == state_id)

    #
    segs = h.windows[idx]

    # grab the segmented raster
    raster_state = []
    img_out = []
    for s in tqdm(segs):
        temp = rasters[s[0]:s[1] + 1]
        raster_state.append(temp)

    raster_state = np.vstack(raster_state).T
    print("raster_state: ", raster_state.shape)
    #
    cell_sums = raster_state.sum(1)

    return cell_sums
