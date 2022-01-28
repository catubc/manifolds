import numpy as np
import os
from tqdm import trange, tqdm
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt

#
import parmap

import os
os.chdir('/home/cat/code/manifolds/')
#
import matplotlib.pyplot as plt

import scipy
import numpy as np
import pandas as pd
from matplotlib.path import Path
from tqdm import tqdm, trange

from calcium import calcium
from wheel import wheel
from visualize import visualize
from tqdm import trange

import cv2
from scipy.io import loadmat
import matplotlib.patches as mpatches

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, convex_hull_plot_2d

####
def find_percent_overlap(cell1, cell2):
    n_pix = max(np.where(cell1 == 1)[0].shape[0],
                np.where(cell2 == 1)[0].shape[0])

    temp = cell1 + cell2
    idx = np.where(temp == 2)[0]

    return idx.shape[0] / n_pix

####
class Longitudinal():

    def __init__(self, root_dir, animal_id, sessions):

        self.root_dir = root_dir
        self.animal_id = animal_id

        #
        self.sessions = sessions

        # imaging FOV size; may want to automatically load this
        self.size = [512,512]


    def make_homography_matrix_from_ROIMatchPub_clicks_load_multi_session(self, fname):

        # load mouse click points                    *
        data = scipy.io.loadmat(fname)    #         session id
        transformations = data['roiMatchData']['rois'][0][0][0]
        print (" # of sessions to be transformed: ", transformations.shape,
               "  (first session will have identity matrix)")

        # make all the homographic matrices based on id = 0 and id = k
        hh_array = []
        diag = np.diag((1,1,1))
        hh_array.append(diag)
        for k in range(1, len(transformations), 1):
            pts1 = data['roiMatchData']['rois'][0][0][0][k][0][0]['trans'][0][0]['fixed_out']
            pts2 = data['roiMatchData']['rois'][0][0][0][k][0][0]['trans'][0][0]['moving_out']

            #
            matrix = cv2.findHomography(pts1, pts2)
            hh = matrix[0]
            M = np.linalg.inv(hh)

            #
            hh_array.append(M)

        self.hh_array = hh_array

    def make_homography_matrix_from_ROIMatchPub_clicks(self, fname):

        sess1 = self.sessions[0][-2:]
        sess2 = self.sessions[1][-2:]

        # load mouse click points                    *
        data = scipy.io.loadmat(fname)    #         session id
        pts1 = data['roiMatchData']['rois'][0][0][0][1][0][0]['trans'][0][0]['fixed_out']
        pts2 = data['roiMatchData']['rois'][0][0][0][1][0][0]['trans'][0][0]['moving_out']

        # compute homography (returns two vals)
        matrix = cv2.findHomography(pts1, pts2)
        hh = matrix[0]

        # make inverted matrix
        M = np.linalg.inv(hh)

        self.hh = hh
        self.M = M

        print ("Inverse homographic matrix: \n", self.M)

    def transform_cells_all_sessions(self):

        #
        x, y = np.meshgrid(np.arange(self.size[0]),
                           np.arange(self.size[1]))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points_mesh = np.vstack((x, y)).T

        #
        fname_out_masks = os.path.join(self.root_dir, self.animal_id,'masks_'+self.cell_boundary+'.npy')
        fname_out_contours = os.path.join(self.root_dir, self.animal_id,'contours_'+self.cell_boundary+'.npy')
        fname_out_allcell_masks = os.path.join(self.root_dir, self.animal_id,'allcell_masks_'+self.cell_boundary+'.npy')

        if os.path.exists(fname_out_masks)==False:

            masks = []
            contours = []
            allcell_masks = []
            ctr=0
            for session in self.sessions:
                mask0 = np.zeros((self.size[0],
                                  self.size[1]))

                #
                masks.append([])
                contours.append([])
                allcell_masks.append(mask0)

                #
                c = calcium.Calcium()
                c.root_dir = self.root_dir
                c.animal_id = self.animal_id
                c.session = session
                c.load_suite2p()

                #
                cell_ids = np.arange(c.F.shape[0])
                for cell in tqdm(cell_ids):

                    #
                    contour = c.get_footprint_contour(cell, self.cell_boundary)

                    # this transformation could be done in a single step by keeping track of contour indexes
                    contour = np.float32(contour).reshape(-1, 1, 2)
                    contour = cv2.perspectiveTransform(contour,self.hh_array[ctr]).squeeze()

                    p = Path(contour)  # make a polygon
                    grid = p.contains_points(points_mesh)
                    mask = grid.reshape(self.size[0],
                                        self.size[1])  # now

                    masks[ctr].append(np.float32(mask))
                    contours[ctr].append(contour)
                    allcell_masks[ctr]+= mask
                ctr+=1
            #

            self.masks = np.array(masks, dtype=object)
            self.contours = np.array(contours, dtype=object)
            self.all_cell_masks = np.array(allcell_masks, dtype=object)

            np.save(fname_out_masks, self.masks)
            np.save(fname_out_contours, self.contours)
            np.save(fname_out_allcell_masks, self.all_cell_masks)
        else:

            self.masks = np.load(fname_out_masks,allow_pickle=True)
            self.contours = np.load(fname_out_contours,allow_pickle=True)
            self.all_cell_masks = np.load(fname_out_allcell_masks,allow_pickle=True)



    def get_transformed_cell_masks(self):

        #
        x, y = np.meshgrid(np.arange(self.size[0]),
                           np.arange(self.size[1]))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points_mesh = np.vstack((x, y)).T

        #
        mask1 = np.zeros((self.size[0],
                          self.size[1]))
        mask2 = np.zeros((self.size[0],
                          self.size[1]))

        masks1 = []
        masks2 = []
        for ctr, session in enumerate(self.sessions):

            c = calcium.Calcium()
            c.root_dir = self.root_dir
            c.animal_id = self.animal_id
            c.session = session
            c.load_suite2p()

            cell_ids = np.arange(c.F.shape[0])
            for cell in tqdm(cell_ids, desc='processing session '+str(ctr)):

                #
                contour = c.get_footprint_contour(cell)

                # tranform second session contours
                if ctr == 1:
                    contour = np.float32(contour).reshape(-1, 1, 2)
                    contour = cv2.perspectiveTransform(contour,
                                                       self.M).squeeze()

                p = Path(contour)  # make a polygon
                grid = p.contains_points(points_mesh)
                mask = grid.reshape(self.size[0],
                                    self.size[1])  # now

                if ctr == 0:
                    mask1 += mask
                    masks1.append(mask)
                if ctr == 1:
                    mask2 += mask
                    masks2.append(mask)

        #
        self.masks1 = np.float32(masks1)
        self.masks2 = np.float32(masks2)

        #
        self.all_cell1_mask = np.clip(mask1, 0, 1)
        self.all_cell2_mask = np.clip(mask2, 0, 1)

    def get_match_array_list2(self, sessions):

        ########################################################
        ############# LOAD PAIRWISE MATRICES ###################
        ########################################################
        match_arrays = []
        for k in range(len(sessions)):
            match_arrays.append([])
            for p in range(len(sessions)):
                match_arrays[k].append([])

        #
        #if self.cell_boundary=='convex_hull':
        #    prefix = ''
        for k in range(0, len(sessions), 1):
            for p in range(0, len(sessions), 1):
                if k == p:
                    continue

                s1 = sessions[k]
                s2 = sessions[p]

                #
                try:
                    fname_out = os.path.join(self.root_dir, 'match_array_' + str(s1) + "_" +
                                             str(s2) + "_"+self.cell_boundary + '.npy')
                    match_array = np.load(fname_out)
                except:
                    fname_out = os.path.join(self.root_dir,
                                             'match_array_' + str(s2) + "_" + str(s1) +
                                             "_"+ self.cell_boundary+'.npy')
                    match_array = np.load(fname_out).T

                match_arrays[k][p] = match_array

        ########################################################
        ############ LOOP OVER SEQUENTIAL MATCHES ##############
        ########################################################
        n_ctr = 0
        final_arrays = []
        all_links = []
        for t in trange(len(sessions) - 1):

            #
            order1 = np.arange(len(sessions), dtype=np.int32)
            order2 = order1.copy()
            order1[0] = order2[t]
            order1[t] = order2[0]

            #
            for c1 in range(match_arrays[order1[0]][order1[1]].shape[0]):
                #print("c1: ", c1)
                links = np.zeros(len(sessions)) + np.nan
                links[0] = c1
                starting_depth = 0
                find_next_match2(match_arrays, starting_depth, c1, self.thresh, links, order1)

                #
                links = links[order1]
                final_arrays.append(links)
                if np.isnan(links).sum() == 0:
                    n_ctr += 1

            all_links.append(final_arrays)
        all_links = np.vstack(all_links)
        print ("all links: ", all_links.shape)
        print (all_links)
        all_links = np.unique(all_links, axis=0)

        #
        final_links = []
        for k in range(all_links.shape[0]):
            temp = all_links[k]
            if np.isnan(temp).sum() == 0:
                final_links.append(temp)

        final_links = np.vstack(final_links)

        print ("total # celsl found: ", final_links.shape[0])
        return (final_links)



    def get_match_array_list(self, sessions):

        ########################################################
        ############# LOAD PAIRWISE MATRICES ###################
        ########################################################
        match_arrays = []
        for k in range(0, len(sessions) - 1):
            s1 = sessions[k]
            s2 = sessions[k+1]

            #
            fname_out = os.path.join(self.root_dir, 'match_array_'+str(s1)+"_"+str(s2)+'.npy')
            match_array = np.load(fname_out)
            match_arrays.append(match_array)

        ########################################################
        ############ LOOP OVER SEQUENTIAL MATCHES ##############
        ########################################################
        n_ctr = 0
        final_arrays = []
        for c1 in range(match_arrays[0].shape[0]):
            links = np.zeros(len(sessions)) + np.nan
            links[0] = c1
            starting_depth = 0
            find_next_match(match_arrays, starting_depth, c1, self.thresh, links)

            #
            final_arrays.append(links)
            if np.isnan(links).sum() == 0:
                n_ctr += 1
                print (links)

        print("# of matching cells: ", n_ctr)

        return (np.vstack(final_arrays))
        #return (np.vstack(all_links))



    def plot_match_matrix_multi(self, sessions):


        #
        n_cells = []
        img = np.zeros((len(sessions),len(sessions)))
        for s1 in range(0, len(sessions)-1):
            for s2 in range(s1+1, len(sessions),1):
                fname_out = os.path.join('/media/cat/4TB/donato/DON-003343/match_array_'+str(s1)+
                                 "_"+str(s2)+'.npy')
                match_array = np.load(fname_out)
                if s1==0 and s2==1:
                    n_cells.append(match_array.shape[0])
                if s1==0:
                    n_cells.append(match_array.shape[1])
                idx = np.where(match_array>self.thresh)

                img[s1,s2] = idx[0].shape[0]/match_array.shape[0]


        plt.imshow(img)
        xticks = []
        for s3 in sessions:
            xticks.append(self.sessions[s3])

        plt.xticks(np.arange(len(sessions)), xticks, rotation=45,fontsize=10)
        plt.yticks(np.arange(len(sessions)), xticks, fontsize=10)
        clb = plt.colorbar()
        # clb = plt.colorbar()
        # clb.ax.tick_params(labelsize=8)
        clb.ax.set_title(' % cell overlap', fontsize=10)

        return n_cells

    def get_match_array_pairs(self, idx1, idx2):

        #
        fname_out = os.path.join(self.root_dir,
                                 self.animal_id,
                                 'match_array_'+str(idx1)+
                                 "_"+str(idx2)+"_"+self.cell_boundary+'.npy')
        if os.path.exists(fname_out):
            self.match_array = np.load(fname_out)
            return

        if self.parallel:

            ids_array = np.array_split(np.arange(len(self.masks[idx1])),self.n_cores)

            res = parmap.map(get_match_array_parallel,
                            ids_array, self.masks[idx1], self.masks[idx2],
                            pm_processes = self.n_cores,
                            pm_pbar = True)

            res = np.array(res)
            match_array = np.sum(res,axis=0)

        else:

            match_array = np.zeros((self.masks1.shape[0],
                                    self.masks2.shape[0]))

            #
            for c1 in trange(self.masks1.shape[0]):
                for c2 in range(self.masks2.shape[0]):
                    cell1 = self.masks1[c1]
                    cell2 = self.masks2[c2]

                    # check for at least 1 pixel overlap
                    if np.max(cell1 + cell2) < 2:
                        continue

                    #
                    res = find_percent_overlap(cell1, cell2)
                    match_array[c1, c2] = res

        self.match_array = match_array
        np.save(fname_out, match_array)

    def get_match_array(self):

        #
        if self.parallel:

            ids_array = np.array_split(np.arange(self.masks1.shape[0]),self.n_cores)
            res = parmap.map(get_match_array_parallel,
                       ids_array, self.masks1, self.masks2,
                       pm_processes = self.n_cores,
                       pm_pbar = True)

            print ("# res: ", len(res))
            res = np.array(res)
            print (res.shape)
            match_array = np.sum(res,axis=0)

        else:

            match_array = np.zeros((self.masks1.shape[0],
                                    self.masks2.shape[0]))

            #
            for c1 in trange(self.masks1.shape[0]):
                for c2 in range(self.masks2.shape[0]):
                    cell1 = self.masks1[c1]
                    cell2 = self.masks2[c2]

                    # check for at least 1 pixel overlap
                    if np.max(cell1 + cell2) < 2:
                        continue

                    #idx = np.where(cell1 > 0)
                    res = find_percent_overlap(cell1, cell2)
                    match_array[c1, c2] = res

        self.match_array = match_array

    #
    def make_plotting_data_pair(self, idx1, idx2, match_array):

        img = np.zeros((self.size[0],
                        self.size[1]))

        sess1_ctr = np.zeros((match_array.shape[0]))
        sess2_ctr = np.zeros((match_array.shape[1]))
        cell1_ids = []
        cell2_ids = []
        for c1 in trange(match_array.shape[0]):
            for c2 in range(match_array.shape[1]):
                temp = match_array[c1, c2]
                if temp >= self.thresh:
                    temp1 = self.masks[idx1][c1]
                    temp2 = self.masks[idx2][c2]
                    cell1_ids.append(c1)
                    cell2_ids.append(c2)
                    img += temp1
                    img += temp2 * 2
                    sess1_ctr[c1] = 1
                    sess2_ctr[c2] = 1

        # this is the image continaining sums of footprints
        self.both_cells_image_map = np.clip(img, 0, 4)

        #
        self.sess1_matching_cells = sess1_ctr
        self.sess2_matching_cells = sess2_ctr

        #
        self.cell1_ids = np.unique(cell1_ids)
        self.cell2_ids = np.unique(cell2_ids)


    def make_plotting_data(self):

        img = np.zeros((self.size[0],
                        self.size[1]))

        sess1_ctr = np.zeros((self.match_array.shape[0]))
        sess2_ctr = np.zeros((self.match_array.shape[1]))
        cell1_ids = []
        cell2_ids = []
        for c1 in range(self.match_array.shape[0]):
            for c2 in range(self.match_array.shape[1]):
                temp = self.match_array[c1, c2]
                if temp >= self.thresh:
                    temp1 = self.masks1[c1]
                    temp2 = self.masks2[c2]
                    cell1_ids.append(c1)
                    cell2_ids.append(c2)
                    img += temp1
                    img += temp2 * 2
                    sess1_ctr[c1] = 1
                    sess2_ctr[c2] = 1

        # this is the image continaining sums of footprints
        self.both_cells_image_map = np.clip(img, 0, 4)

        #
        self.sess1_matching_cells = sess1_ctr
        self.sess2_matching_cells = sess2_ctr

        #
        self.cell1_ids = np.unique(cell1_ids)
        self.cell2_ids = np.unique(cell2_ids)


    ########################################
    def plot_overlap_masks(self):

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        img = self.both_cells_image_map.copy()
        idx = np.where(img == 0)
        img[idx] = np.nan

        cmap_name = 'Set1'
        img = plt.imshow(img,
                         cmap=cmap_name)

        handles = []

        cmap = plt.get_cmap(cmap_name, 4)

        # manually define a new patch
        handles.append(mpatches.Patch(color=cmap(0), label='Cell #1'))
        handles.append(mpatches.Patch(color=cmap(1), label='Cell #2'))
        handles.append(mpatches.Patch(color=cmap(2), label='Two cells overlap'))
        handles.append(mpatches.Patch(color=cmap(3), label='Three or more cells overlap'))

        # plot the legend
        plt.legend(handles=handles,
                   fontsize=20,
                   loc='upper center')

        plt.xlim(0, 512)
        plt.ylim(512, 0)

        plt.suptitle("Min overlap between 2 cells (threshold): "+ str(self.thresh*100) +
                     "%\n # cells sess1: "+str(np.sum(self.sess1_matching_cells))+
                     " , of total cells: "+str(self.sess1_matching_cells.shape[0])+
                     " , # cells sess 2: "+str(np.sum(self.sess2_matching_cells))+
                    " , of total cells: " + str(self.sess2_matching_cells.shape[0]))

    #
    def plot_overlap_contours_pairs(self, sess1, sess2):
        plt.figure()
        plt.subplot(1,1,1)

        #
        self.plot_contours_transformed_session(sess1,self.cell1_ids,'red')

        #
        self.plot_contours_transformed_session(sess2, self.cell2_ids,'black')

        plt.xlim(0,512)
        plt.ylim(512,0)

        plt.suptitle("Sessions: "+str(self.sessions[sess1]) + "  " +str(self.sessions[sess2]) +
                     "\nMin overlap between 2 cells (threshold): "+ str(self.thresh*100) +
                     "%\n # cells sess1: "+str(np.sum(self.sess1_matching_cells))+
                     " , of total cells: "+str(self.sess1_matching_cells.shape[0])+
                     " , # cells sess 2: "+str(np.sum(self.sess2_matching_cells))+
                    " , of total cells: " + str(self.sess2_matching_cells.shape[0]))

    #
    def plot_overlap_contours(self):
        fig=plt.figure()
        ax=plt.subplot(1,1,1)

        #
        self.plot_contours_transformed_session_cell_ids(self.M,False,
                                               self.sessions[0],
                                               self.cell1_ids,
                                              'red')
        #
        self.plot_contours_transformed_session_cell_ids(self.M,True,
                                                   self.sessions[1],
                                                   self.cell2_ids,
                                                  'black')

        plt.xlim(0,512)
        plt.ylim(512,0)

        plt.suptitle("Min overlap between 2 cells (threshold): "+ str(self.thresh*100) +
                     "%\n # cells sess1: "+str(np.sum(self.sess1_matching_cells))+
                     " , of total cells: "+str(self.sess1_matching_cells.shape[0])+
                     " , # cells sess 2: "+str(np.sum(self.sess2_matching_cells))+
                    " , of total cells: " + str(self.sess2_matching_cells.shape[0]))

    #
    def plot_contours_transformed_session(self,
                                          sess,
                                          cell_ids,
                                          clr):
        for k in cell_ids:

            points = self.contours[sess][k]
            plt.plot(points[:, 0],
                     points[:, 1],
                     c=clr,
                     linewidth=2,
                     alpha=.7)

    #
    def plot_contours_transformed_session_cell_ids(self, M,
                                                   transform_flag,
                                                   session,
                                                   cell_ids,
                                                   clr):
        root_dir = '/media/cat/4TB/donato/'
        animal_id = 'DON-006084'

        c = calcium.Calcium()
        c.root_dir = root_dir
        c.animal_id = animal_id
        c.session = session
        c.load_suite2p()

        for k in cell_ids:
            points = c.get_footprint_contour(k)
            if transform_flag:
                points = points.astype('float32').reshape(-1, 1, 2)
                points = cv2.perspectiveTransform(points,
                                                  M).squeeze()
            plt.plot(points[:, 0],
                     points[:, 1],
                     c=clr,
                     linewidth=2,
                     alpha=.7)

#
def get_match_array_parallel_multi(ids1, masks_array):

    # assume < 2000 cells per session, and max 10 sessions
    match_array = np.zeros((2000,
                            2000,10))

    # match first session with 2nd session
    masks2 = masks_array[0]
    for c1 in ids1:
        cell1 = masks1[c1]
        for c2 in range(len(masks2)):
            cell2 = masks2[c2]

            # check for at least 1 pixel overlapping
            if np.max(cell1 + cell2) < 2:
                continue

            #


            #
            res = find_percent_overlap(cell1, cell2)
            match_array[c1, c2] = res

    return match_array

#
def get_match_array_parallel(ids1, masks1, masks2):

    match_array = np.zeros((len(masks1),
                            len(masks2)))

    #
    for c1 in ids1:
        for c2 in range(len(masks2)):
            cell1 = masks1[c1]
            cell2 = masks2[c2]
            #print (np.max(cell1), np.max(cell2))

            # check for at least 1 pixel overlapping
            if np.max(cell1 + cell2) < 2:
                continue

            #print (c1, c2, " have overlap")
            #idx = np.where(cell1 > 0)
            res = find_percent_overlap(cell1, cell2)
            match_array[c1, c2] = res

    return match_array

#
def find_next_match(match_arrays, depth, cell1, thresh, links):
    if depth == len(match_arrays):
        return
    #
    loc_array = match_arrays[depth]

    #
    idx = np.where(loc_array[cell1] >= thresh)[0]
    vals = loc_array[cell1][idx]

    #
    if idx.shape[0] > 0:
        cell2 = idx[0]  # for now just pick the first matching cell; sometimes there are 2 or more

        # if more than 1 match, select highest overlap
        if idx.shape[0]>1:
            #temp =
            cell2 = idx[np.argmax(vals)]
            #print ("alterantiv cell2 ", cell2, " overlaps: ", vals, links)

        links[depth + 1] = cell2
        find_next_match(match_arrays, depth + 1, cell2, thresh, links)

#
def find_next_match2(match_arrays, depth, cell1, thresh, links, order):
    if (depth + 1) >= len(match_arrays):
        return
    #
    #print('order', order, ' depth:', depth, 'len arrays', len(match_arrays))
    loc_array = match_arrays[order[depth]][order[depth + 1]]
    #print("loc_array: ", loc_array.shape, depth, cell1, links)

    #
    idx = np.where(loc_array[cell1] >= thresh)[0]
    vals = loc_array[cell1][idx]

    #
    if idx.shape[0] > 0:
        cell2 = idx[0]  # for now just pick the first matching cell; sometimes there are 2 or more

        # if more than 1 match, select highest overlap
        if idx.shape[0] > 1:
            # temp =
            cell2 = idx[np.argmax(vals)]
            # print ("alterantiv cell2 ", cell2, " overlaps: ", vals, links)

        links[depth + 1] = cell2
        find_next_match2(match_arrays, depth + 1, cell2, thresh, links, order)