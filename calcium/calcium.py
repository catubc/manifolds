import numpy as np
import os
from tqdm import trange, tqdm
from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import umap
from sklearn.decomposition import PCA
import pickle as pk
import scipy

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

#
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def run_UMAP(data,
             n_neighbors=50,
             min_dist=0.1,
             n_components=3,
             metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )

    u = fit.fit_transform(data)

    return u

#
class Calcium():

    def __init__(self):
        self.sample_rate = 30
        #print ("Sample rate: ", self.sample_rate, "hz")

        self.verbose = False

        #
        self.keep_plot = True

    #
    def load_calcium(self):

        self.calcium_data = np.load(self.fname)

    #
    def load_suite2p(self):
        print ('')
        print ('')

        self.F = np.load(os.path.join(self.root_dir,
                                      self.animal_id,
                                      self.session,
                                      'suite2p',
                                      'plane0',
                                      'F.npy'), allow_pickle=True)
        self.Fneu = np.load(os.path.join(self.root_dir,
                                         self.animal_id,
                                         self.session,
                                         'suite2p',
                                         'plane0',
                                      'Fneu.npy'), allow_pickle=True)

        self.iscell = np.load(os.path.join(self.root_dir,
                                           self.animal_id,
                                           self.session,
                                           'suite2p',
                                           'plane0',
                                      'iscell.npy'), allow_pickle=True)


        self.ops = np.load(os.path.join(self.root_dir,
                                        self.animal_id,
                                        self.session,
                                        'suite2p',
                                        'plane0',
                                      'ops.npy'), allow_pickle=True)

        self.spks = np.load(os.path.join(self.root_dir,
                                         self.animal_id,
                                         self.session,
                                         'suite2p',
                                         'plane0',
                                      'spks.npy'), allow_pickle=True)

        self.stat = np.load(os.path.join(self.root_dir,
                                         self.animal_id,
                                         self.session,
                                         'suite2p',
                                         'plane0',
                                      'stat.npy'), allow_pickle=True)

        self.session_dir = os.path.join(self.root_dir,
                                   self.animal_id,
                                   self.session,
                                   'suite2p',
                                   'plane0')

        if self.verbose:
            print ("        self.F (fluorescence): ", self.F.shape)
            print ("         self.Fneu (neuropile): ", self.Fneu.shape)
            print ("         self.iscell (cell classifier output): ", self.iscell.shape)
            print ("         self.ops: ", self.ops.shape)
            print ("         self.spks (deconnvoved spikes): ", self.spks.shape)
            print ("         self.stat (footprints?): ", self.stat.shape)

        ############################################################
        ################## REMOVE NON-CELLS ########################
        ############################################################
        #
        idx = np.where(self.iscell[:,0]==1)[0]
        self.F = self.F[idx]
        self.Fneu = self.Fneu[idx]

        self.spks = self.spks[idx]
        self.stat = self.stat[idx]



    #
    def standardize(self, traces):

        fname_out = os.path.join(self.root_dir,
                                 'standardized.npy')

        if True:
            #os.path.exists(fname_out)==False:
            traces_out = traces.copy()
            for k in trange(traces.shape[0]):

                temp = traces[k]
                temp -= np.median(temp)
                temp = (temp)/(np.max(temp)-np.min(temp))
        #

                #temp -= np.median(temp)
                traces_out[k] = temp



        #     np.save(fname_out, traces_out)
        # else:
        #     traces_out = np.load(fname_out)

        return traces_out

    #
    def plot_traces(self, traces, ns,
                    label='',
                    color=None,
                    alpha=1.0,
                    linewidth=1.0):

        if self.keep_plot==False:
            plt.figure()


        t = np.arange(traces.shape[1])/self.sample_rate

        for ctr, k in enumerate(ns):
            if color is None:
                plt.plot(t,traces[k]+1*ctr, label=label,
                         alpha=alpha,
                         linewidth=linewidth)
            else:
                plt.plot(t,traces[k]+1*ctr, label=label,
                         color=color,
                         alpha=alpha,
                         linewidth=linewidth)


                #plt.plot(t, traces[k] + 1 * ctr)
            #print (k, np.median(traces[k]))

        plt.ylabel("First 100 neurons")
        plt.xlabel("Time (sec)")
        #plt.yticks([])
        plt.xlim(t[0],t[-1])
        #plt.show()

    #
    def plot_raster(self, ax, bn, galvo_times, track_times):

        # get front padding of raster:
        start_DC = galvo_times[0]/10000.

        # convert to frame time
        start_DC_frame_time = int(start_DC*self.sample_rate)
        print ("front padding: ", start_DC , "sec", start_DC_frame_time , ' in frame times')
        start_pad = np.zeros((bn.shape[0], start_DC_frame_time))

        # get end padding of raster
        end_DC = track_times[-1] - galvo_times[-1]/10000

        # convert to frame time
        end_DC_frame_time = int(end_DC * self.sample_rate)
        print ("end padding: ", end_DC , "sec", end_DC_frame_time , ' in frame times')
        end_pad = np.zeros((bn.shape[0], end_DC_frame_time))

        # padd the image with white space
        bn = np.hstack((start_pad, bn))
        bn = np.hstack((bn, end_pad))


        #
        ax.imshow(bn,
                   aspect='auto', cmap='Greys',
                   interpolation='none')

        # ax1.xlabel("Imaging frame")




        ax.set_ylabel("Neuron", fontsize=20)
        ax.set_xticks([])
        ax.set_ylim(0,bn.shape[0])


    def low_pass_filter(self, traces):

        #print (


        #
        traces_out = traces.copy()
        for k in trange(traces.shape[0]):
            #
            temp = traces[k]

            #
            temp = butter_lowpass_filter(temp,
                                         self.high_cutoff,
                                         self.sample_rate,
                                         order=4)

            #
            traces_out[k] = temp

        #
        return traces_out

    def binarize_fluorescence(self):

        fname_out = os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session,
                                 'suite2p',
                                 'plane0',
                                 'binarized_traces.npz'
                                 )

        if os.path.exists(fname_out):
            return

        #################################################
        ########### DETREND - REMOVE MEAN ###############
        #################################################
        F_detrended = self.F - np.mean(self.F, axis=0)


        ############################################################
        #### SMOOTH OASIS SPIKES WITH 4X LINEAR SAVGOY FILTER ######
        ############################################################
        self.spks_filtered = self.savgol_filter(self.spks, F_detrended)


        ############################################################
        ######## ZERO SPIKING ACTIVITY BELOW 1X STD THRESHOLD ######
        ############################################################
        #
        for k in range(self.spks_filtered.shape[0]):
            idx = np.where(self.spks_filtered[k] < self.oasis_thresh)[0]
            self.spks_filtered[k, idx] = 0


        ############################################################
        ################## BINARIZE > THRESHOLD ####################
        ############################################################
        self.F_bn2 = self.binarize_onphase(self.spks_filtered,
                                           self.F_minwidth)


        ############################################################
        ########## COMPUTE GRADIENTS TO GRAB ONPHASE ONLY ##########
        ############################################################
        der = np.float32(np.gradient(self.spks_filtered, axis=1))
        idx = np.where(der < 0)
        der[idx] = 0

        # multiply gradient with binarized to get
        spks_binarized_onphase = np.zeros(self.F_bn2.shape)
        for k in trange(spks_binarized_onphase.shape[0]):
            spks_binarized_onphase[k] = self.F_bn2[k] * der[k]

        # rebinarize the product
        self.F_bn_onphase = self.binarize_onphase(spks_binarized_onphase,
                                                  self.oasis_onphase_minwidth)


        ############################################################
        ####################### PEAKS ONLY #########################
        ############################################################

        # argrelmax only
        spks_argrelmax = np.zeros(self.spks_filtered.shape)

        #
        argrelmax_width = self.peak_width  # window to search for peaks within
        for k in range(spks_argrelmax.shape[0]):
            temp = self.spks_filtered[k].copy()*self.F_bn2[k]
            peaks = scipy.signal.argrelmax(temp, axis=0, order=argrelmax_width)

            #
            peaks = peaks[0]
            for p in range(peaks.shape[0]):
                spks_argrelmax[k, peaks[p] - 5:peaks[p] + 5] = 1

        #
        np.savez(fname_out,
                 binarized_onphase = self.F_bn_onphase,
                 binarized_peak = spks_argrelmax)


    #
    def savgol_filter(self, traces, F_detrended):

        #
        window_length = 31
        polyorder = 1
        deriv = 0
        delta = 1
        # n_loops = 1
        traces_out = traces.copy()

        # SMooth Oasis spikes first using savgolay filter + lowpass
        for k in trange(traces.shape[0]):
            temp = traces_out[k].copy()

            #
            temp = scipy.signal.savgol_filter(temp,
                                               window_length,
                                               polyorder,
                                               deriv = deriv,
                                               delta = delta)

            temp = butter_lowpass_filter(temp, 1, 30)
            traces_out[k] = temp

        # find best shift correlation between Smooth Oasis and Fluorescence
        # usign first 10 neurons -  which usually have highest SNR
        all_shifts = []
        for n in range(10):
            # shifts
            shifts = []
            for s in range(-60, 60, 1):
                temp1 = np.roll(traces_out[n],s)

                temp2 = F_detrended[n]

                shifts.append(np.correlate(temp1, temp2, mode='valid'))

            all_shifts.append(shifts)

        shifts = np.array(all_shifts).squeeze()
        maxes = np.argmax(shifts,axis=1)

        # take correct shift as the median
        final_shift = int(np.mean(maxes))-60

        # reshift the traces to match the data
        if True:
            for k in trange(traces_out.shape[0]):
                temp = np.roll(traces_out[k],
                                        final_shift)
                # zero out the rolled values
                #print ("temp: ", temp.shape)

                temp[:final_shift*2] = 0
                temp[-final_shift*2:] = 0

                #
                traces_out[k] = temp


        return traces_out



    #
    def band_pass_filter(self, traces):

        #print (

        #
        traces_out = traces.copy()
        for k in trange(traces.shape[0]):
            #
            temp = traces[k]

            #
            temp = butter_bandpass_filter(temp,
                                         self.low_cutoff,
                                         self.high_cutoff,
                                         self.sample_rate,
                                         order=4)

            #
            traces_out[k] = temp

        #
        return traces_out

    def binarize_onphase(self, traces,
                         min_width):
        '''
           Function that converts continuous float value traces to
           zeros and ones based on some threshold

           Here threshold is set to standard deviation /10.

            Retuns: binarized traces
        '''

        #
        traces_bin = traces.copy()

        #
        for k in trange(traces.shape[0]):
            temp = traces[k].copy()

            std = np.std(temp,axis=0)

            # find threshold crossings
            idx1 = np.where(temp>=std/5.)[0]  # may want to use absolute threshold here!!!
            #idx1 = np.where(temp>1)[0]

            temp = temp*0
            temp[idx1] = 1

            # FIND BEGINNIGN AND ENDS OF FLUORescence above some threshold
            from scipy.signal import chirp, find_peaks, peak_widths

            #
            peaks, _ = find_peaks(temp)  # middle of the pluse/peak

            #
            widths, heights, starts, ends = peak_widths(temp, peaks)

            xys = np.int32(np.vstack((starts, ends)).T)
            idx = np.where(widths < min_width)[0]
            xys = np.delete(xys, idx, axis=0)
            # print ("xys clean: ", xys.shape)
            traces_bin[k] = traces_bin[k]*0

            # fill missing with 1s
            for p in range(xys.shape[0]):
                traces_bin[k,xys[p,0]:xys[p,1]] = 1



        return traces_bin

        #
    def binarize_derivative(self, traces, thresh=2):

        fname_out = os.path.join(self.root_dir,
                                 'binarized_derivative.npy')
        if True:
        # if os.path.exists(fname_out) == False:

            #
            traces_out = traces.copy() * 0
            traces_out_anti_aliased = traces.copy() * 0  # this generates minimum of 20 time steps for better vis

            #
            for k in trange(traces.shape[0]):
                temp = traces[k]

                #std = np.std(temp)
                #idx = np.where(temp >= std * thresh)[0]

                #traces_out[k] = 0
                #traces_out[k, idx] = 1

                grad = np.gradient(temp)
                traces_out[k] = grad


                #
                # for id_ in idx:
                #     traces_out_anti_aliased[k, id_:id_ + 20] = 1
                #     if k > 0:
                #         traces_out_anti_aliased[k - 1, id_:id_ + 20] = 1

            np.save(fname_out, traces_out)
        else:
            traces_out = np.load(fname_out)
            traces_out_anti_aliased = traces_out.copy()

            # clip aliased data back down
            # traces_aliased = np.clip(traces_out_anti_aliased, 0,1)

        return traces_out, traces_out_anti_aliased


    #
    def binarize(self, traces, thresh = 2):

        fname_out = os.path.join(self.root_dir,
                                 'binarized.npy')
        if os.path.exists(fname_out)==False:
            traces_out = traces.copy()*0
            traces_out_anti_aliased = traces.copy()*0  # this generates minimum of 20 time steps for better vis
            for k in trange(traces.shape[0]):
                temp = traces[k]

                std = np.std(temp)

                idx = np.where(temp>=std*thresh)[0]

                traces_out[k] = 0
                traces_out[k,idx] = 1

                for id_ in idx:
                    traces_out_anti_aliased[k,id_:id_+20] = 1
                    if k>0:
                        traces_out_anti_aliased[k-1,id_:id_+20] = 1

            np.save(fname_out, traces_out)
        else:
            traces_out = np.load(fname_out)
            traces_out_anti_aliased = traces_out.copy()

            # clip aliased data back down
            # traces_aliased = np.clip(traces_out_anti_aliased, 0,1)

        return traces_out, traces_out_anti_aliased


    def compute_PCA(self, X):
        #

        # run PCA

        fname_out = os.path.join(self.root_dir, 'pca.pkl')

        if os.path.exists(fname_out)==False:

            pca = PCA()
            X_pca = pca.fit_transform(X)


            pk.dump(pca, open(fname_out, "wb"))

            #
            np.save(fname_out.replace('pkl','npy'), X_pca)
        else:
            with open(fname_out, 'rb') as file:
                pca = pk.load(file)

            X_pca = np.load(fname_out.replace('pkl','npy'))

        return pca, X_pca



    def compute_TSNE(self, X):
        #
        #print(self.root_dir)

        fname_out = os.path.join(self.root_dir, 'tsne.npz')
        #print ("Fname out: ", fname_out)

        try:
            data = np.load(fname_out, allow_pickle=True)
            X_tsne_gpu = data['X_tsne_gpu']

        except:

            n_components = 2
            perplexity = 100
            learning_rate = 10

            #
            X_tsne_gpu = TSNE(n_components=n_components,
                              perplexity=perplexity,
                              learning_rate=learning_rate).fit_transform(X)

            np.savez(fname_out,
                     X_tsne_gpu=X_tsne_gpu,
                     n_components=n_components,
                     perplexity=perplexity,
                     learning_rate=learning_rate
                     )


        return X_tsne_gpu


    def compute_UMAP(self, X):
        #
        print(self.root_dir)

        fname_out = os.path.join(self.root_dir, 'umap.npz')

        try:
            data = np.load(fname_out, allow_pickle=True)
            X_umap = data['X_umap']
        except:

            n_components = 2
            min_dist = 0.1
            n_neighbors = 50
            metric = 'euclidean'

            #
            X_umap = run_UMAP(X,
                              n_neighbors,
                              min_dist,
                              n_components,
                              metric)

            np.savez(fname_out,
                     X_umap=X_umap,
                     n_components=n_components,
                     min_dist=min_dist,
                     n_neighbors=n_neighbors,
                     metric=metric
                     )

        return X_umap


    def find_sequences(self, data, thresh=1):

        #
        segs = []
        seg = []
        seg.append(0)

        clrs = []
        ctr = 0
        clrs.append(ctr)

        #
        for k in trange(1, data.shape[0], 1):

            temp = dist = np.linalg.norm(data[k] - data[k - 1])

            if temp <= thresh:
                seg.append(k)
                clrs.append(ctr)
            else:
                segs.append(seg)
                seg = []
                seg.append(k)

                #
                ctr = np.random.randint(10000)
                clrs.append(ctr)

        # add last segment if missed:
        if len(segs[-1]) > 1:
            segs.append(seg)

        return segs, clrs



