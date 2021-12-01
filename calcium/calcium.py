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



        ############################################################
        ################## REMOVE NON-CELLS ########################
        ############################################################
        #
        idx = np.where(self.iscell[:,0]==1)[0]
        self.F = self.F[idx]
        self.Fneu = self.Fneu[idx]

        self.spks = self.spks[idx]
        self.stat = self.stat[idx]

        #############################################################
        ########### COMPUTE GLOBAL MEAN - REMOVE MEAN ###############
        #############################################################

        stds = np.std(self.F, axis=1)
        y = np.histogram(stds, bins=np.arange(0, 100, .5))
        argmax = np.argmax(y[0])
        self.std_global = y[1][argmax]

        if self.verbose:
            print ("        self.F (fluorescence): ", self.F.shape)
            print ("         self.Fneu (neuropile): ", self.Fneu.shape)
            print ("         self.iscell (cell classifier output): ", self.iscell.shape)
            print ("         # of good cells: ", np.where(self.iscell==1)[0].shape)
            print ("         self.ops: ", self.ops.shape)
            print ("         self.spks (deconnvoved spikes): ", self.spks.shape)
            print ("         self.stat (footprints?): ", self.stat.shape)
            print ("         mean std over all cells : ", self.std_global)

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

        #
        t = np.arange(traces.shape[1])/self.sample_rate

        for ctr, k in enumerate(ns):
            if color is None:
                plt.plot(t,traces[k], label=label,
                         alpha=alpha,
                         linewidth=linewidth)
            else:
                plt.plot(t,traces[k], label=label,
                         color=color,
                         alpha=alpha,
                         linewidth=linewidth)

        # print ("np mean: ", np.mean(traces[k]))

        #plt.ylabel("First 100 neurons")
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
        #
        traces_out = traces.copy()
        for k in trange(traces.shape[0]):
            #
            temp = traces[k]

            #
            temp = butter_lowpass_filter(temp,
                                         self.high_cutoff,
                                         self.sample_rate,
                                         order=1)
            #
            traces_out[k] = temp

        #
        return traces_out

    def load_binarization(self):

        #
        fname_out = os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session,
                                 'suite2p',
                                 'plane0',
                                 'binarized_traces.npz'
                                 )

        if os.path.exists(fname_out):
            data = np.load(fname_out, allow_pickle=True)
            self.F_onphase = data['F_onphase']
            self.F_upphase = data['F_upphase']
            self.spks = data['spks']
            self.spks_smooth_upphase = data['spks_smooth_upphase']

            # raw and filtered data;
            self.F_filtered = data['F_filtered']
            self.spks_x_F = data['oasis_x_F']

            # parameters saved to file as dictionary
            self.oasis_thresh_prefilter = data['oasis_thresh_prefilter']
            self.min_thresh_std_oasis = data['min_thresh_std_oasis']
            self.min_thresh_std_Fluorescence_onphase = data['min_thresh_std_Fluorescence_onphase']
            self.min_thresh_std_Fluorescence_upphase = data['min_thresh_std_Fluorescence_upphase']
            self.min_width_event_Fluorescence = data['min_width_event_Fluorescence']
            self.min_width_event_oasis = data['min_width_event_oasis']
            self.min_event_amplitude = data['min_event_amplitude']

            print (self.F_filtered.shape)

        else:
            self.binarize_fluorescence()

    def binarize_fluorescence(self):

        fname_out = os.path.join(self.root_dir,
                                 self.animal_id,
                                 self.session,
                                 'suite2p',
                                 'plane0',
                                 'binarized_traces.npz'
                                 )

        if os.path.exists(fname_out)==False or self.recompute:


            ####################################################
            ########### FILTER FLUROESCENCE TRACES #############
            ####################################################
            self.high_cutoff = 1
            self.low_cutoff = 0.1

            self.F_filtered = self.low_pass_filter(self.F)

            #self.F_filtered = self.band_pass_filter(self.F)  # This has too many artifacts
            self.F_filtered -= np.median(self.F_filtered, axis=1)[None].T

            ####################################################
            ###### BINARIZE FILTERED FLUORESCENCE ONPHASE ######
            ####################################################
            stds = np.ones(self.F_filtered.shape[0])+self.std_global
            #print (" std global: ", stds, " vs std local ", np.std(self.F))

            self.F_onphase_bin = self.binarize_onphase(self.F_filtered,
                                                       stds,
                                                       self.min_width_event_Fluorescence,
                                                       self.min_thresh_std_Fluorescence_onphase)

            # detect onset of ONPHASE to ensure UPPHASES overlap at least in one location with ONPHASE
            def detect_onphase(traces):
                a = traces.copy()
                locs = []
                for k in range(a.shape[0]):
                    idx = np.where((a[k][1:]-a[k][:-1])==1)[0]
                    locs.append(idx)

                locs = np.array(locs)
                return locs

            onphases = detect_onphase(self.F_onphase_bin)

            ####################################################
            ###### BINARIZE FILTERED FLUORESCENCE UPPHASE ######
            ####################################################
            # THIS STEP SOMETIMES MISSES ONPHASE COMPLETELY DUE TO GRADIENT;
            # So we minimally add onphases from above
            der = np.float32(np.gradient(self.F_filtered,
                                         axis=1))
            min_slope = 0
            idx = np.where(der <= min_slope)
            F_upphase = self.F_filtered.copy()
            F_upphase[idx]=0

            #
            self.F_upphase_bin = self.binarize_onphase(F_upphase,
                                                       stds,    # use the same std array as for full Fluorescence
                                                       self.min_width_event_Fluorescence//2,
                                                       self.min_thresh_std_Fluorescence_upphase)

            # make sure that upphase data has at least on the onphase start
            # some of the cells miss this
            for k in range(len(onphases)):
                idx = np.int32(onphases[k])
                for id_ in idx:
                    self.F_upphase_bin[k,id_:id_+2] = 1

            ############################################################
            ################## THRESHOLD RAW OASIS #####################
            ############################################################
            for k in range(self.spks.shape[0]):
                # idx = np.where(c.spks_filtered[k]<stds[k]*1.0)[0]
                idx = np.where(self.spks[k] < self.oasis_thresh_prefilter)[0]
                self.spks[k, idx] = 0

            ############################################################
            ################# SMOOTH OASIS SPIKES ######################
            ############################################################
            # self.spks_filtered = self.smooth_traces(self.spks, F_detrended)
            self.spks_filtered = self.smooth_traces(self.spks,
                                                    self.F_filtered)

            #################################################
            ##### SCALE OASIS BY FLUORESCENCE ###############
            #################################################
            self.spks_x_F = self.spks_filtered * self.F_filtered

            ############################################################
            ###### UPPHASE DETECTION ON OASIS SMOOTH/SCALED DATA #######
            ############################################################
            der = np.float32(np.gradient(self.spks_x_F,
                                         axis=1))
            #
            idx = np.where(der <= 0)

            #
            spks_upphase = self.spks_x_F.copy()
            spks_upphase[idx]=0

            # compute some measure of signal quality
            #sq = stds_all(spks_upphase)
            #print ('SQ: spikes :', sq)
            spks_upphase2 = spks_upphase.copy()
            spks_upphase2[idx] = np.nan
            stds = np.nanstd(spks_upphase2, axis=1)

            self.spks_upphase_bin = self.binarize_onphase(spks_upphase,
                                                          stds,
                                                          #self.min_width_event // 3,
                                                          self.min_width_event_oasis,
                                                          self.min_thresh_std_oasis)

            ############################################################
            ############# CUMULATIVE OASIS SCALING #####################
            ############################################################
            self.spks_smooth_bin = self.scale_binarized(self.spks_upphase_bin,
                                                         self.spks)

            #
            np.savez(fname_out,
                     # binarization data
                     F_onphase=self.F_onphase_bin,
                     F_upphase=self.F_upphase_bin,
                     spks=self.spks,
                     spks_smooth_upphase=self.spks_smooth_bin,

                     # raw and filtered data;
                     F_filtered=self.F_filtered,
                     oasis_x_F = self.spks_x_F,

                     # parameters saved to file as dictionary
                     oasis_thresh_prefilter=self.oasis_thresh_prefilter,
                     min_thresh_std_oasis=self.min_thresh_std_oasis,
                     min_thresh_std_Fluorescence_onphase=self.min_thresh_std_Fluorescence_onphase,
                     min_thresh_std_Fluorescence_upphase=self.min_thresh_std_Fluorescence_upphase,
                     min_width_event_Fluorescence=self.min_width_event_Fluorescence,
                     min_width_event_oasis=self.min_width_event_oasis,
                     min_event_amplitude=self.min_event_amplitude,
                     allow_pickle=True
                     )
        # else:
        #     data = np.load(fname_out, allow_pickle=True)
        #     self.F_onphase = data['F_onphase'],
        #     self.F_upphase = data['F_upphase'],
        #     self.spks = data['spks'],
        #     self.spks_smooth_upphase = data['spks_smooth_upphase'],
        #
        #     # raw and filtered data;
        #     self.F_filtered = data['F_filtered'],
        #     self.spks_x_F = data['oasis_x_F'],
        #
        #     # parameters saved to file as dictionary
        #     self.oasis_thresh_prefilter = data['oasis_thresh_prefilter'],
        #     self.min_thresh_std_oasis = data['min_thresh_std_oasis'],
        #     self.min_thresh_std_Fluorescence_onphase = data['min_thresh_std_Fluorescence_onphase'],
        #     self.min_thresh_std_Fluorescence_upphase = data['min_thresh_std_Fluorescence_upphase'],
        #     self.min_width_event_Fluorescence = data['min_width_event_Fluorescence'],
        #     self.min_width_event_oasis = data['min_width_event_oasis'],
        #     self.min_event_amplitude = data['min_event_amplitude']


    #
    def smooth_traces(self, traces, F_detrended):

        # params for savgol filter
        window_length = 11
        polyorder = 1
        deriv = 0
        delta = 1

        # params for exponential kernel
        M = 100
        tau = 100  # !3 sec decay
        d_exp = scipy.signal.exponential(M, 0, tau, False)
        #d_step = np.zeros(100)
        #d_step[25:75]=1

        #
        traces_out = traces.copy()

        # Smooth Oasis spikes first using savgolay filter + lowpass
        for k in trange(traces.shape[0]):
            temp = traces_out[k].copy()

            # savgol filter:
            if False:
                temp = scipy.signal.savgol_filter(temp,
                                               window_length,
                                               polyorder,
                                               deriv = deriv,
                                               delta = delta)
            # convolve with an expnential function
            else:
                temp = np.convolve(temp, d_exp, mode='full')[:temp.shape[0]]

            # if True:
            #
            #     temp =


            if True:
                temp = butter_lowpass_filter(temp, 2, 30)
            traces_out[k] = temp

        #
        # if False:
        #     # find best shift correlation between Smooth Oasis and Fluorescence
        #     # usign first 10 neurons -  which usually have highest SNR
        #     all_shifts = []
        #     for n in range(10):
        #         # shifts
        #         shifts = []
        #         for s in range(-60, 60, 1):
        #             temp1 = np.roll(traces_out[n],s)
        #
        #             temp2 = F_detrended[n]
        #
        #             shifts.append(np.correlate(temp1, temp2, mode='valid'))
        #
        #         all_shifts.append(shifts)
        #
        #     shifts = np.array(all_shifts).squeeze()
        #     maxes = np.argmax(shifts,axis=1)
        #
        #     # take correct shift as the median
        #     final_shift = int(np.mean(maxes))-60
        #
        #     # reshift the traces to match the data
        #     if True:
        #         for k in trange(traces_out.shape[0]):
        #             temp = np.roll(traces_out[k],
        #                                     final_shift)
        #             # zero out the rolled values
        #             #print ("temp: ", temp.shape)
        #
        #             temp[:final_shift*2] = 0
        #             temp[-final_shift*2:] = 0
        #
        #             #
        #             traces_out[k] = temp


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
                                         order=1)

            #
            traces_out[k] = temp

        #
        return traces_out

    def scale_binarized(self, traces, traces_scale):

        #
        from scipy.signal import chirp, find_peaks, peak_widths

        #
        for k in trange(traces.shape[0]):
            temp = traces[k].copy()

            #
            peaks, _ = find_peaks(temp)  # middle of the pluse/peak

            #
            widths, heights, starts, ends = peak_widths(temp, peaks)

            xys = np.int32(np.vstack((starts, ends)).T)

            # number of time steps after peak to add oasis spikes
            buffer = 5
            for t in range(xys.shape[0]):
                # peak = np.max(F[k,xys[t,0]:xys[t,1]])
                peak = np.sum(traces_scale[k,xys[t,0]:xys[t,1]+buffer])

                temp[xys[t,0]:xys[t,1]] *= peak
                if np.max(temp[xys[t,0]:xys[t,1]])<self.min_event_amplitude:
                    #print( "Evetn too small: ", np.max(temp[xys[t,0]:xys[t,1]]),
                          # xys[t,0], xys[t,1])
                    temp[xys[t,0]:xys[t,1]+1]=0

            traces[k] = temp

        return traces

    def binarize_onphase(self,
                         traces,
                         val_scale,
                         min_width_event,
                         min_thresh_std):
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

            # find threshold crossings standard deviation based
            val = val_scale[k]
            idx1 = np.where(temp>=val*min_thresh_std)[0]  # may want to use absolute threshold here!!!

            #
            temp = temp*0
            temp[idx1] = 1

            # FIND BEGINNIGN AND ENDS OF FLUORescence above some threshold
            from scipy.signal import chirp, find_peaks, peak_widths
            peaks, _ = find_peaks(temp)  # middle of the pluse/peak
            widths, heights, starts, ends = peak_widths(temp, peaks)

            #
            xys = np.int32(np.vstack((starts, ends)).T)
            idx = np.where(widths < min_width_event)[0]
            xys = np.delete(xys, idx, axis=0)

            traces_bin[k] = traces_bin[k]*0

            # fill the data with 1s
            buffer = 0
            for p in range(xys.shape[0]):
                traces_bin[k,xys[p,0]:xys[p,1]+buffer] = 1

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


    def compute_PCA(self, X, suffix=''):
        #

        # run PCA

        fname_out = os.path.join(self.root_dir, 'pca_'+suffix+'.pkl')

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


    def compute_UMAP(self, X, n_components = 3, text=''):
        #
        print(self.root_dir)

        fname_out = os.path.join(self.root_dir, text+'umap.npz')

        try:
            data = np.load(fname_out, allow_pickle=True)
            X_umap = data['X_umap']
        except:

            n_components = n_components
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



