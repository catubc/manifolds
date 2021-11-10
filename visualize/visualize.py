import numpy as np
import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd


class Visualize():

    def __init__(self):
        self.keep_plot = False

    def plot_traces(self, traces, n=100):

        if self.keep_plot==False:
            plt.figure()

        t = np.arange(traces.shape[1] ) /self.sample_rate
        for k in trange(n):
            plt.plot(t ,traces[k ] + 1 *k)

        plt.ylabel("First 100 neurons")
        plt.xlabel("Time (sec)")
        # plt.yticks([])
        plt.xlim(t[0] ,t[-1])
        plt.show()

    def plot_raster(self, ax, rasters, time_stamps):



        ax1.imshow(bn,
                   aspect='auto', cmap='Greys',
                   interpolation='none')

        # ax1.xlabel("Imaging frame")
        ax1.set_ylabel("Neuron", fontsize=20)
        ax1.set_xticks([])


    def plot_scatter_3d(self,
                     data,
                     clrs=None,
                     title='',
                     cmap='viridis',
                     rows=1,
                     cols=1
                     ):

        #
        if clrs is None:
            clrs = np.arange(data.shape[0])

        #

        #
        if data.shape[1] == 2:

            fig = make_subplots(rows=1, cols=1)
            #
            df = pd.DataFrame(data[:, :2],
                              columns=["DIM1", 'DIM2'])
            print (df)
            fig = px.scatter(
                df, x='DIM1', y='DIM2',
                color=clrs,
                width=1200,
                height=750
            )

            fig.layout.coloraxis.colorbar.title = title
            fig.layout.coloraxis.colorscale = cmap
            # fig.update_layout(height=800, width=1400)
            #
            fig.layout.coloraxis.colorbar.title = title
            fig.layout.coloraxis.colorscale = cmap
            #fig.show()


        #
        else:

            fig = make_subplots(rows=1, cols=1,
                                specs=[[{'type': 'scene'}]])

            # df = pd.DataFrame(data[:, :3],
            #                   columns=["DIM1", 'DIM2', "DIM3"])
            fig.add_traces([go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                marker=dict(size=2,
                            color=clrs,
                            colorscale=cmap,
                            opacity=1,
                            colorbar=dict(thickness=25,
                                          title=title,
                                          # x=cols/10+2,
                                          outlinewidth=0

                                          )

                            )
            ),

            ],

                rows=rows,
                cols=cols)

            fig.layout.coloraxis.colorbar.title = title
            fig.layout.coloraxis.colorscale = cmap
            fig.update_layout(height=800, width=1400)

            fig.layout.coloraxis.colorbar.title = title
            fig.layout.coloraxis.colorscale = cmap
            #fig.show()

        return fig