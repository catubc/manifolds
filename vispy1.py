# pyline: disable=no-member
""" plot3d using existing visuals : LinePlotVisual """

import numpy as np
import sys

from vispy import app, visuals, scene

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', 
                           title='plot3d', 
                           show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6

###############################
from scipy.io import loadmat
track = np.load('/media/cat/256GB/donato/DON-003343/DON-003343_20210222/wheel/20210222/TRD-2P/wheel.npy')

print (track.shape)
track_sum = np.cumsum(track)
print (track_sum.shape)
print (track_sum)

#
track_pos = np.float32(-track_sum)

#
n_cycles_per_rotation = 500
track_circular = (track_pos % n_cycles_per_rotation)*360/n_cycles_per_rotation


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

#
t = np.arange(track_circular.shape[0])/10000.

x, y = pol2cart(250, track_circular)

#
N = 50000000
x = x[:N]
y = y[:N]
z = t[:N]


# # prepare data
# N = 60
# x = np.sin(np.linspace(-2, 2, N)*np.pi)
# y = np.cos(np.linspace(-2, 2, N)*np.pi)
# z = np.linspace(-2, 2, N)

# plot
pos = np.c_[z, y, x]
Plot3D(pos, 
	   #width=2.0, 
	   color='blue',
       #edge_color='w', 
       #symbol='o', 
       #face_color=(0.2, 0.2, 1, 0.8),
       parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
