
import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from scipy.ndimage import uniform_filter1d



### PLOTTING ###

fontsize = 24
dpi = 100
clim = [-0.001, 0.001]

nlev = 128
nclev = 6
cmap = plot.cm.inferno


if __name__ == "__main__":

    # Get data
    loc = "."
    frame = 50
    period = 2.0 * np.pi

    fn = "%s/snapshots/snapshots_s%d.h5" % (loc, frame)
    output  = h5py.File(fn, mode = 'r')
    t_arr    = output['scales']['sim_time'][:] / period

    var = "uz"
    data = output['tasks'][var]

    data3D = data[0]
    vertical_velocity = data3D

    print np.shape(data)
    print np.shape(data3D)

    taxis = t_arr[0]
    #dataXZ = data3D[:, 0, :]
    #data2D[i,:] = np.mean(dataXZ, axis = 1)

    xaxis   = data.dims[1][0][:]
    zaxis   = data.dims[2][0][:]

    # Figure
    # Set up figure
    fig = plot.figure(figsize = (7, 6), dpi = dpi)
    ax = fig.add_subplot(111)

    # Data
    rad = xaxis
    z_angles = zaxis

    x = rad
    y = (z_angles - np.pi / 2) * (180.0 / np.pi)
    result = ax.pcolormesh(x, y, np.transpose(vertical_velocity), cmap = cmap)

    cbar = fig.colorbar(result)
    result.set_clim(clim[0], clim[1])

    # Save, Show, and Close
    save_directory = "."
    version = None
    show = True

    if version is None:
        save_fn = "%s/verticalVelocityMap_%04d.png" % (save_directory, frame)
    else:
        save_fn = "%s/v%04d_verticalVelocityMap_%04d.png" % (save_directory, version, frame)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)

