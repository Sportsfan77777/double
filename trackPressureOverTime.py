import os
import numpy as np
from scipy.ndimage import filters as ff
import csv

import argparse
from multiprocessing import Pool
from multiprocessing import Array as mp_array

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot

import h5py
import util

###############################################################################

### Input Parameters ###

def new_argument_parser(description = "Plot gas density maps."):
    parser = argparse.ArgumentParser()

    # Frame Selection
    parser.add_argument('frames', type = int, nargs = '+',
                         help = 'select single frame or range(start, end, rate). error if nargs != 1 or 3')
    parser.add_argument('-c', dest = "num_cores", type = int, default = 1,
                         help = 'number of cores (default: 1)')

    return parser

### Parse Arguments ###
args = new_argument_parser().parse_args()

### Get Input Parameters ###

# Frames
frame_range = util.get_frame_range(args.frames)

# Number of Cores 
num_cores = args.num_cores

num_x = 256

def get_pressure(args):
    i, frame = args

    period = 2.0 * np.pi

    output  = h5py.File("snapshots/snapshots_s%d.h5" % frame, mode = 'r')
    times    = output['scales']['sim_time'][:] / period
    p_data    = output['tasks']["p"]

    p_averaged = np.average(vx_sq, axis = 0)
    
    start = i * num_x
    end = start + num_x
    pressure_over_time[start:end] = p_averaged

frame_range = util.get_frame_range(args.frames)
num_cores = args.num_cores

pressure_over_time = mp_array("d", num_x * len(frame_range))
pool_args = [(i, frame) for i, frame in enumerate(frame_range)]

p = Pool(num_cores)
p.map(get_pressure, pool_args)
p.terminate()

pressure_array = np.array(pressure_over_time)
pressure_field = np.reshape(pressure_array, shape = (len(frame_range), num_x))

#### PLOTTING ####

linewidth = 3
fontsize = 16

def make_plot(show = False):
    plot.figure()

    xs   = vx_data.dims[1][0][:]
    zs   = vx_data.dims[2][0][:]

    x = frame_range
    y = xs
    result = ax.pcolormesh(x, y, np.transpose(pressure_field), cmap = cmap)

    cbar = fig.colorbar(result)
    result.set_clim(clim[0], clim[1])

    # Axes
    plot.xlabel(r"$t$", fontsize = fontsize)
    plot.ylabel(r"$x$", fontsize = fontsize)

    title2 = "Pressure over time"
    plot.title("%s" % (title2), y = 1.015, fontsize = fontsize + 1)

    cbar.set_label(r"$P$", fontsize = fontsize, rotation = 270, labelpad = 25)

    # Save, Show, and Close
    cwd = os.getcwd().split("/")[-1]

    if version is None:
        save_fn = "%s/averagedPressureOverTime-%s.png" % (save_directory, cwd)
    else:
        save_fn = "%s/v%04d_/averagedPressureOverTime-%s.png" % (save_directory, version, cwd)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)

plot.savefig("pressure-over-time-%s.png" % cwd, bbox_inches = 'tight')
