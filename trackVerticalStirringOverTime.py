import os
import numpy as np
from scipy.ndimage import filters as ff
import csv
import pickle as p

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

num_x = 768

def get_alpha(args):
    i, frame = args

    print(i)

    period = 2.0 * np.pi

    output  = h5py.File("snapshots/snapshots_s%d.h5" % frame, mode = 'r')
    times    = output['scales']['sim_time'][:] / period
    vz_data    = output['tasks']["uz"]
    #vy_data    = output['tasks']["uy"]

    vx_vy = np.abs(vz_data[0][0] * vz_data[0][0])

    #print(np.shape(p_averaged))

    #start = i * num_x
    #end = start + num_x
    alpha_over_time[i] = np.average(vx_vy)

frame_range = util.get_frame_range(args.frames)
num_cores = args.num_cores

alpha_over_time = mp_array("d", len(frame_range))
pool_args = [(i, frame) for i, frame in enumerate(frame_range)]

if True:
  pool = Pool(num_cores)
  pool.map(get_alpha, pool_args)
  pool.terminate()

  alpha_array = np.array(alpha_over_time)
  #pressure_field = np.reshape(pressure_array, shape = (len(frame_range), num_x))
else:
  alpha_array = p.load(open("vertical-stirring-N10.p", "rb"))


#### PLOTTING ####

linewidth = 3
fontsize = 16

save_directory = "."
version = None
dpi = 100

#cmap = "inferno"
cmap = "seismic"

def make_plot(show = False):
    fig = plot.figure()
    ax = fig.add_subplot(111)

    output  = h5py.File("snapshots/snapshots_s1.h5", mode = 'r')
    times    = output['scales']['sim_time'][:]
    p_data    = output['tasks']["p"]
    xs   = p_data.dims[1][0][:]
    zs   = p_data.dims[2][0][:]

    x = np.array(frame_range) - 1
    y = alpha_array

    #print(np.shape(x), np.shape(y), np.shape(pressure_field))

    result = plot.plot(x, y, linewidth = 3, label = r"$N^2 = -0.10$")

    #p.dump(times, open("timeVS-N10.p", "wb"))
    p.dump(x, open("timeVS-N10.p", "wb"))
    p.dump(y, open("vertical-stirring-N10.p", "wb"))

    if False:
       y2 = p.load(open("vertical-stirring-N3.p", "rb"))
       y3 = p.load(open("vertical-stirring-N1.p", "rb"))
       result2 = plot.plot(x, y2, linewidth = 3, label = r"$N^2 = -0.03$")
       result3 = plot.plot(x, y3, linewidth = 3, label = r"$N^2 = -0.01$")

    plot.legend(loc = "upper right")

    # Axes
    plot.xlim(x[0], x[-1])
    plot.ylim(1e-5, 1.0)

    plot.yscale("log")

    plot.xlabel(r"$t - t_\mathrm{arbitrary}$", fontsize = fontsize)
    plot.ylabel(r"$\delta v_\mathrm{z}^2$", fontsize = fontsize)

    title2 = "Vertical Stirring"
    plot.title("%s" % (title2), y = 1.015, fontsize = fontsize + 1)

    # Save, Show, and Close
    cwd = os.getcwd().split("/")[-1]

    if version is None:
        save_fn = "%s/averagedVerticalStirringOverTime-%s.png" % (save_directory, cwd)
    else:
        save_fn = "%s/v%04d_/averagedVerticalStirringOverTime-%s.png" % (save_directory, version, cwd)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)

make_plot()

