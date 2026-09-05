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
from matplotlib import rcParams as rc

#from new_double_diffusive_latter10 import q as this_q
#from new_double_diffusive_latter10 import Reynolds as this_Re
#from new_double_diffusive_latter10 import big_lambda as this_lambda

import h5py
import util

###############################################################################

this_q = 1e-5
this_Re = 1e5
this_lambda = 1.0

###############################################################################

### Input Parameters ###

def new_argument_parser(description = "Plot gas density maps."):
    parser = argparse.ArgumentParser()

    # Frame Selection
    #parser.add_argument('frames', type = int, nargs = '+',
    #                     help = 'select single frame or range(start, end, rate). error if nargs != 1 or 3')
    #parser.add_argument('-c', dest = "num_cores", type = int, default = 1,
    #                     help = 'number of cores (default: 1)')

    return parser

### Parse Arguments ###
args = new_argument_parser().parse_args()

### Get Input Parameters ###

# Frames
#frame_range = util.get_frame_range(args.frames)

# Number of Cores 
#num_cores = args.num_cores

frame_range = p.load(open("timeVS-N10.p", "rb"))
alpha_array = p.load(open("vertical-stirring-N10.p", "rb"))

#### PLOTTING ####

linewidth = 3
fontsize = 20

save_directory = "."
version = None
dpi = 100

#cmap = "inferno"
cmap = "seismic"

labelsize = 18
rc['xtick.labelsize'] = labelsize
rc['ytick.labelsize'] = labelsize

def make_plot(show = False):
    fig = plot.figure()
    ax = fig.add_subplot(111)

    output  = h5py.File("snapshots/snapshots_s1.h5", mode = 'r')
    times    = output['scales']['sim_time'][:]
    p_data    = output['tasks']["p"]
    xs   = p_data.dims[1][0][:]
    zs   = p_data.dims[2][0][:]

    #p.dump(times, open("timeVS-N10.p", "wb"))
    #p.dump(alpha_array, open("vertical-stirring-N10.p", "wb"))

    x1 = np.array(frame_range)
    y1 = alpha_array

    print (np.shape(x1), np.shape(y1))

    #print(np.shape(x), np.shape(y), np.shape(pressure_field))
    x1 = x1 - x1[np.argmax(y1)]
    result = plot.plot(x1, y1, linewidth = 3, label = r"$N^2 = -0.10$")

    if True:
       x2 = p.load(open("timeVS-N3.p", "rb"))
       y2 = p.load(open("vertical-stirring-N3.p", "rb"))
       x2 = x2 - x2[np.argmax(y2)]

       max_i = np.argmax(x2)
       print(y2[max_i])

       x3 = p.load(open("timeVS-N1.p", "rb"))
       y3 = p.load(open("vertical-stirring-N1.p", "rb"))
       x3 = x3 - x3[np.argmax(y3)]

       result2 = plot.plot(x2, y2, linewidth = 3, label = r"$N^2 = -0.03$")
       result3 = plot.plot(x3, y3, linewidth = 3, label = r"$N^2 = -0.01$")

    plot.legend(loc = "upper right")

    # Axes
    plot.xlim(-50, x1[-1])
    plot.ylim(1e-5, 1.0)

    plot.yscale("log")

    plot.xlabel(r"$t - t_\mathrm{peak}$", fontsize = fontsize)
    plot.ylabel(r"$\delta v_\mathrm{z}^2$", fontsize = fontsize)

    title2 = "Vertical Stirring"
    plot.title("%s" % (title2), y = 1.015, fontsize = fontsize + 1)

    text_x = -40; text_y = 0.4; line_y = 0.45
    plot.text(text_x, text_y, r"$\Lambda = %.1f$" % this_lambda, horizontalalignment = 'left', fontsize = fontsize - 5)
    plot.text(text_x, text_y * line_y, "$q = %.1e$" % this_q, horizontalalignment = 'left', fontsize = fontsize - 5)
    plot.text(text_x, text_y * line_y * line_y, "$Re = %.1e$" % this_Re, horizontalalignment = 'left', fontsize = fontsize - 5)

    # Save, Show, and Close
    cwd = os.getcwd().split("/")[-1]

    if version is None:
        save_fn = "%s/averagedVerticalStirringOverTimeComparison-%s.png" % (save_directory, cwd)
    else:
        save_fn = "%s/v%04d_/averagedVerticalStirringOverTimeComparison-%s.png" % (save_directory, version, cwd)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)

make_plot()
