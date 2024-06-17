
"""
dedalus plot vx map: (z vs x)
"""


import sys, os, subprocess
import pickle, glob
from multiprocessing import Pool
import argparse

import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams as rc
from matplotlib import pyplot as plot

import h5py
import util

#from colormaps import cmaps
#for key in cmaps:
#    plot.register_cmap(name = key, cmap = cmaps[key])

###############################################################################

### Input Parameters ###

def new_argument_parser(description = "Plot gas density maps."):
    parser = argparse.ArgumentParser()

    # Frame Selection
    parser.add_argument('frames', type = int, nargs = '+',
                         help = 'select single frame or range(start, end, rate). error if nargs != 1 or 3')
    parser.add_argument('-c', dest = "num_cores", type = int, default = 1,
                         help = 'number of cores (default: 1)')

    # Files
    parser.add_argument('--dir', dest = "save_directory", default = "radialVelocityMaps",
                         help = 'save directory (default: radialVelocityMaps)')

    # Plot Parameters (variable)
    parser.add_argument('--hide', dest = "show", action = 'store_false', default = True,
                         help = 'for single plot, do not display plot (default: display plot)')
    parser.add_argument('-v', dest = "version", type = int, default = None,
                         help = 'version number (up to 4 digits) for this set of plot parameters (default: None)')

    parser.add_argument('--range', dest = "r_lim", type = float, nargs = 2, default = None,
                         help = 'radial range in plot (default: [r_min, r_max])')

    # Plot Parameters (contours)
    parser.add_argument('--contour', dest = "use_contours", action = 'store_true', default = False,
                         help = 'use contours or not (default: do not use contours)')
    parser.add_argument('--low', dest = "low_contour", type = float, default = 1.1,
                         help = 'lowest contour (default: 1.1)')
    parser.add_argument('--high', dest = "high_contour", type = float, default = 3.5,
                         help = 'highest contour (default: 3.5)')
    parser.add_argument('--num_levels', dest = "num_levels", type = int, default = None,
                         help = 'number of contours (choose this or separation) (default: None)')
    parser.add_argument('--separation', dest = "separation", type = float, default = 0.1,
                         help = 'separation between contours (choose this or num_levels) (default: 0.1)')
    
    # Plot Parameters (rarely need to change)
    parser.add_argument('--cmap', dest = "cmap", default = "seismic",
                         help = 'color map (default: seismic)')
    parser.add_argument('--cmax', dest = "cmax", type = float, default = 0.0002,
                         help = 'maximum radial velocity in colorbar (default: 0.2)')

    parser.add_argument('--fontsize', dest = "fontsize", type = int, default = 16,
                         help = 'fontsize of plot annotations (default: 16)')
    parser.add_argument('--dpi', dest = "dpi", type = int, default = 100,
                         help = 'dpi of plot annotations (default: 100)')

    return parser

###############################################################################

### Parse Arguments ###
args = new_argument_parser().parse_args()

### Get Input Parameters ###

# Frames
frame_range = util.get_frame_range(args.frames)

# Number of Cores 
num_cores = args.num_cores

# Files
save_directory = args.save_directory
if not os.path.isdir(save_directory):
    os.mkdir(save_directory) # make save directory if it does not already exist

# Plot Parameters (variable)
show = args.show

#rad = np.linspace(r_min, r_max, num_rad)
#theta = np.linspace(0, 2 * np.pi, num_theta)

version = args.version
if args.r_lim is None:
    pass #x_min = r_min; x_max = r_max
else:
    x_min = args.r_lim[0]; x_max = args.r_lim[1]

# Plot Parameters (contours)
use_contours = args.use_contours
low_contour = args.low_contour
high_contour = args.high_contour
num_levels = args.num_levels
if num_levels is None:
    separation = args.separation
    num_levels = int(round((high_contour - low_contour) / separation + 1, 0))

# Plot Parameters (constant)
cmap = args.cmap
clim = [-args.cmax, args.cmax]

fontsize = args.fontsize
dpi = args.dpi

###############################################################################

##### PLOTTING #####

alpha = 0.7

labelsize = 18
rc['xtick.labelsize'] = labelsize
rc['ytick.labelsize'] = labelsize

def make_plot(frame, show = False):
    # Set up figure
    fig = plot.figure(figsize = (7, 6), dpi = dpi)
    ax = fig.add_subplot(111)

    # Read data
    #eps, va2 = disk_info
    #stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info

    period = 2.0 * np.pi

    var = "ux"
    output  = h5py.File("snapshots/snapshots_s%d.h5" % frame, mode = 'r')
    times    = output['scales']['sim_time'][:] / period
    vx_data    = output['tasks'][var]

    xs   = vx_data.dims[1][0][:]
    zs   = vx_data.dims[2][0][:]

    #for m in range(0, snapshots_per_set):
    vx3D      = vx_data[0]
    #times[i]    = time[0]
    vxXZ      = vx3D[:,:]
    #data2D[i,:] = np.mean(vxXZ, axis = 1)

    velocity = vxXZ

    ### Plot ###
    x = xs
    y = zs
    result = ax.pcolormesh(x, y, np.transpose(velocity), cmap = cmap)

    cbar = fig.colorbar(result)
    result.set_clim(clim[0], clim[1])

    # Axes
    plot.xlabel(r"$x$", fontsize = fontsize)
    plot.ylabel(r"$z$", fontsize = fontsize)

    title2 = r"$t = %d$ $\mathrm{orbits}$" % (frame)
    plot.title("%s" % (title2), y = 1.015, fontsize = fontsize + 1)

    cbar.set_label(r"$vx$", fontsize = fontsize, rotation = 270, labelpad = 25)

    # Save, Show, and Close
    if version is None:
        save_fn = "%s/radialVelocityMap_%04d.png" % (save_directory, frame)
    else:
        save_fn = "%s/v%04d_radialVelocityMap_%04d.png" % (save_directory, version, frame)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)


##### Make Plots! #####

# Iterate through frames

if len(frame_range) == 1:
    make_plot(frame_range[0], show = show)
else:
    if num_cores > 1:
        p = Pool(num_cores) # default number of processes is multiprocessing.cpu_count()
        p.map(make_plot, frame_range)
        p.terminate()
    else:
        for frame in frame_range:
            make_plot(frame)
