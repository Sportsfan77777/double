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

with open('analysis.txt', 'r') as f:
   reader = csv.reader(f)
   data = list(reader)
   analysis = np.array(data, dtype = 'float')

#print np.shape(analysis)

smooth = lambda array, kernel_size : ff.gaussian_filter(array, kernel_size)

def get_energy(args):
    i, frame = args

    period = 2.0 * np.pi

    output  = h5py.File("snapshots/snapshots_s%d.h5" % frame, mode = 'r')
    times    = output['scales']['sim_time'][:] / period
    vx_data    = output['tasks']["ux"]
    vy_data    = output['tasks']["uy"]
    vz_data    = output['tasks']["uz"]

    vx_sq = np.multiply(vx_data, vx_data)
    vy_sq = np.multiply(vy_data, vy_data)
    vz_sq = np.multiply(vz_data, vz_data)
    energy_field = vx_sq + vy_sq + vz_sq

    ux_sq = np.average(np.average(np.average(vx_sq, axis = 0), axis = 0), axis = 0)
    uy_sq = np.average(np.average(np.average(vy_sq, axis = 0), axis = 0), axis = 0)
    uz_sq = np.average(np.average(np.average(vz_sq, axis = 0), axis = 0), axis = 0)
    energy = np.average(np.average(np.average(energy_field, axis = 0), axis = 0), axis = 0)
    
    energy_over_time[i] = energy
    ux_squared_over_time[i] = ux_sq
    uy_squared_over_time[i] = uy_sq
    uz_squared_over_time[i] = uz_sq

frame_range = util.get_frame_range(args.frames)
num_cores = args.num_cores

energy_over_time = mp_array("d", len(frame_range))
ux_squared_over_time = mp_array("d", len(frame_range))
uy_squared_over_time = mp_array("d", len(frame_range))
uz_squared_over_time = mp_array("d", len(frame_range))
pool_args = [(i, frame) for i, frame in enumerate(frame_range)]

p = Pool(num_cores)
p.map(get_energy, pool_args)
p.terminate()

#### PLOTTING ####

linewidth = 3
fontsize = 16

plot.figure()

#x = analysis[:, 0] #/ (2.0 * np.pi)
#y = analysis[:, 1]
#y2 = smooth(y, 1)
#y3 = np.diff(np.log(y2)) / np.diff(x)

#print min(y3), max(y3)

#growth = 0.05
#plot.plot([0, max(x)], [growth, growth], c = 'k', linewidth = linewidth)

x = frame_range
y = np.array(energy_over_time)
y2 = np.array(ux_squared_over_time)
y3 = np.array(uy_squared_over_time)
y4 = np.array(uz_squared_over_time)
plot.plot(x, y, c = "b", linewidth = linewidth, label = r"$|u|^2$")
plot.plot(x, y2, c = "r", linewidth = linewidth, label = r"$u_\mathrm{x}^2$")
plot.plot(x, y3, c = "r", linewidth = linewidth, label = r"$u_\mathrm{y}^2$")
plot.plot(x, y4, c = "r", linewidth = linewidth, label = r"$u_\mathrm{z}^2$")

plot.legend(loc = "upper left")

#plot.plot(x[:-1], y3, linewidth = linewidth)

#print(len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-8, 1.0)

plot.yscale('log')

plot.xlabel('t', fontsize = fontsize)
plot.ylabel(r'$<v^2>$', fontsize = fontsize)
plot.title('Angular Momentum Flux', fontsize = fontsize + 1)

cwd = os.getcwd().split("/")[-1]
plot.savefig("energy-%s.png" % cwd, bbox_inches = 'tight')
