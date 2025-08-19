import os
import numpy as np
from scipy.ndimage import filters as ff
import csv

from multiprocessing import Pool
from multiprocessing import Array as mp_array

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot

###############################################################################

### Input Parameters ###

def new_argument_parser(description = "Plot gas density maps."):
    parser = argparse.ArgumentParser()

    # Frame Selection
    parser.add_argument('frames', type = int, nargs = '+',
                         help = 'select single frame or range(start, end, rate). error if nargs != 1 or 3')

### Parse Arguments ###
args = new_argument_parser().parse_args()

with open('analysis.txt', 'r') as f:
   reader = csv.reader(f)
   data = list(reader)
   analysis = np.array(data, dtype = 'float')

#print np.shape(analysis)

smooth = lambda array, kernel_size : ff.gaussian_filter(array, kernel_size)

def get_flux(args):
    i, frame = args

    output  = h5py.File("snapshots/snapshots_s%d.h5" % frame, mode = 'r')
    times    = output['scales']['sim_time'][:] / period
    vx_data    = output['tasks']["ux"]
    vy_data    = output['tasks']["uy"]

    vx_times_vy = np.multiply(vx_data, vy_data)
    flux = np.average(vx_times_vy, axis=(0,-1))
    
    flux_over_time[i] = flux

frame_range = util.get_frame_range(args.frames)

flux_over_time = mp_array("d", len(frame_range))
pool_args = [(i, frame) for i, frame in enumerate(frame_range)]

p = Pool(num_cores)
p.map(get_flux, pool_args)
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
y = np.array(flux_over_time)
y2 = -1.0 * y
plot.plot(x, y, c = "b", linewidth = linewidth)
plot.plot(x, y2, c = "r", linewidth = linewidth)

#plot.plot(x[:-1], y3, linewidth = linewidth)

print(max(y3))

#print(len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-8, 1.0)

plot.yscale('log')

plot.xlabel('t', fontsize = fontsize)
plot.ylabel(r'max($<v_\mathrm{x} v_\mathrm{y}>$)', fontsize = fontsize)
plot.title('Angular Momentum Flux', fontsize = fontsize + 1)

cwd = os.getcwd().split("/")[-1]
plot.savefig("angular_momentum_flux-%s.png" % cwd, bbox_inches = 'tight')
