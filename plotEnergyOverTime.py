import os
import numpy as np
from scipy.ndimage import filters as ff
import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot

with open('analysis.txt', 'r') as f:
   reader = csv.reader(f)
   data = list(reader)
   analysis = np.array(data, dtype = 'float')

#print np.shape(analysis)

smooth = lambda array, kernel_size : ff.gaussian_filter(array, kernel_size)

linewidth = 3
fontsize = 16

plot.figure()

x = analysis[:, 0] / (2.0 * np.pi)
y = analysis[:, 2] # u^2
y2 = analysis[:, 3] # ux^2
y3 = analysis[:, 4] # uy^2
y4 = analysis[:, 5] # uz^2

#print min(y3), max(y3)

plot.plot(x, y, linewidth = linewidth, c = 'k', label = r"$|u|^2$")
plot.plot(x, y2, linewidth = linewidth, c = 'b', label = r"$u_\mathrm{x}^2$")
plot.plot(x, y3, linewidth = linewidth, c = 'r', label = r"$u_\mathrm{y}^2$")
plot.plot(x, y4, linewidth = linewidth, c = 'g', label = r"$u_\mathrm{z}^2$")

plot.legend(loc = "upper left")

#print(max(y3))

#print( len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-8, 1.0)

plot.yscale('log')

plot.xlabel('t', fontsize = fontsize)
plot.ylabel(r'$<v^2>$', fontsize = fontsize)
plot.title('Energy', fontsize = fontsize + 1)

cwd = os.getcwd().split("/")[-1]
plot.savefig("energy-over-time-%s.png" % cwd, bbox_inches = 'tight')
