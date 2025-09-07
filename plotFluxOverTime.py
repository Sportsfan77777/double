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

with open('magnetic_analysis.txt', 'r') as f:
   reader = csv.reader(f)
   data = list(reader)
   analysisM = np.array(data, dtype = 'float')

#print np.shape(analysis)

smooth = lambda array, kernel_size : ff.gaussian_filter(array, kernel_size)

linewidth = 3
fontsize = 16

plot.figure()

x = analysis[:, 0] / (2.0 * np.pi)
y = np.power(analysis[:, 2], 2) # u^2
y2 = np.power(analysis[:, 3], 2) # ux^2
y3 = np.power(analysis[:, 4], 2) # uy^2
y4 = np.power(analysis[:, 5], 2) # uz^2

y_flux = analysis[:, 6] # ux uy
y_fluxM = analysisM[:, 5] # Bx By

#print min(y3), max(y3)

plot.plot(x, y_flux, linewidth = linewidth, c = 'b', label = r"$u_\mathrm{x} u_\mathrm{y}$ (+)")
plot.plot(x, -y_flux, linewidth = linewidth, c = 'r', label = r"$u_\mathrm{x} u_\mathrm{y}$ (-)")
plot.plot(x, y_fluxM, linewidth = linewidth, c = 'darkblue', label = r"$B_\mathrm{x} B_\mathrm{y}$ (+)")
plot.plot(x, -y_fluxM, linewidth = linewidth, c = 'darkred', label = r"$B_\mathrm{x} B_\mathrm{y}$ (-)")

plot.legend(loc = "upper left")

#print(max(y3))

#print( len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-8, 1.0)

plot.yscale('log')

plot.xlabel('t', fontsize = fontsize)
plot.ylabel(r'$Q_\mathrm{x} Q_\mathrm{y}$', fontsize = fontsize)
plot.title('Flux', fontsize = fontsize + 1)

cwd = os.getcwd().split("/")[-1]
plot.savefig("flux-over-time-%s.png" % cwd, bbox_inches = 'tight')
