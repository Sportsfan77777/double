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

x = analysis[:, 0] #/ (2.0 * np.pi)
y = analysis[:, 1]
y2 = smooth(y, 10)
y3 = np.diff(np.log(y2)) / np.diff(x)

#print min(y3), max(y3)

plot.plot(x, y, linewidth = linewidth, alpha = 0.6)
plot.plot(x, y2, linewidth = linewidth)
plot.plot(x[:-1], y3, linewidth = linewidth)

#print( len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-8, 1.0)

plot.yscale('log')

plot.xlabel('t', fontsize = fontsize)
plot.ylabel(r'max($\delta v$)', fontsize = fontsize)
plot.title('Growth Rate Test', fontsize = fontsize + 1)

plot.savefig("growth-rate-log.png", bbox_inches = 'tight')