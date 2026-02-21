import os
import numpy as np
from scipy.ndimage import filters as ff
import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plot
from matplotlib import rcParams as rc

with open('analysis.txt', 'r') as f:
   reader = csv.reader(f)
   data = list(reader)
   analysis = np.array(data, dtype = 'float')

#print np.shape(analysis)

smooth = lambda array, kernel_size : ff.gaussian_filter(array, kernel_size)

linewidth = 3
fontsize = 20

labelsize = 18
rc['xtick.labelsize'] = labelsize
rc['ytick.labelsize'] = labelsize

fig = plot.figure()
ax = fig.add_subplot(111)

x = analysis[:, 0] #/ (2.0 * np.pi)
y = analysis[:, 1]
y2 = smooth(y, 1)
y3 = np.diff(np.log(y2)) / np.diff(x)

x /= (2.0 * np.pi)

#print min(y3), max(y3)

growth = 0.045 #0.00374

plot.plot(x, y, linewidth = linewidth, alpha = 0.6, label = r"$\delta v_\mathrm{max}$")
plot.plot([0, max(x)], [growth, growth], c = 'k', linewidth = linewidth, linestyle = "--",  label = "$s$ (theory)")
#plot.plot(x, y2, linewidth = linewidth, alpha = 0.6, label = "")
plot.plot(x[:-1], y3, linewidth = linewidth, alpha = 0.6, label = r"$s$ (measured)")

plot.legend(loc = "lower right", fontsize = fontsize - 4)

print(max(y3))

#print( len(x), len(y))

plot.xlim(x[0], x[-1])
#plot.ylim(0, max(y))
plot.ylim(10**-6, 1.0)

plot.yscale('log')

plot.xlabel('t [$\mathrm{orbits}$]', fontsize = fontsize)
plot.ylabel(r'$\delta v_\mathrm{max}$ [$r_0 \Omega$]', fontsize = fontsize)
plot.title('Growth Rate Test', fontsize = fontsize + 1)

text_x = 0.05 * plot.xlim()[-1]
text_y = 0.3 * plot.ylim()[-1]

text = r"$q = 10^{-6}$"
text2 = r"$\Lambda = 1.0$"
text3 = r"$N^2 = -0.1$"
text4 = r"$\mathrm{Re} = 10^{-5}$"
plot.text(text_x, text_y, text, fontsize = fontsize - 4)
plot.text(text_x, 0.3 * text_y, text2, fontsize = fontsize - 4)
plot.text(text_x, 0.3 * 0.1 * text_y, text3, fontsize = fontsize - 4)
plot.text(text_x, 0.3 * 0.3 * 0.1 * text_y, text4, fontsize = fontsize - 4)

ax2 = ax.twinx()
ax2.set_yscale("log")
#ax.get_shared_y_axes().join(ax, ax2)
ax2.set_ylim(ax.get_ylim())

cwd = os.getcwd().split("/")[-1]
plot.savefig("growth-rate-log-formal-%s.png" % cwd, bbox_inches = 'tight')
