"""
Solve fifth-order characteristic equation to reproduce Figure 1 from Latter+ 2010 (growth rate 's' vs. wavenumber 'k')
"""

import matplotlib
#matplotlib.use('Agg')
from matplotlib import rcParams as rc
from matplotlib import pyplot as plot

import math
import numpy as np
from numpy import linalg as linear


def get_growth_rates(N_squared = -0.1): 
    #big_lambda = 1.0e16 # 1.0
    #N_squared = 0.0 # -0.1
    q = 1.0e-6
    kappa_squared = 1.0
    omega_squared_power = -3.0
    eta = 2.34e17 # 0.0
    xi = q * eta

    ks = np.linspace(0.01, 4.5, 100) / np.sqrt(xi)
    #ks = np.logspace(-2, np.log10(4.5), 100) / np.sqrt(xi)
    growth_rates = np.zeros(len(ks))

    coefficients = np.zeros(4)

    for i, k in enumerate(ks):
        coefficients[0] = 1.0
        # a3
        coefficients[1] = xi * np.power(k, 2.0)
        # a2
        coefficients[2] = N_squared + kappa_squared
        # a1
        coefficients[3] = xi * np.power(k, 2.0) * kappa_squared

        roots = np.roots(coefficients)
        growth_rates[i] = np.max(np.real(roots))

        print i, k, growth_rates[i]
        for ci, c in enumerate(coefficients):
            print "a%d %.e" % (ci, c)
        print "Roots:", roots
        print

    return ks * np.sqrt(xi), growth_rates
    #return ks, growth_rates

ks, growth_rates1 = get_growth_rates(N_squared = -0.01)
#ks, growth_rates2 = get_growth_rates(big_lambda = 1.0, N_squared = 0.0)
#ks, growth_rates3 = get_growth_rates(big_lambda = 1.0, N_squared = -0.1)

#### PLOTTING ####

linewidth = 3
fontsize = 18
labelsize = 16

def make_plot():
    #plot.figure()
    x = ks # * np.sqrt(alfven_velocity_squared)
    y1 = growth_rates1
    #y2 = growth_rates2
    #y3 = growth_rates3
    plot.plot(x, y1, c = 'b', linewidth = linewidth, linestyle = "-")
    #plot.plot(x, y2, c = 'r', linewidth = linewidth, linestyle = "--")
    #plot.plot(x, y3, c = 'purple', linewidth = linewidth, linestyle = "-")

    plot.xlim(min(x), max(x))
    plot.ylim(0, 3.0e-3)

    plot.xlabel(r"$k$ $(\xi / \kappa)^{1/2}$", fontsize = fontsize)
    plot.ylabel(r"$s$ $(\kappa)$", fontsize = fontsize)
    plot.title("Latter 2016: Figure 1 (reproduced)", fontsize = fontsize + 1)

    #plot.xscale("log")

    plot.savefig("latter2016-fig1-reproduced.png")
    plot.show()


make_plot()