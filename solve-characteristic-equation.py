"""
Solve fifth-order characteristic equation to reproduce Figure 1 from Latter+ 2010 (growth rate 's' vs. wavenumber 'k')
"""

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams as rc
from matplotlib import pyplot as plot

import math
import numpy as np
from numpy import linalg as linear


def get_growth_rates(big_lambda = 1.0e16, N_squared = -0.1): 
    #big_lambda = 1.0e16 # 1.0
    #N_squared = 0.0 # -0.1
    q = 1.0e-6
    kappa_squared = 1.0
    omega_squared_power = -3.0
    eta = 2.34e17 # 0.0
    xi = q * eta
    alfven_velocity_squared = big_lambda * eta

    ks = np.logspace(-2, 2, 100) / np.sqrt(alfven_velocity_squared)
    growth_rates = np.zeros(len(ks))

    coefficients = np.zeros(6)

    for i, k in enumerate(ks):
        coefficients[0] = 1.0
        # a4
        coefficients[1] = (2.0 * eta + xi) * np.power(k, 2.0)
        # a3
        coefficients[2] = N_squared + kappa_squared + 2.0 * alfven_velocity_squared * np.power(k, 2.0) + np.power(eta, 2.0) * np.power(k, 4.0) + 2.0 * eta * xi * np.power(k, 4.0)
        # a2
        coefficients[3] = 2.0 * N_squared * eta * np.power(k, 2.0) + 2.0 * (eta + xi) * alfven_velocity_squared * np.power(k, 4.0) + (2.0 * eta + xi) * np.power(k, 2.0) * kappa_squared + np.power(eta, 2.0) * xi * np.power(k, 6.0)
        # a1
        coefficients[4] = alfven_velocity_squared * np.power(k, 2.0) * omega_squared_power + np.power(alfven_velocity_squared, 2.0) * np.power(k, 4.0) + (np.power(eta, 2.0) * np.power(k, 2.0) + alfven_velocity_squared) * np.power(k, 2.0) * N_squared + (2.0 * eta * xi + np.power(eta, 2.0)) * np.power(k, 4.0) * kappa_squared + 2.0 * eta * xi * alfven_velocity_squared * np.power(k, 6.0)
        # a0
        coefficients[5] = N_squared * eta * alfven_velocity_squared * np.power(k, 4.0) + xi * np.power(alfven_velocity_squared, 2.0) * np.power(k, 6.0) + np.power(eta, 2.0) * xi * np.power(k, 6.0) * kappa_squared + xi * alfven_velocity_squared * np.power(k, 4.0) * omega_squared_power

        roots = np.roots(coefficients)
        growth_rates[i] = np.max(roots)

        print i, k, growth_rates[i]
        for ci, c in enumerate(coefficients):
            print "a%d %.e" % (ci, c)
        print roots
        print

    return ks * np.sqrt(alfven_velocity_squared), growth_rates

ks, growth_rates1 = get_growth_rates(big_lambda = 1.0e10, N_squared = -0.1)
ks, growth_rates2 = get_growth_rates(big_lambda = 1.0, N_squared = 0.0)
ks, growth_rates3 = get_growth_rates(big_lambda = 1.0, N_squared = -0.1)

#### PLOTTING ####

linewidth = 3
fontsize = 18
labelsize = 16

def make_plot():
    #plot.figure()
    x = ks # * np.sqrt(alfven_velocity_squared)
    y1 = growth_rates1
    y2 = growth_rates2
    y3 = growth_rates3
    plot.plot(x, y1, c = 'purple', linewidth = linewidth, linestyle = "--")
    plot.plot(x, y2, c = 'r', linewidth = linewidth, linestyle = "--")
    plot.plot(x, y3, c = 'b', linewidth = linewidth, linestyle = "-")

    plot.xlim(min(x), max(x))
    plot.ylim(0, 0.8)

    plot.xlabel(r"$k$", fontsize = fontsize)
    plot.ylabel(r"$s$", fontsize = fontsize)
    plot.title("Latter+ 2010: Figure 1 (reproduced)", fontsize = fontsize + 1)

    plot.xscale("log")

    plot.savefig("latter2010-fig1-reproduced.png")
    plot.show()


make_plot()