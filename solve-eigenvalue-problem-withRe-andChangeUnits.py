"""
Solve fifth-order characteristic equation to reproduce Figure 1 from Latter+ 2010 (growth rate 's' vs. wavenumber 'k')
"""

import sys, os
import itertools
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams as rc
from matplotlib import pyplot as plot



import math
import numpy as np
from numpy import linalg as linear

#B0 = 1.0
omega = 1.0
cs2 = 1.0

kappa_squared = 1.0
omega_power = -1.5
omega_squared_power = -3.0
omega_squared = 1.0

Reynolds = 1.0e6
nu = 1.0 / Reynolds
nuM = 0 #nu

beta = 1.0e5
alfven_velocity_squared = 2.0 * cs2 / beta
#eta = 2.34e17 # 0.0

def get_growth_rates(roberts_q = 1.0e-6, big_lambda = 1.0): 
    #big_lambda = 1.0e16 # 1.0
    N_squared = -0.1
    q = roberts_q
    eta = alfven_velocity_squared / big_lambda
    xi = q * eta
    #alfven_velocity_squared = big_lambda * eta
    ks = np.logspace(-2, 8, 200) #/ np.sqrt(alfven_velocity_squared)
    growth_rates = np.zeros(len(ks))

    coefficients = np.zeros(5)
    I = 1j

    for i, k in enumerate(ks):
        print i, k
        matrix = np.zeros((5, 5), dtype = np.complex128)

        ksq = k**2
        dissipation = nu * ksq
        dissipationM = nuM * ksq

        matrix[0, 0] = -dissipation
        matrix[0, 1] = 2.0
        matrix[0, 2] = I * k * alfven_velocity_squared
        matrix[0, 3] = 0
        matrix[0, 4] = -N_squared / omega_squared

        matrix[1, 0] = -0.5
        matrix[1, 1] = -dissipation
        matrix[1, 2] = 0
        matrix[1, 3] = I * k * alfven_velocity_squared
        matrix[1, 4] = 0

        matrix[2, 0] = I * k
        matrix[2, 1] = 0
        matrix[2, 2] = -np.power(k, 2.0) * eta - dissipationM
        matrix[2, 3] = 0
        matrix[2, 4] = 0

        matrix[3, 0] = 0
        matrix[3, 1] = I * k
        matrix[3, 2] = omega_power
        matrix[3, 3] = -np.power(k, 2.0) * eta - dissipationM
        matrix[3, 4] = 0

        matrix[4, 0] = 1.0
        matrix[4, 1] = 0
        matrix[4, 2] = 0
        matrix[4, 3] = 0
        matrix[4, 4] = -np.power(k, 2.0) * xi


        # a4
        #coefficients[0] = (2.0 * eta + xi) * np.power(k, 2.0)
        # a3
        #coefficients[1] = N_squared + kappa_squared + 2.0 * alfven_velocity_squared * np.power(k, 2.0) + np.power(eta, 2.0) * np.power(k, 4.0) + 2.0 * eta * xi * np.power(k, 4.0)
        # a2
        #coefficients[2] = 2.0 * N_squared * eta * np.power(k, 2.0) + 2.0 * (eta + xi) * alfven_velocity_squared * np.power(k, 4.0) + (2.0 * eta + xi) * np.power(k, 2.0) * kappa_squared + np.power(eta, 2.0) * xi * np.power(k, 6.0)
        # a1
        #coefficients[3] = alfven_velocity_squared * np.power(k, 2.0) * omega_squared_power + np.power(alfven_velocity_squared, 2.0) * np.power(k, 4.0) + (np.power(eta, 2.0) * np.power(k, 2.0) + alfven_velocity_squared) * np.power(k, 2.0) * N_squared + (2.0 * eta * xi + np.power(eta, 2.0)) * np.power(k, 4.0) * kappa_squared + 2.0 * eta * xi * alfven_velocity_squared * np.power(k, 6.0)
        # a0
        #coefficients[4] = N_squared * eta * alfven_velocity_squared * np.power(k, 4.0) + xi * np.power(alfven_velocity_squared, 2.0) * np.power(k, 6.0) + np.power(eta, 2.0) * xi * np.power(k, 6.0) * kappa_squared + xi * alfven_velocity_squared * np.power(k, 4.0) * omega_squared_power

        #eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = np.linalg.eigvals(matrix)
        growth_rates[i] = np.max(eigenvalues)

        print i, k, growth_rates[i]
        for ei, e in enumerate(eigenvalues):
            print "s%d %.e" % (ei, e)
        #print roots
        print

    #return ks * np.sqrt(alfven_velocity_squared), growth_rates
    return ks, growth_rates

#ks, growth_rates1 = get_growth_rates(big_lambda = 1.0e10, N_squared = -0.1)
#ks, growth_rates2 = get_growth_rates(big_lambda = 1.0, N_squared = 0.0)
#ks, growth_rates3 = get_growth_rates(big_lambda = 1.0, N_squared = -0.1)

#### PLOTTING ####

#version = 632

linewidth = 3
fontsize = 18
labelsize = 16

def make_plot(version = None, big_lambda = 1.0, show = False):
    #plot.figure()

    # Data
    roberts_qs = np.logspace(-10, 0, 11)
    #roberts_qs = np.linspace(1e-8, 1e-7, 10)
    #big_lambda = 1.0e-2
    eta = alfven_velocity_squared / big_lambda

    for i, q_i in enumerate(roberts_qs):
       ks, growth_rates = get_growth_rates(roberts_q = q_i, big_lambda = big_lambda)

       x = ks #* np.sqrt(alfven_velocity_squared)
       y = growth_rates
       plot.plot(x, y, linewidth = linewidth, label = "q = %.1e" % q_i)

    plot.legend(loc = "upper right")

    max_y = 1.5 * max(y)
    plot.xlim(min(x), max(x))
    plot.ylim(0, max_y)

    plot.xscale("log")

    plot.xlabel(r"$k$ $[\Omega / v_\mathrm{0}]$", fontsize = fontsize)
    plot.ylabel(r"$s$ $[\Omega]$", fontsize = fontsize)
    plot.title("Latter+ 2010: Figure 1 (reproduced)", fontsize = fontsize + 1)

    x_text = 1.4 * min(x); y_text = 0.93 * max_y; y_line = 0.06 * max_y
    plot.text(x_text, y_text, r"$\Lambda = %.1e$" % (big_lambda), fontsize = fontsize - 4)
    plot.text(x_text, y_text - 1.0 * y_line, r"$Re = %.1e$" % (Reynolds), fontsize = fontsize - 4)
    plot.text(x_text, y_text - 2.0 * y_line, r"$\beta = %.1e$" % (beta), fontsize = fontsize - 4)
    plot.text(x_text, y_text - 3.0 * y_line, r"$v_\mathrm{A}^2 = %.1e$" % (alfven_velocity_squared), fontsize = fontsize - 4)

    if version is None:
       plot.savefig("latter2010-v0-fig1.png", bbox_inches = "tight")
    else:
       plot.savefig("v%04d_latter2010-v0-fig1.png" % version, bbox_inches = "tight")

    if show:
       plot.show()


#make_plot(show = False)

v = 640
versions = np.array(range(3)) + v

num_lambda = 3
big_lambdas = np.logspace(-2, 0, num_lambda)

for i, (v_i, big_lambda_i) in enumerate(zip(versions, big_lambdas)):
    make_plot(version = v_i, big_lambda = big_lambda_i, show = False)
