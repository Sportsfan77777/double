"""
Solve fifth-order characteristic equation to reproduce Figure 1 from Latter+ 2010 (growth rate 's' vs. wavenumber 'k')
"""

import matplotlib
#matplotlib.use('Agg')
from matplotlib import rcParams as rc
from matplotlib import pyplot as plot

import scipy
from scipy.linalg import eigvals
from scipy.optimize import fsolve

import math
import numpy as np
from numpy import linalg as linear


def get_growth_rates(N_squared = -0.1): 
    #big_lambda = 1.0e16 # 1.0
    #N_squared = 0.0 # -0.1
    q = 1.0e-6
    kappa_squared = 1.0
    omega_power = -1.5
    omega_squared_power = -3.0
    omega_squared = 1.0
    eta = 2.34e17 # 0.0
    xi = q * eta
    omega0 = 1.0
    B0 = 1.0

    ks = np.linspace(0.01, 4.5, 100) / np.sqrt(xi)
    growth_rates = np.zeros(len(ks))

    coefficients = np.zeros(3)
    I = 1j

    for i, k in enumerate(ks):
        print i, k
        matrix = np.zeros((3, 3), dtype = np.complex128)
        matrix_b = np.diag([1,1,1])

        # [dP/rhog0, dux, duy, dtheta]
        matrix[0, 0] = 0
        matrix[0, 1] = 2.0
        matrix[0, 2] = -N_squared / omega_squared

        matrix[1, 0] = -0.5
        matrix[1, 1] = 0
        matrix[1, 2] = 0

        matrix[2, 0] = 1.0
        matrix[2, 1] = 0
        matrix[2, 2] = -xi * np.power(k, 2.0) * kappa_squared

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
        #eigenvalues = np.linalg.eigvals(matrix)
        #growth_rates[i] = np.max(eigenvalues)

        eigenvalues, eigenvectors = scipy.linalg.eig(matrix, matrix_b)

        growth = eigenvalues.real
        growth[growth == np.inf] = -np.inf

        gmax = np.argmax(growth)
        eigenvalue = eigenvalues[gmax]

        growth_rates[i] = eigenvalue

        print i, k, growth_rates[i]
        for ei, e in enumerate(eigenvalues):
            print "s%d %.e" % (ei+1, e)
        #print roots
        print

    return ks * np.sqrt(xi), growth_rates

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
    #plot.plot(x, y3, c = 'b', linewidth = linewidth, linestyle = "-")

    plot.xlim(min(x), max(x))
    plot.ylim(0, 3.0e-3)

    plot.xlabel(r"$k$ $(\xi / \kappa)^{1/2}$", fontsize = fontsize)
    plot.ylabel(r"$s$", fontsize = fontsize)
    plot.title("Latter 2016: Figure 1 (reproduced)", fontsize = fontsize + 1)

    plot.savefig("latter2016-fig1-reproduced-from-matrix.png")
    plot.show()


make_plot()