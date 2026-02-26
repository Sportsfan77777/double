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

#Reynolds = 1.0e8
#nu = 1.0 / Reynolds
nuM = 0 #nu

beta = 1.0e5
alfven_velocity_squared = 2.0 * cs2 / beta
#eta = 2.34e17 # 0.0

N_squared = -0.05

def get_growth_rates(roberts_q = 1.0e-6, big_lambda = 1.0, Reynolds = 1.0e2):
    #big_lambda = 1.0e16 # 1.0
    nu = 1.0 / Reynolds
    q = roberts_q
    eta = alfven_velocity_squared / big_lambda
    xi = q * eta
    #alfven_velocity_squared = big_lambda * eta
    ks = np.logspace(-2, 8, 200) #/ np.sqrt(alfven_velocity_squared)
    growth_rates = np.zeros(len(ks))

    kz_mri = 4.0 * big_lambda / np.sqrt(alfven_velocity_squared)
    lambda_z = 2.0 * np.pi / kz_mri
    Lbox = 2.0 * lambda_z

    coefficients = np.zeros(5)
    I = 1j

    for i, k in enumerate(ks):
        #print i, k
        matrix = np.zeros((5, 5), dtype = np.complex128)

        ksq = k**2
        dissipation = nu * ksq * np.power(Lbox, 2.0)
        dissipationM = 0 #nuM * ksq

        matrix[0, 0] = -dissipation
        matrix[0, 1] = 2.0
        matrix[0, 2] = I * k
        matrix[0, 3] = 0
        matrix[0, 4] = -N_squared / omega_squared

        matrix[1, 0] = -0.5
        matrix[1, 1] = -dissipation
        matrix[1, 2] = 0
        matrix[1, 3] = I * k
        matrix[1, 4] = 0

        matrix[2, 0] = I * k
        matrix[2, 1] = 0
        matrix[2, 2] = -np.power(k, 2.0) / big_lambda - dissipationM
        matrix[2, 3] = 0
        matrix[2, 4] = 0

        matrix[3, 0] = 0
        matrix[3, 1] = I * k
        matrix[3, 2] = omega_power
        matrix[3, 3] = -np.power(k, 2.0) / big_lambda - dissipationM
        matrix[3, 4] = 0

        matrix[4, 0] = 1.0
        matrix[4, 1] = 0
        matrix[4, 2] = 0
        matrix[4, 3] = 0
        matrix[4, 4] = -q * np.power(k, 2.0) / big_lambda

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

        #print i, k, growth_rates[i]
        #for ei, e in enumerate(eigenvalues):
        #    print "s%d %.e" % (ei, e)
        #print roots
        #print

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
cmap = "YlOrRd_r"
#cmap = "seismic"

labelsize = 14
rc['xtick.labelsize'] = labelsize
rc['ytick.labelsize'] = labelsize

def make_plot(version = None, Reynolds_number = 1.0e8, show = False):
    fig = plot.figure()
    ax = fig.add_subplot(111)

    # Data
    num_lambda = 121 # 121
    big_lambdas = np.logspace(0, -5, num_lambda)

    roberts_qs = np.logspace(-10, 0, 71) # 71
    #roberts_qs = np.linspace(1e-8, 1e-7, 10)
    #big_lambda = 1.0e-2
    #eta = alfven_velocity_squared / big_lambda

    growth_rate_grid = np.zeros((num_lambda, len(roberts_qs)))

    print Reynolds_number

    for i, big_lambda_i in enumerate(big_lambdas):
       for j, q_i in enumerate(roberts_qs):
          ks, growth_rates = get_growth_rates(roberts_q = q_i, big_lambda = big_lambda_i, Reynolds = Reynolds_number)

          mri_cutoff = big_lambda_i * 4.0
          mri_cutoff_i = np.searchsorted(ks, mri_cutoff)
          #print "Cutoff:", mri_cutoff_i
          growth_rates[:mri_cutoff_i] = 0.0

          max_growth_rate = np.max(growth_rates)
          growth_rate_grid[i, j] = np.log10(max_growth_rate)

    print Reynolds_number

    x = big_lambdas #* np.sqrt(alfven_velocity_squared)
    y = roberts_qs
    result = ax.pcolormesh(x, y, np.transpose(growth_rate_grid), cmap = cmap)

    clim = [-3, 0]
    cbar = fig.colorbar(result)
    result.set_clim(clim[0], clim[1])

    cbar.set_label(r"$s_\mathrm{max}$ [$\Omega^{-1}$]", fontsize = fontsize, rotation = 270, labelpad = 25)

    # MRI cutoffs based on scale height
    #beta1 = 1e5; mri_x1 = np.sqrt(2) * np.power(beta1, -0.5)
    beta2 = 2e4; mri_x2 = np.sqrt(2) * np.power(beta2, -0.5)
    beta3 = 1e3; mri_x3 = np.sqrt(2) * np.power(beta3, -0.5)

    #plot.plot([mri_x1, mri_x1], [40.0 * min(y), max(y)], c = 'k', linestyle = "--", alpha = 0.6)
    plot.plot([mri_x2, mri_x2], [40.0 * min(y), max(y)], c = 'k', linestyle = "--", alpha = 0.6)
    plot.plot([mri_x3, mri_x3], [40.0 * min(y), max(y)], c = 'k', linestyle = "-")

    plot.xlim(min(x), max(x))
    plot.ylim(min(y), max(y))

    plot.xscale("log")
    plot.yscale("log")

    plot.xlabel(r"$\Lambda$", fontsize = fontsize)
    plot.ylabel(r"$q$", fontsize = fontsize)
    plot.title("Maximum Growth Rates", fontsize = fontsize + 1)

    #x_text = 0.023 * max(x); y_text = 0.1 * max(y); y_line = 0.2 # * max(y)
    x_text = 2.0 * min(x); y_text = 0.1 * max(y); y_line = 0.2 # * max(y)
    #plot.text(x_text, y_text, r"$\Lambda = %.1e$" % (big_lambda), fontsize = fontsize - 4)
    plot.text(x_text, y_text, r"$Re = 10^{%d}$" % (int(round(np.log10(Reynolds_number), 0))), fontsize = fontsize - 4)
    plot.text(x_text, y_text * 1.0 * y_line, r"$N^2 = %.2f$" % (N_squared), fontsize = fontsize - 4)
    #plot.text(x_text, y_text - 3.0 * y_line, r"$v_\mathrm{A}^2 = %.1e$" % (alfven_velocity_squared), fontsize = fontsize - 4)

    rddi_x = 0.023
    y_label = 2e-5 * max(y)
    plot.text(10.0 * min(x), y_label, "COS", verticalalignment = "center", fontsize = fontsize + 10)
    plot.text(rddi_x, 3.0 * y_label, "R-DDI", fontsize = fontsize, horizontalalignment =  "center", rotation = 90)
    plot.text(2.8 * min(x), 3e-5 * y_label, "R-DDI", fontsize = fontsize - 5, horizontalalignment =  "center", rotation = -30)
    plot.text(0.10 * max(x), y_label, "MRI", verticalalignment = "center", fontsize = fontsize + 6)

    y_mri_text = 2.3 * min(y)
#    plot.text(0.3 * mri_x1, y_mri_text, r"$\beta = $", fontsize = fontsize - 5)
    plot.text(2.0 * mri_x2, 5.0 * y_mri_text, "MRI cutoffs with", fontsize = fontsize - 5, horizontalalignment = "center")
    #plot.text(1.1 * mri_x1, y_mri_text, r"$\beta = 10^{5}$", fontsize = fontsize - 5, horizontalalignment = "right")
    plot.text(0.5 * mri_x2, y_mri_text, r"$\beta = 2 \times 10^{4}$", fontsize = fontsize - 5, horizontalalignment = "center")
    plot.text(0.9 * mri_x3, y_mri_text, r"$10^{3}$", fontsize = fontsize - 5)

    if version is None:
       plot.savefig("latter2010-v0-growth-rate-grid-normalized-log-formal-c.png", bbox_inches = "tight")
    else:
       plot.savefig("v%04d_latter2010-v0-growth-rate-grid-normalized-log-formal-c.png" % version, bbox_inches = "tight")

    if show:
       plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)

#make_plot(show = False)

v = 9007

num_Reynolds = 1
Reynolds_numbers = np.logspace(11, 11, num_Reynolds)

versions = np.array(range(num_Reynolds)) + v

for i, (v_i, Reynolds_i) in enumerate(zip(versions, Reynolds_numbers)):
    print v_i, Reynolds_i
    make_plot(version = v_i, Reynolds_number = Reynolds_i, show = False)

#make_plot(version = v, show = False)
