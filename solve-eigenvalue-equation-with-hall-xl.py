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


def get_growth_rates(big_lambda = 1.0e16, N_squared = -0.1, big_lambda_H = 0.2): 
    #big_lambda = 1.0e16 # 1.0
    #N_squared = 0.0 # -0.1
    q = 1.0e-6
    kappa_squared = 1.0
    omega_power = -1.5
    omega_squared_power = -3.0
    omega_squared = 1.0
    eta = 2.34e17 # 0.0
    xi = q * eta
    alfven_velocity_squared = big_lambda * eta
    omega0 = 1.0
    B0 = 1.0

    if big_lambda_H == 0:
        big_lambda_H = 1.0e-10

    ks = np.logspace(-2, 2, 100) #/ np.sqrt(alfven_velocity_squared)
    growth_rates = np.zeros(len(ks))

    coefficients = np.zeros(5)
    I = 1j

    for i, k in enumerate(ks):
        print i, k
        matrix = np.zeros((5, 5), dtype = np.complex128)

        matrix[0, 0] = 0
        matrix[0, 1] = 2.0
        matrix[0, 2] = I * k
        matrix[0, 3] = 0
        matrix[0, 4] = -N_squared / omega_squared

        matrix[1, 0] = -0.5
        matrix[1, 1] = 0
        matrix[1, 2] = 0
        matrix[1, 3] = I * k
        matrix[1, 4] = 0

        matrix[2, 0] = I * k
        matrix[2, 1] = 0
        matrix[2, 2] = -np.power(k, 2.0) / big_lambda
        matrix[2, 3] = -0.5 * np.power(k, 2.0) / big_lambda_H # Note: \eta_H = c / (4 \pi n_e e) = 0.5 / Ha  [But... are the Hall minus signs backwards??? There is a negative already there, plus another from the k^2!!]
        matrix[2, 4] = 0

        matrix[3, 0] = 0
        matrix[3, 1] = I * k
        matrix[3, 2] = omega_power + 0.5 * np.power(k, 2.0) / big_lambda_H # Ha = v_A^2 / v_H^2; and v_H^2 = \Omega B_0 c / (2 \pi n_e e)
        matrix[3, 3] = -np.power(k, 2.0) / big_lambda
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

        print i, k, growth_rates[i]
        for ei, e in enumerate(eigenvalues):
            print "s%d %.e" % (ei, e)
        #print roots
        print

    #return ks * np.sqrt(alfven_velocity_squared), growth_rates
    return ks, growth_rates


#### PLOTTING ####

linewidth = 3
fontsize = 18
labelsize = 16

def make_plot(hall_i, hall_effect):
    plot.figure(figsize = (7, 6))

    ## GET DATA ##
    print "HALL EFFECTS (confirmed):", hall_effect_i, hall_effect

    ks, growth_rates1 = get_growth_rates(big_lambda = 1.0e10, N_squared = -0.1, big_lambda_H = hall_effect)
    ks, growth_rates2 = get_growth_rates(big_lambda = 1.0, N_squared = 0.0, big_lambda_H = hall_effect)
    ks, growth_rates3 = get_growth_rates(big_lambda = 1.0, N_squared = -0.1, big_lambda_H = hall_effect)

    ks, growth_rates4 = get_growth_rates(big_lambda = 1.0e10, N_squared = 0.1, big_lambda_H = hall_effect)
    ks, growth_rates5 = get_growth_rates(big_lambda = 1.0e10, N_squared = 0.0, big_lambda_H = hall_effect)

    ## PLOT ##
    x = ks # * np.sqrt(alfven_velocity_squared)
    y1 = growth_rates1
    y2 = growth_rates2
    y3 = growth_rates3
    y4 = growth_rates4
    y5 = growth_rates5
    plot.plot(x, y1, c = 'purple', linewidth = linewidth, linestyle = "--", label = r"$\Lambda = \infty, N^2 = -0.1$")
    plot.plot(x, y5, c = 'g', linewidth = linewidth, linestyle = "--", label = r"$\Lambda = \infty, N^2 = 0$")
    plot.plot(x, y4, c = 'g', linewidth = linewidth, linestyle = "--", label = r"$\Lambda = \infty, N^2 = 0.1$")

    plot.plot(x, y2, c = 'r', linewidth = linewidth, linestyle = "--", label = r"$\Lambda = 1, N^2 = 0$")
    plot.plot(x, y3, c = 'b', linewidth = linewidth, linestyle = "-", label = r"$\Lambda = 1, N^2 = -0.1$")

    plot.xlim(min(x), max(x))
    plot.ylim(0, 0.8)

    plot.legend(loc = "upper left")

    plot.xlabel(r"$\hat{k}$ [$\Omega / v_\mathrm{A}$]", fontsize = fontsize)
    plot.ylabel(r"$\hat{s}$ [$\Omega$]", fontsize = fontsize)
    plot.title("Latter+ 2010: Figure 1 (with Hall Ha = %.2f)" % hall_effect, fontsize = fontsize + 1)

    plot.xscale("log")

    plot.savefig("latter2010-fig1-from-matrix-with-hall-%04d.png" % hall_i, bbox_inches = 'tight')
    plot.show()

    plot.cla()


num_hall = 50 # really half of num_hall
hall_effects = np.concatenate( (-1.0 * np.logspace(-4, np.log10(2), num_hall)[::-1], [0], np.logspace(-4, np.log10(2), num_hall)) )
hall_effects = np.linspace(-2, 2, 2 * num_hall + 1)
#hall_effects = np.array([-2, 0, 2])
for hall_effect_i, hall_effect in enumerate(hall_effects):
    print "HALL EFFECTS:", hall_effect_i, hall_effect
    make_plot(hall_effect_i, hall_effect)