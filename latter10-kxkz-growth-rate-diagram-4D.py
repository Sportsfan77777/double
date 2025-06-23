"""
Plot growth rates as function of kx and kz
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

import numpy as np
import scipy
from scipy.linalg import eigvals
#from scipy.optimize import fsolve

#from local_vsi_latter18 import OneFluidEigen


### MATRICES ###

# Units
Omega0 = 1.0
Hg = 1.0
rhog0 = 1.0
mu0 = 1.0

cs = Hg * Omega0
cs2 = cs * cs

# Secondary parameters
logReynolds = 7
Reynolds = np.power(10, logReynolds) # Reynolds number for setting viscosity / diffusion

ten_q = 0.5
vertical_shear_q = 0.0 #-ten_q / 10.0 # Hg

nu = Omega0 * Hg**2 / Reynolds # Viscosity and diffusion coefficient
nuM = nu
pert_amp = 1.0e-6

# Variable parameters
roberts_q = 1.0e-5
q = roberts_q # 1.0e-6 # Roberts q

big_lambda = 1e8 # 1.0e16 or 1.0
N_squared = -0.1 #-0.1 # or 0.0

# Problem parameters
kappa_squared = 1.0
omega_squared = 1.0
omega_power = -1.5
omega_squared_power = -3.0
eta = 2.34e12
xi = roberts_q * eta

beta = 1.0e5 # Plasma beta parameter for (inverse) vertical field strength
#va2 = 2.0 * cs2 / beta # Alfven velocity squared
alfven_velocity_squared = 2.0 * cs2 / beta # Alfven velocity squared
eta = alfven_velocity_squared / big_lambda
xi = roberts_q * eta

#alfven_velocity_squared = big_lambda * eta

class OneFluidMatrices:
    def __init__(self, kx, kz):
        self.kx = kx
        self.kz = kz

    def Latter2010(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((5, 5), dtype = np.cdouble)
        matrix_b = np.diag([1,1,1,1,1])

        dissipation = nu * ksq

        #[dvgx, dvgy, dBx, dBy, dtheta]
        matrix_a[0] = np.array([0, 2, ikz * va2, 0, -N_squared / omega_squared])
        matrix_a[1] = np.array([-0.5, 0, 0, ikz * va2, 0])
        matrix_a[2] = np.array([ikz, 0, -ksq * eta, 0, 0])
        matrix_a[3] = np.array([0, ikz, omega_power, -ksq * eta, 0])
        matrix_a[4] = np.array([1, 0, 0, 0, -q * ksq / big_lambda])

        return (matrix_a, matrix_b)

    def Latter2010v0(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((5, 5), dtype = np.cdouble)
        matrix_b = np.diag([1,1,1,1,1])

        dissipation = nu * ksq

        #[dvgx, dvgy, dBx, dBy, dtheta]
        matrix_a[0] = np.array([0, 2, ikz, 0, -N_squared / omega_squared])
        matrix_a[1] = np.array([-0.5, 0, 0, ikz, 0])
        matrix_a[2] = np.array([ikz, 0, -ksq / big_lambda, 0, 0])
        matrix_a[3] = np.array([0, ikz, omega_power, -ksq / big_lambda, 0])
        matrix_a[4] = np.array([1, 0, 0, 0, -xi / big_lambda])

        return (matrix_a, matrix_b)

    def Latter2010xz(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((8, 8), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1,1,1,1])

        dissipation = nu * ksq
        dissipationM = nuM * ksq

        # REMEMBER TO ADD IN DISSIPATION

        #[dP/rhog0, dvgx, dvgy, dvgz, dBx, dBy, dBz, dtheta]
        matrix_a[0] = np.array([0, ikx, 0, ikz, 0, 0, 0, 0])
        matrix_a[1] = np.array([-ikx, -dissipation, 2, 0, ikz, 0, 0, -N_squared / omega_squared])
        matrix_a[2] = np.array([0, -0.5, -dissipation, 0, 0, ikz, 0, 0])
        matrix_a[3] = np.array([-ikz, 0, 0, -dissipation, 0, 0, ikz, 0])
        matrix_a[4] = np.array([0, ikz, 0, 0, -ksq / big_lambda - dissipationM, 0, 0, 0])
        matrix_a[5] = np.array([0, 0, ikz, 0, omega_power, -ksq / big_lambda - dissipationM, 0, 0])
        matrix_a[6] = np.array([0, 0, 0, ikz, 0, 0, -ksq / big_lambda - dissipationM, 0])
        matrix_a[7] = np.array([0, 1, 0, 0, 0, 0, 0, -q * ksq / big_lambda])

        return (matrix_a, matrix_b)

    def Latter2010xz_v0(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((8, 8), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1,1,1,1])

        dissipation = nu * ksq
        dissipationM = nuM * ksq

        # REMEMBER TO ADD IN DISSIPATION

        #[dP/rhog0, dvgx, dvgy, dvgz, dBx, dBy, dBz, dtheta]
        matrix_a[0] = np.array([0, ikx, 0, ikz, 0, 0, 0, 0])
        matrix_a[1] = np.array([-ikx, -dissipation, 2.0, 0, ikz * alfven_velocity_squared, 0, 0, -N_squared / omega_squared])
        matrix_a[2] = np.array([0, -0.5, -dissipation, 0, 0, ikz * alfven_velocity_squared, 0, 0])
        matrix_a[3] = np.array([-ikz, 0, 0, -dissipation, 0, 0, ikz * alfven_velocity_squared, 0])
        matrix_a[4] = np.array([0, ikz, 0, 0, -ksq * eta - dissipationM, 0, 0, 0])
        matrix_a[5] = np.array([0, 0, ikz, 0, omega_power, -ksq * eta - dissipationM, 0, 0])
        matrix_a[6] = np.array([0, 0, 0, ikz, 0, 0, -ksq * eta - dissipationM, 0])
        matrix_a[7] = np.array([0, 1.0, 0, 0, 0, 0, 0, -xi * ksq])

        return (matrix_a, matrix_b)

    def Teed2021(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((5, 5), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1])

        dissipation = nu * ksq

        # [dP/rhog0, dux, duy, duz, dtheta]
        matrix_a[0] = np.array([0, ikx, 0, ikz, 0])
        matrix_a[1] = np.array([-ikx, -dissipation, 2.0 * Omega0, 0, -N_squared / omega_squared])
        matrix_a[2] = np.array([0, -0.5 * Omega0, -dissipation, 0, 0])
        matrix_a[3] = np.array([-ikz, 0, 0, -dissipation, 0])
        matrix_a[4] = np.array([0, 1, 0, 0, -xi * ksq / kappa_squared])

        #matrix_a[0] = np.array([0, ikx, 0, ikz, 0])
        #matrix_a[1] = np.array([-ikx, -dissipation, 2.0 * Omega0, 0, -N_squared / omega_squared])
        #matrix_a[2] = np.array([0, -0.5 * Omega0, -dissipation, vertical_shear_q * Omega0, 0])
        #matrix_a[3] = np.array([-ikz, 0, 0, -dissipation, 0])
        #matrix_a[4] = np.array([1, 0, 0, 0, -roberts_q * ksq / big_lambda])

        return (matrix_a, matrix_b)

    def Teed2021_kzonly(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((3, 3), dtype = np.cdouble)
        matrix_b = np.diag([1,1,1])

        dissipation = nu * ksq

        # [dux, duy, dtheta]
        matrix_a[0] = np.array([-dissipation, 2.0 * Omega0, -N_squared / omega_squared])
        matrix_a[1] = np.array([-0.5 * Omega0, -dissipation, 0])
        matrix_a[2] = np.array([1, 0, -xi * ksq / kappa_squared])

        #matrix_a[0] = np.array([0, ikx, 0, ikz, 0])
        #matrix_a[1] = np.array([-ikx, -dissipation, 2.0 * Omega0, 0, -N_squared / omega_squared])
        #matrix_a[2] = np.array([0, -0.5 * Omega0, -dissipation, vertical_shear_q * Omega0, 0])
        #matrix_a[3] = np.array([-ikz, 0, 0, -dissipation, 0])
        #matrix_a[4] = np.array([1, 0, 0, 0, -roberts_q * ksq / big_lambda])

        return (matrix_a, matrix_b)

def OneFluidEigen(kx, kz):
    matrix = OneFluidMatrices(kx, kz)
    a, b = matrix.Latter2010xz_v0()

    try:
        eigenvalues, eigenvectors = scipy.linalg.eig(a, b)
    except:
        return (0.0, [0.0])

    #print "kx, kz: (%.3e, %.3e)" % (kx, kz)
    #for ei, e in enumerate(eigenvalues):
         #print "s%d %.e" % (ei+1, e)
    #print

    growth = eigenvalues.real
    growth[growth == np.inf] = -np.inf

    gmax = np.argmax(growth)
    eigenvalue = eigenvalues[gmax]
    eigenvector = eigenvectors[:, gmax]

    #print("eigenvalue=",eigenvalue)

    if eigenvalue > 0:
        norm = eigenvector[2] # azimuthal gas velocity
        eigenvector *= pert_amp / norm * Hg * Omega0 # fix units
    #if pert_amp == 0.0:
    #    eigenvector *= 0.0

    return (eigenvalue, eigenvector)

### HELPER FUNCTIONS ###

def get_growth_rate(kx, kz):
    growth, eigenvector = OneFluidEigen(kx, kz)
    #print "Growth: ", growth
    #print "Eigenvector: ", eigenvector

    return growth, eigenvector

### PLOTTING ###

linewidth = 3
fontsize = 16

dpi = 100
cmap = "seismic_r"

version = None
version = 46
save_directory = "."

log_axes = True

def make_plot(show = False):
    fig = plot.figure(figsize = (7, 6), dpi = dpi)
    ax = fig.add_subplot(111)

    # Data
    num_ks = 200

    ks = np.linspace(0.01, 4.5, 100) #/ np.sqrt(alfven_velocity_squared)
    kxs = ks[:]
    kzs = ks[:]

    #kxs = np.linspace(1, 1e5, num_ks)
    #kzs = np.linspace(1, 1e5, num_ks)

    if log_axes:
        kxs = np.logspace(-2, 2, num_ks) / np.sqrt(alfven_velocity_squared)
        kzs = np.logspace(-2, 2, num_ks) / np.sqrt(alfven_velocity_squared)

    growth_rates = np.zeros((len(kxs), len(kzs)))

    for i, kx_i in enumerate(kxs):
        for j, kz_j in enumerate(kzs):
            growth, eigenvector = get_growth_rate(kx_i, kz_j)
            growth_rates[i, j] = growth

            if kx_i == kxs[10] and kz_j == kzs[150]:
                print kx_i , kz_j
                print kx_i * np.sqrt(alfven_velocity_squared), kz_j * np.sqrt(alfven_velocity_squared)
                print "Growth: ", growth
                print "Eigenvector: ", eigenvector

    # Plot
    print np.sqrt(alfven_velocity_squared)
    x = kzs * np.sqrt(alfven_velocity_squared)
    y = kxs * np.sqrt(alfven_velocity_squared)
    #result = ax.pcolormesh(x, y, np.transpose(growth_rates), cmap = cmap)
    result = ax.pcolormesh(x, y, growth_rates, cmap = cmap)

    max_growth = np.max(growth_rates)
    max_growth_two = np.max(growth_rates[:,len(kzs)/2:])
    print max_growth
    print max_growth_two

    cbar = fig.colorbar(result)
    max_color = max_growth
    result.set_clim(-max_color, max_color)

    #cbar.set_label(r"", fontsize = fontsize, rotation = 270, labelpad = 25)

    # Axes
    plot.xlim(min(x), max(x))
    plot.ylim(min(y), max(y))

    if log_axes:
        plot.xscale("log")
        plot.yscale("log")

    # Annotate
    plot.xlabel(r'$k_\mathrm{z}$ $(\Omega / v_\mathrm{A})$', fontsize = fontsize)
    plot.ylabel(r'$k_\mathrm{x}$ $(\Omega / v_\mathrm{A})$', fontsize = fontsize)
    #plot.xlabel(r'$k_\mathrm{z}$ $(\Omega / v_\mathrm{0})$', fontsize = fontsize)
    #plot.ylabel(r'$k_\mathrm{x}$ $(\Omega / v_\mathrm{0})$', fontsize = fontsize)
    plot.title(r'Growth Rates (Latter+ 2010)', fontsize = fontsize + 1)

    x_text = 0.04 * max(x); y_text = 0.92 * max(x)
    if log_axes:
        x_text = 1.4 * min(x); y_text = 0.65 * max(x); y_line = 0.6
    plot.text(x_text, y_text, "Max growth: %.5f" % max_growth, fontsize = fontsize - 2)

    if log_axes:
        plot.text(x_text, y_text * y_line, "Max growth (kz > 1.5): %.5f" % max_growth_two, fontsize = fontsize - 2)

    # Save, Show, and Close
    log_q = np.log10(q) + 10
    log_big_lambda = np.log10(big_lambda) + 2

    if version is None:
        save_fn = "%s/latter10xz-v0-growth-rate-diagram-q%d-bigLambda%d-Re%d.png" % (save_directory, log_q, log_big_lambda, logReynolds)
    else:
        save_fn = "%s/v%04d_latter10xz-v0-growth-rate-diagram-q%d-bigLambda%d-Re%d.png" % (save_directory, version, log_q, log_big_lambda, logReynolds)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)


make_plot(show = True)

"""
def make_plots():
    num_hall = 50 # really half of num_hall
    hall_effects = np.concatenate( (-1.0 * np.logspace(-4, np.log10(2), num_hall)[::-1], [0], np.logspace(-4, np.log10(2), num_hall)) )
    hall_effects = np.linspace(-2, 2, 2 * num_hall + 1)
    #hall_effects = np.array([-2, 0, 2])
    for hall_effect_i, hall_effect in enumerate(hall_effects):
        print "HALL EFFECTS:", hall_effect_i, hall_effect
        make_plot(hall_i = hall_effect_i, hall_effect = hall_effect)
"""

