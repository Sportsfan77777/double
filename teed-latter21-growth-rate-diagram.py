"""
Plot growth rates as function of kx and kz
"""

import numpy as np
import matplotlib.pyplot as plot

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
logReynolds = 10
Reynolds = np.power(10, logReynolds) # Reynolds number for setting viscosity / diffusion

ten_q = 0.5
vertical_shear_q = 0.0 #-ten_q / 10.0 # Hg
roberts_q = 1.0e-6

nu = Omega0 * Hg**2 / Reynolds # Viscosity and diffusion coefficient
pert_amp = 1.0e-6

# Variable parameters
N_squared = -0.1 #-0.1 # or 0.0

# Problem parameters
kappa_squared = 1.0
omega_squared = 1.0
omega_power = -1.5
omega_squared_power = -3.0
eta = 2.34e17
xi = roberts_q * eta

class OneFluidMatrices:
    def __init__(self, kx, kz):
        self.kx = kx
        self.kz = kz

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
    a, b = matrix.Teed2021()

    eigenvalues, eigenvectors = scipy.linalg.eig(a, b)

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
    return growth

### PLOTTING ###

linewidth = 3
fontsize = 16

dpi = 100
cmap = "seismic_r"

version = None
version = 2
save_directory = "."

log_axes = True

def make_plot(show = True):
    fig = plot.figure(figsize = (7, 6), dpi = dpi)
    ax = fig.add_subplot(111)

    # Data
    num_ks = 200

    ks = np.linspace(0.01, 4.5, 100) / np.sqrt(xi)
    kxs = ks[:]
    kzs = ks[:]

    #kxs = np.linspace(1, 1e5, num_ks)
    #kzs = np.linspace(1, 1e5, num_ks)

    if log_axes:
        kxs = np.logspace(-2, 2, num_ks) / np.sqrt(xi)
        kzs = np.logspace(-2, 2, num_ks) / np.sqrt(xi)

    growth_rates = np.zeros((len(kxs), len(kzs)))

    for i, kx_i in enumerate(kxs):
        for j, kz_j in enumerate(kzs):
            growth_rates[i, j] = get_growth_rate(kx_i, kz_j)

    # Plot
    x = kxs * np.sqrt(xi)
    y = kzs * np.sqrt(xi)
    result = ax.pcolormesh(x, y, np.transpose(growth_rates), cmap = cmap)

    max_growth = np.max(growth_rates)
    print max_growth

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
    plot.xlabel(r'$k_\mathrm{x}$ $(\xi / \kappa)^{1/2}$', fontsize = fontsize)
    plot.ylabel(r'$k_\mathrm{z}$ $(\xi / \kappa)^{1/2}$', fontsize = fontsize)
    plot.title(r'Growth Rates (Teed + Latter 2021)', fontsize = fontsize + 1)

    x_text = 0.04 * max(x); y_text = 0.92 * max(x)
    if log_axes:
        x_text = 1.4 * min(x); y_text = 0.65 * max(x)
    plot.text(x_text, y_text, "Max growth: %.5f" % max_growth, fontsize = fontsize - 2)

    # Save, Show, and Close
    if version is None:
        save_fn = "%s/teed-latter21-growth-rate-diagram-Re%d.png" % (save_directory, logReynolds)
    else:
        save_fn = "%s/v%04d_teed-latter21-growth-rate-diagram-Re%d.png" % (save_directory, version, logReynolds)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)


make_plot()

