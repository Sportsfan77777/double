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
logReynolds = 7
Reynolds = np.power(10, logReynolds) # Reynolds number for setting viscosity / diffusion

ten_q = 0.5
q = -ten_q / 10.0 # Hg

nu = Omega0 * Hg**2 / Reynolds # Viscosity and diffusion coefficient
pert_amp = 1.0e-3

class OneFluidMatrices:
    def __init__(self, kx, kz):
        self.kx = kx
        self.kz = kz

    def Latter2018(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((4, 4), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1])

        dissipation = nu * ksq

        # [dP/rhog0, dux, duy, duz]
        matrix_a[0] = np.array([0, ikx, 0, ikz]) # Note: This is div v = 0 (the continuity equation)
        matrix_a[1] = np.array([-ikx, -dissipation, 2.0 * Omega0, 0])
        matrix_a[2] = np.array([0, -0.5 * Omega0, -dissipation, q * Omega0])
        matrix_a[3] = np.array([-ikz, 0, 0, -dissipation])

        return (matrix_a, matrix_b)

def OneFluidEigen(kx, kz):
    matrix = OneFluidMatrices(kx, kz)
    a, b = matrix.Latter2018()

    eigenvalues, eigenvectors = scipy.linalg.eig(a, b)

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
save_directory = "."

log_axes = True

def make_plot(show = True):
    fig = plot.figure(figsize = (7, 6), dpi = dpi)
    ax = fig.add_subplot(111)

    # Data
    num_ks = 200
    kxs = np.linspace(1, 1000, num_ks)
    kzs = np.linspace(1, 1000, num_ks)

    if log_axes:
        kxs = np.logspace(0, 3, num_ks)
        kzs = np.logspace(0, 3, num_ks)

    growth_rates = np.zeros((len(kxs), len(kzs)))

    for i, kx_i in enumerate(kxs):
        for j, kz_j in enumerate(kzs):
            growth_rates[i, j] = get_growth_rate(kx_i, kz_j)

    # Plot
    x = kxs
    y = kzs
    result = ax.pcolormesh(x, y, np.transpose(growth_rates), cmap = cmap)

    max_growth = np.max(growth_rates)
    print max_growth

    cbar = fig.colorbar(result)
    max_color = 2.0 * np.abs(q)
    result.set_clim(-max_color, max_color)

    #cbar.set_label(r"", fontsize = fontsize, rotation = 270, labelpad = 25)

    # Axes
    plot.xlim(min(kxs), max(kxs))
    plot.ylim(min(kzs), max(kzs))

    if log_axes:
        plot.xscale("log")
        plot.yscale("log")

    # Annotate
    plot.xlabel(r'$k_\mathrm{x}$', fontsize = fontsize)
    plot.ylabel(r'$k_\mathrm{z}$', fontsize = fontsize)
    plot.title(r'Growth Rates ($q = %.1f$, $Re = 10^{%d}$)' % (q, logReynolds), fontsize = fontsize + 1)

    x_text = 0.04 * max(kxs); y_text = 0.92 * max(kxs)
    if log_axes:
        x_text = 1.4 * min(kxs); y_text = 0.65 * max(kxs)
    plot.text(x_text, y_text, "Max growth: %.5f" % max_growth, fontsize = fontsize - 2)

    # Save, Show, and Close
    if version is None:
        save_fn = "%s/latter18-growth-rate-diagram-q05-Re%d.png" % (save_directory, logReynolds)
    else:
        save_fn = "%s/v%04d_latter18-growth-rate-diagram-q%d-Re%d.png" % (save_directory, version, ten_q, logReynolds)
    plot.savefig(save_fn, bbox_inches = 'tight', dpi = dpi)

    if show:
        plot.show()

    plot.close(fig) # Close Figure (to avoid too many figures)


make_plot()

