"""

"""

import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import fsolve
import dedalus.public as d3
import logging
import scipy
from scipy.linalg import eigvals

logger0 = logging.getLogger(__name__)

# Units
Omega = 1.0
Hg = 1.0
rhog0 = 1.0
mu0 = 1.0

cs = Hg * Omega
cs2 = cs * cs

# Gas disk parameters
etahat = 0.1 # Reduced radial pressure gradient (= eta / h)
beta = 1e5 # Plasma beta parameter for (inverse) vertical field strength
Ha = 1e4 # Hall parameter
Re = 1e8 # Reynolds number for setting viscosity / diffusion
ReM = 1e8 # Magnetic Reynolds number

# Dust parameters
eps = 0.2 # Initial dust / gas ratio
st = 0.1 # Particle size or Stokes number

# Secondary parameters
va2 = 2.0 * cs2 / beta # Alfven velocity squared
Bz0 = np.sqrt(va2 * mu0 * rhog0) # Equilibrium vertical field strength
etaHall = va2 / (2.0 * Omega * Ha) # Hall diffusion coefficient
vscale = etahat * Hg * Omega # Radial drift velocity scale, eta * r * Omega

fd = eps / (1.0 + eps) # Initial dust fraction
fg = 1.0 - fd # Initial gas fraction
tstop = st / Omega # Stopping time (in physical units)

nu = Omega * Hg**2 / Re # Viscosity and diffusion coefficient
nuM = Omega * Hg**2 / ReM # Resistivity

# Exact formulation or improved TVA
approx = 'exact'

# Do we include feedback?
feedback = 'yes'

# Equilibrium / initial conditions (for mixed formulation)
# In mixed form, the gas velocities are measured relative to the dust-free, unperturbed flow

if feedback == 'yes':
    if approx == 'tva':
        vgx0 = 2.0 * fd * fg * st * vscale
        vgy0 = fd * scale # relative to pure gas sub-Keplerian sheaer

        vdx0 = vgx0 - 2.0 * vscale * st * fg
        vfy0 = vgy0 + vscale * fg * fg * st * st

    if approx == 'exact':
        Dsq = st*st + (1.0 + eps)**2
        vgx0 = 2.0 * eps * st * vscale / Dsq
        vgy0 = eps * (1.0 + eps) * vscale / Dsq # relative to pure gas sub-Keplerian shear

        ux0 = -2.0 * st * (1.0 + eps) * vscale / Dsq
        uy0 = st * st * vscale / Dsq

        vdx0 = vgx0 + ux0
        vdy0 = vgy0 + uy0

elif feedback == 'no':
    vgx0 = 0.0
    vgy0 = 0.0
    if approx == 'tva':
        vdx0 = vgx0 - 2.0 * vscale * st
        vdy0 = vgy0 + vscale * st * st
    if approx == 'exact':
        Dsq = 1.0 + st*st
        ux0 = -2.0 * st * vscale / Dsq
        uy0 = st *st * vscale / Dsq

# Simplify the Hall term?
SimplifyHallTerm = True

# Special case of toy model of Hall SI
HallSIToy = True

# Perturbation parameters (also used in 2-D contour plots of growth rates)
pert = 'eigen' # 'random' or 'eigen'
pert_amp = 1e-4 # perturbation amplitude (in units of dvgy / cs)

kx_pert = 40
kz_pert = 5

# Box size, resolution, MPI mesh
if pert == 'eigen':
    lambda_x = 2.0 * np.pi / np.abs(kx_pert) # lambda_x and lambda_y? Is that on purpose?
    lambda_z = 2.0 * np.pi / np.abs(kz_pert)
    Lx, Ly, Lz = lambda_x, lambda_x, lambda_z

else:
    Lbox = 2.0 * Hg
    Lx, Ly, Lz = Lbox, Lbox, Lbox
    low_pass_scales = 0.25

Nx, Ny, Nz = 256, 2, 256
#Nx, Ny, Nz = 1024, 2, 1024

axi = True # axisymmetric flow
if axi == False:
    mesh = (4, 2)
else:
    mesh = None

# Time integration and output cadence
timestepper = d3.RK443 # RK443 or RK222
cfl_number = 0.2
min_timestep = 1e-4 / Omega
if approx == 'exact':
    max_timestep = 1e-1 / Omega
else:
    max_timestep = 1e-1 / Omega

period = 2.0 * np.pi
stop_sim_time = 100.0 # in orbits
stop_sim_time *= period

snapshot_dt = 10.0 * period
analysis_dt = 0.2 * period

checkpoint_dt = 10.0 * period

OutputRes = 512
OutputScale = OutputRes / Nx

# Print problem parameters
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger0.addHandler(ch)
logger0.propagate = False
logger0.info('*******************')
logger0.info('Problem parameters ')
logger0.info('*******************')
logger0.info("etahat     =%4.2f" % etahat)
logger0.info("beta       =%4.2f" % beta)
logger0.info("Ha         =%4.2f" % Ha)

logger0.info("Reynolds   =%4.2f"%Re)
logger0.info("ReynoldsM  =%4.2f"%ReM)

logger0.info("epsilon    =%4.2f" % eps)
logger0.info("stokes     =%4.2f" % st)

# For plotting growth rates
kmin = 1e0 / Hg
kmax = 1e3 / Hg
nkx = 128
nkz = 128

kx_array = np.logspace(np.log10(kmin), np.log10(kmax), nkx)
kz_array = np.logspace(np.log10(kmin), np.log10(kmax), nkz)

# Plotting
fontsize = 24
nlev = 64
nclev = 6
cmap = plot.cm.jet

minv = -5
maxv = 0

levels = np.linspace(minv, maxv, nlev)
clevels = np.linspace(minv, maxv, nclev)

# Lineared equations in matrix form

class OneFluidMatrices:
    def __init__(self, kx, kz):
        self.kx = kx
        self.kz = kz

    def TerminalApproxMixedImproved(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((8, 8), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1,1,1,1])

        dissipation = nu * ksq
        dissipationM = nuM * ksq

        # [dP/rhog0, deps, dvgx, dvgy, dvgz, dBx, dBy, dBz]
        matrix_a[0] = np.array([0, 0, ikx, 0, ikz, 0, 0, 0])
        matrix_a[1] = np.array([tstop*ksq, -ikx * vgx0 - dissipation, 0, 2 * st * ikx, 0, 0, 0, 0])
        matrix_a[2] = np.array([-ikx * fg, -2 * vscale * Omega * fg**2, -dissipation, 2 * Omega, 0, fg * ikz * Bz0 / mu0 / rhog0, -st * fd * fg * 2 * ikz * Bz0 / mu0 / rhog0, 0])
        matrix_a[3] = np.array([-st * fd * fg * 0.5 * ikx, st * vscale * Omega * fg**3 * (1.0 - eps), -0.5 * Omega, -dissipation, 0, 0.5 * st * fd * fg * ikz * Bz0 / mu0 / rhog0, fg * ikz * Bz0 / mu0 / rhog0, 0])
        matrix_a[4] = np.array([-ikz * fg, 0, 0, 0, -dissipation, 0, 0, fg * ikz * Bz0 / mu0 / rhog0])
        matrix_a[5] = np.array([0, 0, ikz * Bz0, 0, 0, -ikx * vgx0 - dissipationM, -etaHall * kz * kz, 0])
        matrix_a[6] = np.array([0, 0, 0, ikz * Bz0, 0, etaHall * ksq - (3.0 / 2.0) * Omega, -ikx * vgx0 - dissipationM, 0])
        matrix_a[7] = np.array([0, 0, 0, 0, ikz * Bz0, 0, etaHall * kx * kz, -ikx * vgx0 - dissipationM])

        return (matrix_a, matrix_b)

    def ExactOneFluid(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((11, 11), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1,1,1,1,1,1,1])

        dissipation = nu * ksq
        dissipationM = nuM * ksq

        # [dP/rhog0, deps, dvgx, dvgy, dvgz, dBx, dBy, dBz, dux, duy, duz]
        matrix_a[0] = np.array([0, 0, ikx, 0, ikz, 0, 0, 0, 0, 0, 0])
        matrix_a[1] = np.array([tstop * ksq, -ikx * vgx0 - dissipation, 0, 2 * st * ikx, 0, 0, 0, 0, 0, 0, 0])
        matrix_a[2] = np.array([-ikx, ux0 / tstop, -ikx * vgx0 - dissipation, 2 * Omega, 0, ikz * Bz0 / mu0 / rhog0, 0, 0, eps / tstop, 0, 0])
        matrix_a[3] = np.array([0, uy0 / tstop, -0.5 * Omega, -ikx * vgx0 - dissipation, 0, 0, ikz * Bz0 / mu0 / rhog0, 0, 0, eps / tstop, 0])
        matrix_a[4] = np.array([-ikz, 0, 0, 0, -ikx * vgx0 - dissipation, 0, 0, ikz * Bz0 / mu0 / rhog0, 0, 0, eps / tstop])
        matrix_a[5] = np.array([0, 0, ikz * Bz0, 0, 0, -ikx * vgx0 - dissipationM, -etaHall * kz * kz, 0, 0, 0, 0])
        matrix_a[6] = np.array([0, 0, 0, ikz * Bz0, 0, etaHall * ksq - (3.0 / 2.0) * Omega, -ikx * vgx0 - dissipationM, 0, 0, 0, 0])
        matrix_a[7] = np.array([0, 0, 0, 0, ikz * Bz0, 0, etaHall * kx * kz, -ikx * vgx0 - dissipationM, 0, 0, 0])
        matrix_a[8] = np.array([ikx, -ux0 / tstop, -ikx * ux0, 0, 0, -ikz * Bz0 / mu0 / rhog0, 0, 0, -(ikx * vdx0 + (1.0 + eps) / tstop) - dissipation, 2 * Omega, 0])
        matrix_a[9] = np.array([0, -uy0 / tstop, 0, -ikx * ux0, 0, 0, -ikz * Bz0 / mu0 / rhog0, 0, -0.5 * Omega, -(ikx * vdx0 + (1.0 + eps) / tstop) - dissipation, 0])
        matrix_a[10] = np.array([ikz, 0, 0, 0, -ikz * ux0, 0, 0, -ikz * Bz0 / mu0 / rhog0, 0, 0, -(ikx * vdx0 + (1.0 + eps) / tstop) - dissipation])

        return (matrix_a, matrix_b)

    def HallSIToyModel(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((7, 7), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1,1,1,1])

        dissipation = nu * ksq
        dissipationM = nuM * ksq

        # [dP/rhog0, dvgx, dvgy, dvgz, dBx, dBy, dBz]
        matrix_a[0] = np.array([0, ikx, 0, ikz, 0, 0, 0])
        matrix_a[1] = np.array([-ikx * fg, -dissipation, 2 * Omega, 0, fg * ikz * Bz0 / mu0 / rhog0, 0, 0])
        matrix_a[2] = np.array([0, -0.5 * Omega, -dissipation, 0, 0, fg * ikz * Bz0 / mu0 / rhog0, 0])
        matrix_a[3] = np.array([-ikz * fg, 0, 0, -dissipation, 0, 0, fg * ikz * Bz0 / mu0 / rhog0])
        matrix_a[4] = np.array([0, ikz * Bz0, 0, 0, -ikx * vgx0 - dissipationM, -etaHall * kz * kz, 0])
        matrix_a[5] = np.array([0, 0, ikz * Bz0, 0, etaHall * ksq - (3.0 / 2.0) * Omega, -ikx * vgx0 - dissipationM, 0])
        matrix_a[6] = np.array([0, 0, 0, ikz * Bz0, 0, etaHall * kx * kz, -ikx * vgx0 - dissipationM])

        return (matrix_a, matrix_b)

    def HallSIToyModelGrowthPoly(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        c3 = 2 * ikx * vgx0
        c2 = kz * kz / ksq * Omega + 2 * fg * kz * kz * va2 + etaHall * kz * kz * (etaHall * ksq - 3 * Omega / 2)
        c1 = 2 * ikx * vgx0 * kz * kz / ksq * (Omega**2 + fg * ksq * va2)
        c0 = (fg * kz * va2)**2 * ksq - 3 * fg * kz**2 * va2 * Omega**2 + (etaHall * kz * Omega)**2 * ksq + 0.5 * etaHall * kz**2 * Omega * (5 * fg * ksq * va2 - 3 * Omega ** 2)
        c0 -= (kx * vgx0 * Omega)**2 # drift effect
        c0 *= kz**2 / ksq

        # Toy model growth rate from characteristic polynomial
        roots = np.roots([1, c3, c2, c1, c0])

        return roots


def OneFluidEigen(kx, kz):
    matrix = OneFluidMatrices(kx, kz)
    if approx == 'tva':
        a, b = matrix.TerminalApproxMixedImproved()
    if approx == 'exact':
        a, b = matrix.ExactOneFluid()
    if HallSIToy == True:
        a, b = matrix.HallSIToyModel()

    eigenvalues, eigenvectors = scipy.linalg.eig(a, b)

    growth = eigenvalues.real
    growth[growth == np.inf] = -np.inf

    gmax = np.argmax(growth)
    eigenvalue = eigenvalues[gmax]
    eigenvector = eigenvectors[:, gmax]

    if eigenvalue > 0:
        norm = eigenvector[3] # azimuthal gas velocity
        eigenvector *= pert_amp / norm * Hg * Omega # fix units
    if pert_amp == 0.0:
        eigenvector *= 0.0

    return (eigenvalue, eigenvector)


def OneFluidEigenMaxGrowth(kx, kz):
    eigenvalue, eigenvector = OneFluidEigen(kx, kz)
    return eigenvalue.real

class GrowthRates:
    def twodim(growthfunc):
        kx2d, kz2d = np.meshgrid(kx_array, kz_array)
        GrwothRates2D = np.vectorize(growthfunc)(kx2d, kz2d)
        return GrwothRates2D

def PlotTwoDim(data2D, title, frame):
    data2D[data2D < 0.0] = 1e-9 # replace decaying modes with small positive growth for log plot

    plot.rc('font', size = fontsize, weight = 'bold')

    fig, ax = plot.subplots(constrained_layout = True)
    cp = plot.contourf(kx_array, kz_array, np.log10(data2D), levels, cmap = cmap)

    cbar = plot.colorbar(cp, ticks = clevels, format = '%.1f', pad = 0)
    cbar.set_label(r'$\log(s_\mathrm{max} / \Omega)$')

    ax.set_box_aspect(1)

    plot.xscale('log')
    plot.yscale('log')

    ax.set_ylim(kmin, kmax)
    ax.set_xlim(kmin, kmax)

    ax.set_xlabel(r'$k_x H_g$')
    ax.set_ylabel(r'$k_z H_g$', labelpad = 0)
    ax.set_title(title, weight = 'bold')

    NcellsPerLambda = 10 # max resolved wavenumber is ~Nx / 2
    max_kx = 2.0 * np.pi * Nx * Hg / (NcellsPerLambda * Lx)
    max_kz = 2.0 * np.pi * Nz * Hg / (NcellsPerLambda * Lz)

    min_kx = 2.0 * np.pi * Hg / Lx
    min_kz = 2.0 * np.pi * Hg / Lz

    plot.vlines(max_kx, min_kz, max_kz, linestyle = 'dotted', color = 'black')
    plot.vlines(max_kz, min_kx, max_kx, linestyle = 'dotted', color = 'black')

    plot.vlines(x = min_kx, ymin = min_kz, ymax = max_kz, linestyle = 'dotted', color = 'black')
    plot.hlines(y = min_kz, ymin = min_kx, ymax = max_kx, linestyle = 'dotted', color = 'black')

    plot.savefig('hallSI_' + fname, dpi = 150)


if __name__ == "__main__":
    # Theoretical grwoth rate for chosen kx_pert and kz_pert
    growth = OneFluidEigenMaxGrowth(kx_pert, kz_pert)

    logger0.info('*******************')
    logger0.info('Theoretical growth ')
    logger0.info('*******************')

    logger0.info("kx        =%4.2f" % kx_pert)
    logger0.info("kz        =%4.2f" % kz_pert)
    logger0.info("growth    =%4.2e" % growth)

    # Plot growth rates as a function of kx, kz
    Growth2D = GrowthRates.twodim(OneFluidEigenMaxGrowth)
    PlotTwoDim(Growth2D, r'Full model', 'growth')

    max_growth = np.amax(Growth2D)
    kopt = np.argmax(Growth2D)
    max_x, max_z = np.unravel_index(kopt, Growth2D.shape)

    kx2d, kz2d = np.meshgrid(kx_array, kz_array)

    logger0.info("max growth=%4.2e" % max_growth)
    logger0.info("opt     kx=%4.2e" % kx2d[max_x, max_z])
    logger0.info("opt     kz=%4.2e" % kx2d[max_x, max_z])