"""
Latter+Papaloizou 2018: (Axisymmetric?) VSI in shearing box
This version is based on my modifications to MKL's original code for hallSI,
which is bad because the modifications were not fully tested.
"""

import numpy as np
import matplotlib.pyplot as plot

import scipy
from scipy.linalg import eigvals
from scipy.optimize import fsolve

from mpi4py import MPI
import dedalus.public as d3

import logging
logger0 = logging.getLogger(__name__)

restart = None

# Units
Omega0 = 1.0
Hg = 1.0
rhog0 = 1.0
mu0 = 1.0

cs = Hg * Omega0
cs2 = cs * cs

# Gas disk parameters
eta_hat = 0.1 # Reduced radial pressure gradient (= eta / h)
beta = 1e5 # Plasma beta parameter for (inverse) vertical field strength
Ha = 1e4 # Hall parameter
Reynolds = 1e5 # Reynolds number for setting viscosity / diffusion
Schmidt = 1

ReynoldsM = Reynolds # Magnetic Reynolds number

# Secondary parameters
nu = Omega0 * Hg**2 / Reynolds # Viscosity and diffusion coefficient
D = nu / Schmidt
vertical_shear_q = 0.0 # Ignore for now
q = 1.0e-6 # Roberts q

va2 = 2.0 * cs2 / beta # Alfven velocity squared
Bz0 = np.sqrt(va2 * mu0 * rhog0) # Equilibrium vertical field strength
big_lambda = 1.0e16

nuM = Omega0 * Hg**2 / ReynoldsM # Resistivity

# IS THIS CORRECT???
vgx0 = 0.0
vgy0 = 0.0

# Variable parameters
#big_lambda = 1.0e16
N_squared = -0.01 # or 0.0

# Problem parameters
kappa_squared = 1.0
omega_squared = 1.0
omega_power = -1.5
omega_squared_power = -3.0
eta = 2.34e17

xi = q * eta
alfven_velocity_squared = big_lambda * eta

# Grid
pert_amp = 1e-5 # perturbation amplitude (in units of dvgy / cs)

kx_pert = 0.1 / alfven_velocity_squared
kz_pert = 10.0 / alfven_velocity_squared

# Box size, resolution, MPI mesh
pert = 'eigen' # 'random' or 'eigen'

if pert == 'eigen':
    lambda_x = 2.0 * np.pi / np.abs(kx_pert) # lambda_x and lambda_y? Is that on purpose?
    lambda_z = 2.0 * np.pi / np.abs(kz_pert)
    Lx, Ly, Lz = lambda_x, lambda_x, lambda_z

else:
    Lbox = 2.0 * np.pi * Hg
    Lx, Ly, Lz = Lbox, Lbox, Lbox
    low_pass_scales = 0.25

Nx, Ny, Nz = 64, 2, 64
#Nx, Ny, Nz = 256, 2, 256
#Nx, Ny, Nz = 1024, 2, 1024

axi = True # axisymmetric flow
if axi == False:
    mesh = (4, 2)
else:
    mesh = None

# Time integration and output cadence
timestepper = d3.RK443 # RK443 or RK222
cfl_number = 0.1 # 0.2
min_timestep = 1e-2 / Omega0
max_timestep = 1e-2 / Omega0

period = 2.0 * np.pi
stop_sim_time = 100 # in orbits
stop_sim_time *= period

snapshot_dt = 1.0 * period
analysis_dt = 0.2 * period

checkpoint_dt = 10.0 * period

OutputRes = 256
OutputScale = OutputRes / Nx

### MPI ###
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dealias = 3.0 / 2.0
dtype = np.float64

# Print problem parameters
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger0.addHandler(ch)
logger0.propagate = False
logger0.info('*******************')
logger0.info('Problem parameters ')
logger0.info('*******************')
logger0.info("etahat     =%4.2f" % eta_hat)
logger0.info("Reynolds   =%4.2f" % Reynolds)

### Bases ###
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype = dtype, mesh = mesh)
x_basis = d3.RealFourier(coords['x'], size = Nx, bounds = (-Lx / 2, Lx / 2), dealias = dealias) # why isn't size "Nx"?
z_basis = d3.RealFourier(coords['z'], size = Nz, bounds = (-Lz / 2, Lz / 2), dealias = dealias)

# Fields
u = dist.VectorField(coords, name='u', bases=(x_basis,z_basis))
uy = dist.Field(name='uy', bases=(x_basis,z_basis))
theta = dist.Field(name='theta', bases=(x_basis,z_basis))
tau_P = dist.Field(name='tau_P')
p = dist.Field(name='p', bases=(x_basis,z_basis))

A = dist.VectorField(coords, name = 'A', bases = (x_basis, z_basis)) # Meridional Mangetic Vector Potential
Ay = dist.Field(name = 'Ay', bases = (x_basis, z_basis)) # Azimuthal Magnetic Vector Potential
phi = dist.Field(name = 'phi', bases = (x_basis, z_basis))
tau_phi = dist.Field(name = 'tau_phi') # why no bases here?

### Coordinate axes and unit vectors ###

x, z = dist.local_grids(x_basis, z_basis)
ex, ez = coords.unit_vector_fields(dist)

### Substitutions for convenience ###
ux = u@ex
uz = u@ez

### Mangetic field substitutions ###

dx = lambda f : d3.Differentiate(f, coords['x'])
dz = lambda f : d3.Differentiate(f, coords['z'])
del2 = lambda f : dx(dx(f)) + dz(dz(f))

Ax = A@ex
Az = A@ez
lapAx = d3.Laplacian(Ax)
lapAz = d3.Laplacian(Az)
lapAy = d3.Laplacian(Ay)

DBx = -dz(Ay)
DBy = -dx(Az) + dz(Ax)
DBz = dx(Ay)

Bx = DBx
By = DBy
Bz = Bz0 + DBz

Bxz = Bx*ex + Bz*ez
DBxz = DBx*ex + DBz*ez

Hall1xz = -Bz0 * lapAy * ex
Hall1y = Bz0 * lapAx

Hall2xz = -ez*lapAx*DBy + ex*lapAz*DBy + ez*lapAy*DBx - ex*lapAy*DBz
Hall2y = lapAx*DBz - lapAz*DBx

Hallxz = Hall1xz + Hall2xz # (curlB cross B) * (1, 0, 1)
Hally = Hall1y + Hall2y # (curlB cross B) * (0, 1, 0)

#Halltermxz = -(etaHall / Bz0) * Hallxz
#Halltermy = -(etaHall / Bz0) * Hally

vcrossBxz = Bz0*uy*ex + ex*(uy*DBz - uz*DBy) + ez*(ux*DBy - uy*DBx)
vcrossBy = -Bz0*ux + (uz*DBx - ux*DBz)

alfven_velocity = Bxz / np.sqrt(mu0 * rhog0)

# Problem
problem = d3.IVP([u, uy, theta, p, tau_P, A, Ay, phi, tau_phi], namespace=locals())
problem.add_equation("dt(u) + grad(p) / rhog0 - 2 * Omega0 * uy * ex + N_squared * theta * ex - nu*lap(u) = Bxz@grad(DBxz)/(mu0*rhog0) -u@grad(u)")
problem.add_equation("dt(uy) + Omega0 * (0.5 * ux - vertical_shear_q * uz) - nu*lap(uy) = Bxz@grad(DBy)/(mu0*rhog0) -u@grad(uy)")

problem.add_equation("dt(theta) - ux - xi*lap(theta) = -u@grad(theta)")

problem.add_equation("div(u) + tau_P = 0")
problem.add_equation("integ(p) = 0")

problem.add_equation("dt(A) - (3.0/2.0)*Omega0*Ay*ex - nuM*lap(A) - grad(phi) = vcrossBxz + vgx0*DBy*ez")
problem.add_equation("dt(Ay) - nuM*lapAy = vcrossBy - vgx0*DBz")
problem.add_equation("div(A) + tau_phi = 0")
problem.add_equation("integ(phi) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
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

        #[dP/rhog0, dvgx, dvgy, dBx, dBy, dtheta]
        matrix_a[0] = np.array([0, 2, ikz, 0, -N_squared / omega_squared])
        matrix_a[1] = np.array([-0.5, 0, 0, ikz, 0])
        matrix_a[2] = np.array([ikz, 0, -ksq / big_lambda, 0, 0])
        matrix_a[3] = np.array([0, ikz, omega_power, -ksq / big_lambda, 0])
        matrix_a[4] = np.array([1, 0, 0, 0, -q * ksq / big_lambda])

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

    def Teed2021_broken(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((4, 4), dtype = np.cdouble)
        matrix_b = np.diag([1,1,1,1,1])

        dissipation = nu * ksq

        #[dvgx, dvgy, dtheta]
        matrix_a[0] = np.array([0, 2, -N_squared / omega_squared])
        matrix_a[1] = np.array([-0.5, 0, 0])
        matrix_a[2] = np.array([1, 0, -xi * ksq * kappa_squared])

        return (matrix_a, matrix_b)

    def Latter2018(self):
        kx = self.kx; kz = self.kz
        ikx = 1j * kx; ikz = 1j * kz
        ksq = self.kx**2 + self.kz**2

        matrix_a = np.zeros((4, 4), dtype = np.cdouble)
        matrix_b = np.diag([0,1,1,1])

        dissipation = nu * ksq

        # [dP/rhog0, dux, duy, duz]
        matrix_a[0] = np.array([0, ikx, 0, ikz])
        matrix_a[1] = np.array([-ikx, -dissipation, 2.0 * Omega0, 0])
        matrix_a[2] = np.array([0, -0.5 * Omega0, -dissipation, q * Omega0])
        matrix_a[3] = np.array([-ikz, 0, 0, -dissipation])

        return (matrix_a, matrix_b)

def OneFluidEigen(kx, kz):
    matrix = OneFluidMatrices(kx, kz)
    a, b = matrix.Latter2010()

    eigenvalues, eigenvectors = scipy.linalg.eig(a, b)

    growth = eigenvalues.real
    growth[growth == np.inf] = -np.inf

    gmax = np.argmax(growth)
    eigenvalue = eigenvalues[gmax]
    eigenvector = eigenvectors[:, gmax]

    print("eigenvalue=",eigenvalue)
    print("eigenvector=",eigenvector)

    if eigenvalue > 0:
        norm = eigenvector[2] # azimuthal gas velocity
        eigenvector *= pert_amp / norm * Hg * Omega0 # fix units
    if pert_amp == 0.0:
        eigenvector *= 0.0

    print("normalized eigenvector=",eigenvector)

    return (eigenvalue, eigenvector)

if pert == 'eigen':
    # Eigenmode perturbation
    growth, eigenvector = OneFluidEigen(kx_pert, kz_pert)
    expik = np.cos(kx_pert * x + kz_pert * z) + 1j * np.sin(kx_pert * x + kz_pert * z)

    #dp  = eigenvector[0]*rhog0
    dux = eigenvector[0]
    duy = eigenvector[1]
    dBx = eigenvector[2]
    dBy = eigenvector[3]
    dtheta = eigenvector[4]

    ksq = kx_pert**2 + kz_pert**2
    dAx = -1j * kz_pert * dBy / ksq
    dAy = -1j * (-kz_pert * dBx) / ksq # (kx_pert * dBz - kz_pert * dBx) / ksq
    #dAz = -1j * kx_pert * dBy / ksq

    u['g'][0] = np.real(dux * expik)
    uy['g'] = np.real(duy * expik)
    #u['g'][1] = np.real(duz * expik)
    p['g'] = rhog0 #np.real(dp * expik)
    theta['g'] = np.real(dtheta * expik)

    A['g'][0] = np.real(dAx * expik)
    Ay['g'] = np.real(dAy * expik)
    #A['g'][1] = np.real(dAz * expik)

elif pert == 'random':
    # Random perturbation in vgy
    uy.fill_random('g', seed = 42, distribution = 'normal', scale = pert_amp * cs)
    uy.low_pass_filter(scales = low_pass_scales)
    u['g'] = 0.0 # All equilibrium velocities are zero
    p['g'] = 0.0

file_handler_mode = 'overwrite'
initial_timestep = min_timestep

# CFL condition
CFL = d3.CFL(solver, initial_dt = initial_timestep, cadence = 10, safety = cfl_number, threshold = 0.1,
            max_change = 1.5, min_change = 0.5, max_dt = max_timestep, min_dt = min_timestep) # should include min_dt
CFL.add_velocity(u)
CFL.add_velocity(alfven_velocity)

# Hydro outputs
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt = snapshot_dt, max_writes = 1, mode = file_handler_mode)
snapshots.add_task(ux, name = 'ux', scales = OutputScale)
snapshots.add_task(uy, name = 'uy', scales = OutputScale)
snapshots.add_task(uz, name = 'uz', scales = OutputScale)
snapshots.add_task(p, name = 'p', scales = OutputScale)
snapshots.add_task(Bxz@ex, name = 'Bx', scales = OutputScale)
snapshots.add_task(By, name = 'By', scales = OutputScale)
snapshots.add_task(Bxz@ez, name = 'Bz', scales = OutputScale)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(ux**2 + uy**2 + uz**2), name = 'du')
flow.add_property(ux**2, name = 'dux2')
flow.add_property(uy**2, name = 'duy2')
flow.add_property(uz**2, name = 'duz2')
flow.add_property(ux**2 + uy**2 + uz**2, name = 'du2')
flow.add_property(ux * uy, name = 'duxduy')

# Prep for analysis output (only first CPU's job)
if rank == 0:
    if restart == None: #Fresh run
        output = open('analysis.txt', 'w')
    else:
        output = open('analysis.txt', 'a')
comm.Barrier()

# Main loop
try:
    logger0.info('Starting main loop')
    log_time = solver.sim_time

    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration - 1) % 10 == 0:
            max_du = flow.max('du')
            logger0.info('Orbits = %4.2e, dt = %4.2e, max(du) = %4.2e' % (solver.sim_time / period, timestep, max_du))
            if np.isnan(max_du) == True:
                logger0.error('max(du) = NaN, abort')

        if solver.sim_time >= log_time:
            #maximum velocities
            max_du = flow.max('du')

            #rms velocities
            rms_du  = np.sqrt(flow.grid_average('du2'))
            rms_dux = np.sqrt(flow.grid_average('dux2'))
            rms_duy = np.sqrt(flow.grid_average('duy2'))
            rms_duz = np.sqrt(flow.grid_average('duz2'))

            #write to analysis by CPU 0
            if rank == 0:
                output.write("{0:9.6e}, {1:9.6e}, {2:9.6e}, {3:9.6e}, {4:9.6e}, {5:9.6e}".format(solver.sim_time, max_du, rms_du, rms_dux, rms_duy, rms_duz)+"\n")
                output.flush()
            comm.Barrier()

            log_time += analysis_dt

except:
    logger0.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

