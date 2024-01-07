"""
Latter+Papaloizou 2018: (Axisymmetric?) VSI in shearing box
This version is based on MKL's original code for hallSI.
"""

import numpy as np
import matplotlib.pyplot as plot

import scipy
from scipy.linalg import eigvals
from scipy.optimize import fsolve

from mpi4py import MPI
import dedalus.public as d3

import argparse
import logging
logger = logging.getLogger(__name__)


'''
Units
'''
Omega = 1.0
Hg    = 1.0
rhog0 = 1.0
mu0   = 1.0

cs    = Hg*Omega
cs2   = cs*cs

'''
Gas disk parameters
'''
etahat     = 0.1       #Reduced radial pressure gradient (=eta/smallh)
beta       = 1e5        #Plasma beta parameter for (inverse) vertical field strength
Re         = 1e8        #Reynolds number for setting viscosity/diffusion
Schmidt = 1

'''
Secondary parameters
'''
nu     = Omega*Hg**2/Re    #Viscosity and diffusion coefficient
D = nu / Schmidt
q = 0.05 # Hg

'''
Perturbation parameters
(also used in plotting 2D contour plots of growth rates)
'''
#pert       = 'random' #'random' or 'eigen' pert
pert      ='eigen' #'random' or 'eigen' pert
pert_amp   = 1e-4    #pert amplitude in terms of dvgy/cs

# if (feedback == 'no') and (pert != 'random') and (pert_amp != 0.0):
#     #linearized equations/eigenmode calculation always accounts for feedback!
#     print("no feedback should be used with random or zero perturbations")
#     quit()

kx_pert    = 40
kz_pert    = 5 #21#16#21#25.2

'''
Box size, resolution, MPI mesh
'''
if pert == 'eigen':
    lambda_x      = 2.0*np.pi/np.abs(kx_pert)
    lambda_z      = 2.0*np.pi/np.abs(kz_pert)
    Lx, Ly, Lz    = lambda_x, lambda_x, lambda_z
else:
    Lbox          = 2*Hg
    Lx, Ly, Lz    = Lbox, Lbox, Lbox
    low_pass_scales = 0.25

Nx, Ny, Nz    = 256, 2, 256

axi = True #are we using the special code for strictly axisymmetric flow?
if axi == False:
    mesh          = (4,2)
else:
    mesh = None

'''
Time integration and output cadence
'''
timestepper   = d3.RK443
#timestepper   = d3.RK222
cfl_number    = 0.2
min_timestep  = 1e-4/Omega
#if approx == 'exact':
max_timestep  = 1e-1/Omega

period        = 2.0*np.pi     #One orbit is 2pi in code units. temp: rescale time by tstop
stop_sim_time = 100          #in orbits (for convenience)
stop_sim_time*= period        #in inverse Omega (used in code)

snapshot_dt       = 10*period
analysis_dt       = 0.2*period

checkpoint_dt     = 10*period

OutputRes  = 512
OutputScale= OutputRes/Nx

'''
For restarting
'''
parser = argparse.ArgumentParser()
parser.add_argument("--restart", nargs='*', help="give checkpoint number")
args = parser.parse_args()
if(args.restart):
    restart = args.restart[0]
else:
    restart = None

'''
MPI stuff
'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

'''
Print problem parameters
'''
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False
logger.info('*******************')
logger.info('Problem parameters ')
logger.info('*******************')
logger.info("etahat     =%4.2f" % etahat)
logger.info("Reynolds   =%4.2f" % Re)

dealias       = 3.0/2.0
dtype         = np.float64

# Bases
coords  = d3.CartesianCoordinates('x', 'z')
dist    = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis  = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
zbasis  = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Fields
'''
vg = meridional gas velocity, relative to pure gas, radially and vertically shearing flow, so vg_eqm is non-zero
vgy= azimuthal gas velocity
W  = Pi/rhog0 (total pressure/gas density)
A  = meridional magnetic vector potential
Ay = azimuthal magnetic vector potential
'''
vg      = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
vgy     = dist.Field(name='uy', bases=(xbasis, zbasis))
tau_P   = dist.Field(name='tau_P')
p      = dist.Field(name='p',bases=(xbasis, zbasis))

#Coordinate axes and unit vectors
x, z    = dist.local_grids(xbasis, zbasis)
ex, ez  = coords.unit_vector_fields(dist)

#Substitutions for convenience. Note: fd, fg, eps are eqm. values set in dvsi_params. fdust, fgas, epsilon are live values
vgx   = vg@ex
vgz   = vg@ez

# Problem
problem = d3.IVP([vg, vgy, p, tau_P], namespace=locals())
problem.add_equation("dt(vg) + grad(p) / rhog0 - 2 * Omega * vgx * ex - nu*lap(vg) = -vg@grad(vg)")
#problem.add_equation("dt(vgy) + grad(p) / rhog0 + Omega * (0.5 * vgx - q * vgz) - nu*lap(vgy) = -vg@grad(vgy)")
problem.add_equation("dt(vgy) + Omega * (0.5 * vgx - q * vgz) - nu*lap(vgy) = -vg@grad(vgy)")
#problem.add_equation("dt(vgy) = 0")
problem.add_equation("div(vg) + tau_P = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver               = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

########################################
# Copied from my own version, not mkl's version

# Initial conditions
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
        matrix_a[0] = np.array([0, ikx, 0, ikz])
        matrix_a[1] = np.array([ikx - dissipation, 0, 2.0 * Omega, 0])
        matrix_a[2] = np.array([0, -0.5 * Omega - dissipation, 0, q * Omega])
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

    if eigenvalue > 0:
        norm = eigenvector[2] # azimuthal gas velocity
        eigenvector *= pert_amp / norm * Hg * Omega # fix units
    if pert_amp == 0.0:
        eigenvector *= 0.0

    return (eigenvalue, eigenvector)

########################################

if restart == None: #Fresh run
    if pert == 'eigen':
        # Eigenmode perturbation
        growth, eigenvector = OneFluidEigen(kx_pert, kz_pert)
        expik      = np.cos(kx_pert*x + kz_pert*z) + 1j*np.sin(kx_pert*x + kz_pert*z)

        dp   = eigenvector[0]
        dvgx = eigenvector[1]
        dvgy = eigenvector[2]
        dvgz = eigenvector[3]

        
        vg['g'][0] = np.real(dvgx*expik)
        vgy['g']   = np.real(dvgy*expik)
        vg['g'][1] = np.real(dvgz*expik)
        p['g']  = np.real(dp*expik)

    elif pert == 'random':

        #random pert in vgy
        vgy.fill_random('g', seed=42, distribution='normal', scale=pert_amp*cs)
        vgy.low_pass_filter(scales=low_pass_scales)
        vg['g'] = 0.0
        p['g'] = 0.0


    file_handler_mode = 'overwrite'
    initial_timestep = min_timestep

# Hydro outputs
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshot_dt, max_writes=1, mode=file_handler_mode)
snapshots.add_task(vgx, name='vgx', scales=OutputScale)
snapshots.add_task(vgy, name='vgy', scales=OutputScale)
snapshots.add_task(vgz, name='vgz', scales=OutputScale)
snapshots.add_task(p, name='p', scales=OutputScale)

# Restart files
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=checkpoint_dt, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL condition
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=cfl_number, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep, min_dt=min_timestep) #should include min_dt
CFL.add_velocity(vg)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(vgx**2 + vgy**2 + vgz**2), name='dvg')
flow.add_property(vgx**2,name='dvgx2')
flow.add_property(vgy**2,name='dvgy2')
flow.add_property(vgz**2,name='dvgz2')
flow.add_property(vgx**2 + vgy**2 + vgz**2,name='dvg2')
flow.add_property(vgx*vgy,name='dvgxdvgy')

#Prep for analysis output (only first CPU's job)
if rank == 0:
    if restart == None: #Fresh run
        output = open('analysis.txt', 'w')
    else:
        output = open('analysis.txt', 'a')
comm.Barrier()

# Main loop
try:
    logger.info('Starting main loop')
    log_time = solver.sim_time
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        if (solver.iteration-1) % 10 == 0:
            max_dvg  = flow.max('dvg')
            logger.info('Orbits=%4.2e, dt=%4.2e, max(dvg)=%4.2e'\
                %(solver.sim_time/period, timestep, max_dvg))
            if np.isnan(max_dvg) == True:
                logger.error('max(dvg) = NaN, abort')
                sys.exit()
        if solver.sim_time >= log_time:
            #maximum velocities, d/g pert
            max_dvg = flow.max('dvg')
            max_eps = 0.0
            min_eps = 0.0
            #rms velocities
            rms_vg  = np.sqrt(flow.grid_average('dvg2'))
            rms_vgx = np.sqrt(flow.grid_average('dvgx2'))
            rms_vgy = np.sqrt(flow.grid_average('dvgy2'))
            rms_vgz = np.sqrt(flow.grid_average('dvgz2'))
            #angular momentum flux
            amflux  = flow.grid_average('dvgxdvgy')
            amfluxd = 0.0

            #write to analysis by CPU 0
            if rank == 0:
                output.write("{0:9.6e}, {1:9.6e}, {2:9.6e}, {3:9.6e}, {4:9.6e}, {5:9.6e}, {6:9.6e}, {7:9.6e}, {8:9.6e}, {9:9.6e}".format(solver.sim_time, \
                    max_dvg, max_eps, rms_vg, rms_vgx, rms_vgy, rms_vgz, amflux, amfluxd, min_eps)+"\n")
                output.flush()
            comm.Barrier()

            log_time += analysis_dt
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

if rank == 0:
    output.close()

        