"""
Toy model of Hall SI: standard MHD equations but with
1) pressure and lorentz forces multiplied by fg
2) extra source term in induction equation to mimic advection of B perturbations by a background flow (but not in other eqns)

This version: enforce axisymmetry by splitting the meridional and azimuthal equations

Equations, numerical parameters, output.
"""

from hallSI_params import *
import argparse
logger = logging.getLogger(__name__)

from scipy.ndimage import uniform_filter1d

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
import os, shutil

'''
MPI stuff
'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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
vg      = dist.VectorField(coords, name='vg', bases=(xbasis, zbasis))
vgy     = dist.Field(name='vgy', bases=(xbasis, zbasis))
W       = dist.Field(name='W', bases=(xbasis, zbasis))
tau_W   = dist.Field(name='tau_W')
A       = dist.VectorField(coords, name='A', bases=(xbasis, zbasis))
Ay      = dist.Field(name='Ay',bases=(xbasis, zbasis))
phi     = dist.Field(name='phi', bases=(xbasis, zbasis))
tau_phi   = dist.Field(name='tau_phi')

#Coordinate axes and unit vectors
x, z    = dist.local_grids(xbasis, zbasis)
ex, ez  = coords.unit_vector_fields(dist)

#Substitutions for convenience. Note: fd, fg, eps are eqm. values set in dvsi_params. fdust, fgas, epsilon are live values
vgx   = vg@ex
vgz   = vg@ez

#For processing the magnetic field
dx      = lambda f: d3.Differentiate(f, coords['x'])
dz      = lambda f: d3.Differentiate(f, coords['z'])
del2    = lambda f: dx(dx(f)) + dz(dz(f))

Ax      = A@ex
Az      = A@ez
lapAx   = d3.Laplacian(Ax)
lapAz   = d3.Laplacian(Az)
lapAy   = d3.Laplacian(Ay)

# divA      = d3.Divergence(A)
# graddivA  = d3.Gradient(divA)
# graddivAx = graddivA@ex
# graddivAz = graddivA@ez

DBx     = -dz(Ay)
DBy     = -dx(Az) + dz(Ax)
DBz     =  dx(Ay)

Bx      = DBx
By      = DBy
Bz      = Bz0 + DBz

Bxz     = Bx*ex + Bz*ez
DBxz    = DBx*ex + DBz*ez

Hall1xz = -Bz0*lapAy*ex
Hall1y  =  Bz0*lapAx

Hall2xz = -ez*lapAx*DBy + ex*lapAz*DBy + ez*lapAy*DBx - ex*lapAy*DBz
Hall2y  = lapAx*DBz - lapAz*DBx

#from grad(div(A)) term (related to curl of B)
# Hall3xz = graddivAx*DBy*ez - graddivAz*DBy*ex
# Hall3y  = -graddivAx*Bz0 - graddivAx*DBz + graddivAz*DBx

Hallxz  = Hall1xz + Hall2xz #this is curl B cross B (x, z components)
Hally   = Hall1y + Hall2y #this is curl B cross B (y component)

Halltermxz  =-(etaHall/Bz0)*Hallxz
Halltermy   =-(etaHall/Bz0)*Hally

vcrossBxz  = Bz0*vgy*ex + ex*(vgy*DBz - vgz*DBy) + ez*(vgx*DBy - vgy*DBx)
vcrossBy   =-Bz0*vgx + (vgz*DBx - vgx*DBz)

valfven   = Bxz/np.sqrt(mu0*rhog0)

vars = [vg, vgy, W, tau_W, A, Ay, phi, tau_phi]
problem = d3.IVP(vars, namespace=locals())

#Gas incompressibility
problem.add_equation("div(vg) + tau_W = 0")
problem.add_equation("integ(W)  = 0")

#Gas momentum
problem.add_equation("dt(vg) + fg*grad(W) - 2*Omega*vgy*ex - nu*lap(vg) = fg*Bxz@grad(DBxz)/(mu0*rhog0) - vg@grad(vg)")
problem.add_equation("dt(vgy)+ 0.5*Omega*vgx - nu*lap(vgy) = fg*Bxz@grad(DBy)/(mu0*rhog0) - vg@grad(vgy)")

#Induction equation
problem.add_equation("dt(A) - (3.0/2.0)*Omega*Ay*ex -nuM*lap(A) - grad(phi) = vcrossBxz + Halltermxz + vgx0*DBy*ez")
problem.add_equation("dt(Ay) -nuM*lapAy = vcrossBy + Halltermy - vgx0*DBz")
problem.add_equation("div(A) + tau_phi =0")
problem.add_equation("integ(phi)  = 0")

# Solver
solver               = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

if restart == None: #Fresh run
    if pert == 'eigen':
        # Eigenmode perturbation
        growth, eigenvector = OneFluidEigen(kx_pert, kz_pert)
        expik      = np.cos(kx_pert*x + kz_pert*z) + 1j*np.sin(kx_pert*x + kz_pert*z)

        dW   = eigenvector[0]
        dvgx = eigenvector[1]
        dvgy = eigenvector[2]
        dvgz = eigenvector[3]
        dBx  = eigenvector[4]
        dBy  = eigenvector[5]
        dBz  = eigenvector[6]

        ksq = kx_pert**2 + kz_pert**2
        dAx = -1j*kz_pert*dBy/ksq
        dAy = -1j*(kx_pert*dBz - kz_pert*dBx)/ksq
        dAz =  1j*kx_pert*dBy/ksq

        W['g']     = np.real(dW*expik)
        vg['g'][0] = np.real(dvgx*expik)
        vgy['g']   = np.real(dvgy*expik)
        vg['g'][1] = np.real(dvgz*expik)
        A['g'][0]  = np.real(dAx*expik)
        Ay['g']    = np.real(dAy*expik)
        A['g'][1]  = np.real(dAz*expik)

    elif pert == 'random':

        #random pert in vgy
        vgy.fill_random('g', seed=42, distribution='normal', scale=pert_amp*cs)
        vgy.low_pass_filter(scales=low_pass_scales)
        vg['g'] = 0.0
        W['g'] = 0.0
        A['g'] = 0.0
        Ay['g']= 0.0

#Note: all equilibrium velocities are zero

    file_handler_mode = 'overwrite'
    initial_timestep = min_timestep
else: #restart simulation
    restart_file = 'checkpoints/checkpoints_s'+restart+'.h5'
    write, initial_timestep = solver.load_state(restart_file)
    initial_timestep  = min_timestep
    file_handler_mode = 'append'

    # we need to remove the last snapshots directory, to re-write it
    if rank == 0:
        snaps  = []
        ctimes = []
        for root, dirs, files in os.walk('./snapshots/'): #put required directory path
            for name in dirs:
                snap = os.path.join(root, name)
                ctime= os.stat(snap).st_mtime
                snaps.append(snap)
                ctimes.append(ctime)
        final_dir = snaps[np.argmax(ctimes)]
        final_h5  = final_dir+'.h5'
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        if os.path.exists(final_h5):
            os.remove(final_h5)

        #delete the appropriate lines in analysis.txt to ensure time always increasing
        trestart    = solver.sim_time
        analysis    = np.loadtxt('analysis.txt', delimiter=',')
        time        = analysis[:,0]
        g1          = np.argmin(np.abs(time-trestart))

        with open("analysis.txt", 'r+') as fp:
            lines = fp.readlines()
            fp.seek(0)
            fp.truncate()
            if trestart < time[g1]:
                fp.writelines(lines[:g1])
            if trestart > time[g1]:
                fp.writelines(lines[:g1+1])
    comm.Barrier()

# Hydro outputs
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshot_dt, max_writes=1, mode=file_handler_mode)
snapshots.add_task(vgx, name='vgx', scales=OutputScale)
snapshots.add_task(vgy, name='vgy', scales=OutputScale)
snapshots.add_task(vgz, name='vgz', scales=OutputScale)
snapshots.add_task(W, name='W', scales=OutputScale)
snapshots.add_task(Bxz@ex, name='Bx', scales=OutputScale)
snapshots.add_task(By, name='By', scales=OutputScale)
snapshots.add_task(Bxz@ez, name='Bz', scales=OutputScale)

# Restart files
checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=checkpoint_dt, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state)

# CFL condition
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=cfl_number, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep, min_dt=min_timestep) #should include min_dt
CFL.add_velocity(vg)
CFL.add_velocity(valfven)

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
