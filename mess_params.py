'''
-Streaming instabilities in Hall-effected disks (live MHD)
-Shearing box approximation
-Gas-based formulation with dust as a second, pressureless fluid coupled via drag forces
-Dust treated approximately via terminal velocity approximation

Problem parameters and inital perturbations.

Units: Omega = Hg = 1. Velocities scaled by cs=Hg*Omega.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import dedalus.public as d3
import logging
logger0 = logging.getLogger(__name__)
import scipy
from scipy.linalg import eigvals

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
Ha         = 1e4       #Hall parameter
Re         = 1e8        #Reynolds number for setting viscosity/diffusion
ReM        = 1e8        #Magnetic Reynolds number

'''
Dust parameters
'''
eps        = 0.2        #Initial dust/gas ratio
st         = 0.1        #Particle size or Stokes number

'''
Secondary parameters
'''
va2    = 2*cs2/beta                   #Alfven velocity squared
Bz0    = np.sqrt(va2*mu0*rhog0)       #Equilbrium vertical field strength
etaHall= va2/(2*Omega*Ha)             #Hall diffusion coefficient
vscale = etahat*Hg*Omega              #Radial drift velocity scale, eta*r*Omega

fd     = eps/(1+eps)    #Initial dust fraction
fg     = 1 - fd         #Initial gas fraction
tstop  = st/Omega       #Stopping time in physical units

nu     = Omega*Hg**2/Re    #Viscosity and diffusion coefficient
nuM    = Omega*Hg**2/ReM   #Resistivity

'''
Exact formulation or (improved) TVA
'''
approx = 'exact'

'''
Do we include feedback?
'''
feedback = 'yes'

'''
Equilibrium/initial conditions (for mixed formulation)
in mixed form, the gas velocities are measure relative to the dust-free, unpert flow
'''
if feedback == 'yes':
    if approx == 'tva':
        vgx0 = 2*fd*fg*st*vscale
        # vgy0 =-fg*vscale #relative to Keplerian shear
        vgy0 = fd*vscale #relative to pure gas sub-Keplerian shear

        vdx0  = vgx0 - 2*vscale*st*fg
        vdy0  = vgy0 + vscale*fg*fg*st*st
    if approx == 'exact':
        Dsq  = st*st + (1+eps)**2
        vgx0 = 2*eps*st*vscale/Dsq
        # vgy0 =-(1 + eps + st**2)*vscale/Dsq #relative to Keplerian shear
        vgy0 = eps*(1+eps)*vscale/Dsq #relative to pure gas sub-Keplerian shear

        ux0  = -2*st*(1+eps)*vscale/Dsq
        uy0  = st*st*vscale/Dsq

        vdx0  = vgx0 + ux0
        vdy0  = vgy0 + uy0

elif feedback == 'no':#then eps-->0 so fd-->0 and fg-->1
    vgx0 = 0.0
    # vgy0 = -vscale #relative to Kep. flow
    vgy0 = 0.0 #relative to pure gas sub-Keplerian shear
    if approx == 'tva':
        vdx0  = vgx0 - 2*vscale*st
        vdy0  = vgy0 + vscale*st*st
    if approx == 'exact':
        Dsq  = 1 + st*st
        ux0  = -2*st*vscale/Dsq
        uy0  = st*st*vscale/Dsq

        vdx0  = vgx0 + ux0
        vdy0  = vgy0 + uy0

'''
Simplify the Hall term? See main code.
'''
SimplifyHallTerm = True

'''
Special case of toy model of Hall-SI (only works with hallSI_toy.py)
'''
HallSIToy = True

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
if approx == 'exact':
    max_timestep  = 1e-1/Omega
else:
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
Print problem parameters
'''
formatter = logging.Formatter('%(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger0.addHandler(ch)
logger0.propagate = False
logger0.info('*******************')
logger0.info('Problem parameters ')
logger0.info('*******************')
logger0.info("etahat     =%4.2f"%etahat)
logger0.info("beta       =%4.2f"%beta)
logger0.info("Ha         =%4.2f"%Ha)

logger0.info("Reynolds   =%4.2f"%Re)
logger0.info("ReynoldsM  =%4.2f"%ReM)

logger0.info("epsilon    =%4.2f"%eps)
logger0.info("stokes     =%4.2f"%st)

'''
For plotting growth rates
'''
kmin = 1e0/Hg #1e-1
kmax = 1e3/Hg #1e4
nkx  = 128
nkz  = 128

kxarr    = np.logspace(np.log10(kmin), np.log10(kmax), nkx)
kzarr    = np.logspace(np.log10(kmin), np.log10(kmax), nkz)

'''
Plotting options
'''
fontsize= 24
nlev    = 64
nclev   = 6
cmap    = plt.cm.jet

minv    = -5
maxv    = 0

levels  = np.linspace(minv,maxv,nlev)
clevels = np.linspace(minv,maxv,nclev)

'''
Linearized equations in matrix form
'''

class OneFluidMatrices:
    def __init__(self, kx, kz):
        self.kx = kx
        self.kz = kz

    def TerminalApproxMixedImproved(self): #[dP/rhog0, deps, dvgx, dvgy, dvgz, dBx, dBy, dBz]
    #need to check these equations again (derive from scratch)
        kx      = self.kx
        kz      = self.kz
        ikx     = 1j*kx
        ikz     = 1j*kz
        ksq     = self.kx**2 + self.kz**2

        mata    = np.zeros((8,8),dtype=np.cdouble)
        matb    = np.diag([0,1,1,1,1,1,1,1])

        dissipation  = nu*ksq
        dissipationM = nuM*ksq

        mata[0] = np.array([0, 0, ikx, 0, ikz, 0, 0, 0])
        mata[1] = np.array([tstop*ksq, -ikx*vgx0-dissipation, 0, 2*st*ikx, 0, 0, 0, 0])
        mata[2] = np.array([-ikx*fg,-2*vscale*Omega*fg**2,-dissipation,2*Omega, 0, fg*ikz*Bz0/mu0/rhog0, -st*fd*fg*2*ikz*Bz0/mu0/rhog0, 0])
        mata[3] = np.array([-st*fd*fg*0.5*ikx, st*vscale*Omega*fg**3*(1-eps),-0.5*Omega,-dissipation, 0, 0.5*st*fd*fg*ikz*Bz0/mu0/rhog0, fg*ikz*Bz0/mu0/rhog0, 0])
        mata[4] = np.array([-ikz*fg,0,0,0,-dissipation, 0, 0, fg*ikz*Bz0/mu0/rhog0])
        mata[5] = np.array([0,0,ikz*Bz0,0,0, -ikx*vgx0 - dissipationM, -etaHall*kz*kz, 0])
        mata[6] = np.array([0,0,0,ikz*Bz0,0, etaHall*ksq-(3.0/2.0)*Omega, -ikx*vgx0 - dissipationM, 0])
        mata[7] = np.array([0,0,0,0,ikz*Bz0, 0, etaHall*kx*kz, -ikx*vgx0 - dissipationM])

        return (mata, matb)

    def ExactOneFluid(self): #[dP/rhog0, deps, dvgx, dvgy, dvgz, dBx, dBy, dBz, dux, duy, duz]
        kx      = self.kx
        kz      = self.kz
        ikx     = 1j*kx
        ikz     = 1j*kz
        ksq     = self.kx**2 + self.kz**2

        mata    = np.zeros((11,11),dtype=np.cdouble)
        matb    = np.diag([0,1,1,1,1,1,1,1,1,1,1])

        dissipation  = nu*ksq
        dissipationM = nuM*ksq

        mata[0] = np.array([0, 0, ikx, 0, ikz, 0, 0, 0, 0, 0, 0])
        mata[1] = np.array([tstop*ksq, -ikx*vgx0-dissipation, 0, 2*st*ikx, 0, 0, 0, 0, 0, 0, 0])
        mata[2] = np.array([-ikx, ux0/tstop, -ikx*vgx0-dissipation, 2*Omega, 0, ikz*Bz0/mu0/rhog0, 0, 0, eps/tstop, 0, 0])
        mata[3] = np.array([0, uy0/tstop, -0.5*Omega, -ikx*vgx0-dissipation, 0, 0, ikz*Bz0/mu0/rhog0, 0, 0, eps/tstop, 0])
        mata[4] = np.array([-ikz,0,0,0,-ikx*vgx0-dissipation, 0, 0, ikz*Bz0/mu0/rhog0, 0, 0, eps/tstop])
        mata[5] = np.array([0,0,ikz*Bz0,0,0, -ikx*vgx0 - dissipationM, -etaHall*kz*kz, 0, 0, 0, 0])
        mata[6] = np.array([0,0,0,ikz*Bz0,0, etaHall*ksq-(3.0/2.0)*Omega, -ikx*vgx0 - dissipationM, 0, 0, 0, 0])
        mata[7] = np.array([0,0,0,0,ikz*Bz0, 0, etaHall*kx*kz, -ikx*vgx0 - dissipationM, 0, 0, 0])
        mata[8] = np.array([ikx, -ux0/tstop, -ikx*ux0, 0, 0, -ikz*Bz0/mu0/rhog0, 0, 0, -(ikx*vdx0 + (1+eps)/tstop)- dissipation, 2*Omega, 0])
        mata[9] = np.array([0, -uy0/tstop, 0, -ikx*ux0, 0, 0, -ikz*Bz0/mu0/rhog0, 0, -Omega/2, -(ikx*vdx0 + (1+eps)/tstop)- dissipation, 0])
        mata[10] = np.array([ikz, 0, 0, 0, -ikx*ux0, 0, 0, -ikz*Bz0/mu0/rhog0, 0, 0, -(ikx*vdx0 + (1+eps)/tstop) - dissipation])

        return (mata, matb)

    def HallSIToyModel(self): #[dP/rhog0, dvgx, dvgy, dvgz, dBx, dBy, dBz]
        kx      = self.kx
        kz      = self.kz
        ikx     = 1j*kx
        ikz     = 1j*kz
        ksq     = self.kx**2 + self.kz**2

        mata    = np.zeros((7,7),dtype=np.cdouble)
        matb    = np.diag([0,1,1,1,1,1,1])

        dissipation  = nu*ksq
        dissipationM = nuM*ksq

        mata[0] = np.array([0, ikx, 0, ikz, 0, 0, 0])
        mata[1] = np.array([-ikx*fg,-dissipation,2*Omega, 0, fg*ikz*Bz0/mu0/rhog0, 0, 0])
        mata[2] = np.array([0,-0.5*Omega,-dissipation, 0, 0, fg*ikz*Bz0/mu0/rhog0, 0])
        mata[3] = np.array([-ikz*fg,0,0,-dissipation, 0, 0, fg*ikz*Bz0/mu0/rhog0])
        mata[4] = np.array([0,ikz*Bz0,0,0, -ikx*vgx0 - dissipationM, -etaHall*kz*kz, 0])
        mata[5] = np.array([0,0,ikz*Bz0,0, etaHall*ksq-(3.0/2.0)*Omega, -ikx*vgx0 - dissipationM, 0])
        mata[6] = np.array([0,0,0,ikz*Bz0, 0, etaHall*kx*kz, -ikx*vgx0 - dissipationM])

        return (mata, matb)

    def HallSIToyModelGrowthPoly(self): #toy model growth rate from solving characteristic polynomial, NO DISSIPATION
        kx      = self.kx
        kz      = self.kz
        ikx     = 1j*kx
        ikz     = 1j*kz
        ksq     = self.kx**2 + self.kz**2

        c3      = 2*ikx*vgx0
        c2      = kz*kz/ksq*Omega + 2*fg*kz*kz*va2 + etaHall*kz*kz*(etaHall*ksq - 3*Omega/2) - (kx*vgx0)**2
        c1      = 2*ikx*vgx0*kz*kz/ksq*(Omega**2 + fg*ksq*va2)
        c0      = (fg*kz*va2)**2*ksq - 3*fg*kz**2*va2*Omega**2 + (etaHall*kz*Omega)**2*ksq + 0.5*etaHall*kz**2*Omega*(5*fg*ksq*va2-3*Omega**2)
        c0     -= (kx*vgx0*Omega)**2 #drift effect
        c0     *= kz**2/ksq

        roots   = np.roots([1, c3, c2, c1, c0])

        return roots

def OneFluidEigen(kx, kz):
        mat              = OneFluidMatrices(kx, kz)
        if approx == 'tva':
            a, b             = mat.TerminalApproxMixedImproved()
        if approx == 'exact':
            a, b             = mat.ExactOneFluid()
        if HallSIToy == True:
            a, b             = mat.HallSIToyModel()

        evals, evect     = scipy.linalg.eig(a, b)

        # if HallSIToy == True:
        #     evals = mat.HallSIToyModelGrowthPoly()

        growth           = evals.real
        growth[growth == np.inf] = -np.inf

        gmax         = np.argmax(growth)
        eigenvalue   = evals[gmax]
        eigenvector  = evect[:,gmax]

        if eigenvalue > 0:
            norm         = eigenvector[3] #azimuthal gas velocity
            eigenvector *= pert_amp/norm*Hg*Omega #to get units correct
        if pert_amp == 0.0:
            eigenvector *= 0.0

        return (eigenvalue, eigenvector)

def OneFluidEigenMaxGrowth(kx, kz):
    eval, evect = OneFluidEigen(kx, kz)
    return eval.real
    # return np.abs(eval.imag)

class GrowthRates:
        def twodim(growthfunc):
                kx2d, kz2d    = np.meshgrid(kxarr,kzarr)
                GrowthRates2D = np.vectorize(growthfunc)(kx2d, kz2d)
                return GrowthRates2D

def PlotTwoDim(data2D, title, fname):
        data2D[data2D<0.0] = 1e-9#replace decaying modes with small positive growth for plotting purposes only

        plt.rc('font',size=fontsize,weight='bold')

        fig, ax = plt.subplots(constrained_layout=True)
        cp      = plt.contourf(kxarr, kzarr, np.log10(data2D), levels, cmap=cmap)

        cbar    = plt.colorbar(cp,ticks=clevels,format='%.1f',pad=0) # Add a colorbar to a plot
        cbar.set_label(r'$\log(s_\mathrm{max}/\Omega)$')

        ax.set_box_aspect(1)

        plt.xscale('log')
        plt.yscale('log')
        ax.set_ylim(kmin, kmax)
        ax.set_xlim(kmin, kmax)

        ax.set_xlabel(r'$k_xH_g$')
        ax.set_ylabel(r'$k_zH_g$',labelpad=0)
        ax.set_title(title,weight='bold')

        #which scales do the sims resolve? estimate based on number of fourier modes used
        #max resolved wavenumber is ~Nx/2
        NcellsPerLambda = 10
        max_kx = 2.0*np.pi*Nx*Hg/(NcellsPerLambda*Lx)
        max_kz = 2.0*np.pi*Nz*Hg/(NcellsPerLambda*Lz)

        min_kx = 2.0*np.pi*Hg/Lx
        min_kz = 2.0*np.pi*Hg/Lz

        plt.vlines(max_kx, min_kz, max_kz, linestyle='dotted', color='black')
        plt.hlines(max_kz, min_kx, max_kx, linestyle='dotted', color='black')

        plt.vlines(x = min_kx, ymin=min_kz, ymax=max_kz, linestyle='dotted', color='black')
        plt.hlines(y = min_kz, xmin=min_kx, xmax=max_kx, linestyle='dotted', color='black')

        plt.savefig('hallSI_'+fname,dpi=150)

if __name__ == "__main__":
    '''
    Theoretical growth rate for chosen kx_pert and kz_pert
    '''
    growth = OneFluidEigenMaxGrowth(kx_pert, kz_pert)

    logger0.info('*******************')
    logger0.info('Theoretical growth ')
    logger0.info('*******************')
    logger0.info("kx        =%4.2f"%kx_pert)
    logger0.info("kz        =%4.2f"%kz_pert)
    logger0.info("growth    =%4.2e"%growth)

    '''
    Plot growth rates as a function of kx, kz
    '''
    Growth2D = GrowthRates.twodim(OneFluidEigenMaxGrowth)
    PlotTwoDim(Growth2D, r'Full model', 'growth')
    max_growth = np.amax(Growth2D)
    kopt       = np.argmax(Growth2D)
    max_x, max_z = np.unravel_index(kopt, Growth2D.shape)

    kx2d, kz2d    = np.meshgrid(kxarr,kzarr)

    logger0.info("max growth=%4.2e"%max_growth)
    logger0.info("opt     kx=%4.2e"%kx2d[max_x, max_z])
    logger0.info("opt     kz=%4.2e"%kz2d[max_x, max_z])
