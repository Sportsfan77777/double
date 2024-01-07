import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py

from scipy.ndimage import uniform_filter1d

'''
utilities
'''
def GetLinearGrowth(time, data, ginterval):
    g1, g2 = ginterval
    fit    = np.polyfit(time[g1:g2+1], np.log(data[g1:g2+1]), 1)
    return fit[0]

'''
plotting parameters
'''
fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

def PlotMaxEvol1D(loc, disk_info, tinterval=(0,10), growth_theory=None, var='vg', yrange=None, logscale=True, avg=1):
    #disk parameters
    eps, va2 = disk_info

    #calculate some normalizations
    va = np.sqrt(va2)

    '''
    Plot 1D evolution and compare with theory (single case only)
    '''

    #read in analysis data
    analysis    = np.loadtxt(loc+'/analysis.txt', delimiter=',')
    time        = analysis[:,0]
    max_dvg     = analysis[:,1]/va
    max_eps     = analysis[:,2]
    rms_vg      = analysis[:,3]/va
    rms_vgx     = analysis[:,4]/va
    rms_vgy     = analysis[:,5]/va
    rms_vgz     = analysis[:,6]/va
    amflux      = analysis[:,7]
    amfluxd     = analysis[:,8]
    min_eps     = analysis[:,9]

    time_orbits = np.copy(time)/(2.0*np.pi) #convert to orbits
    if var == 'vg':
        data = np.copy(max_dvg)
        ylab = r'$max|\delta v|/V_A$'
    elif var == 'epsilon':
        data = np.copy(max_eps)
        ylab = r'$max(\epsilon)$'
    elif var == 'rms':
        data = np.copy(rms_vg)
        ylab = r'$rms(\delta v)/V_A$'
    elif var == 'rmsx':
        data = np.copy(rms_vgx)
        ylab = r'$rms(\delta v_{x})/V_A$'
    elif var == 'rmsy':
        data = np.copy(rms_vgy)
        ylab = r'$rms(\delta v_{y})/V_A$'
    elif var == 'rmsz':
        data = np.copy(rms_vgz)
        ylab = r'$rms(\delta v_{z})/V_A$'
    elif var == 'amflux':
        data = np.copy(amflux)
        ylab = r'$\overline{\delta v_{gx} \delta v_{gy}}$'
    elif var == 'amfluxd':
        data = np.copy(amfluxd)
        ylab = r'$\overline{\delta v_{dx} \delta v_{dy}}$'
    elif var == 'min_eps':
        data = np.copy(min_eps)
        ylab = r'$min(\epsilon)$'

    if avg > 1: #perform a running time average of avg grid points wide
        data = uniform_filter1d(data, size=avg)

    if var == 'vg':
    #fit linear growth
        t1, t2 = tinterval
        g1     = np.argmin(np.abs(time_orbits-t1))
        g2     = np.argmin(np.abs(time_orbits-t2))
        ginterval = (g1, g2)
        growth_sim  = GetLinearGrowth(time, data, ginterval)

        print("{0:^32}".format("************"))
        print("{0:^32}".format("Growth rates"))
        print("{0:^32}".format("************"))
        # print("{0:^20} {1:^20} {2:^20}".format("theory", "sim. (eps)", "sim. (vg)"))
        # print("{0:17.15e} {1:17.15e} {2:17.15e}".format(growth.real, growth_eps, growth_vg))
        print("sim. (vg)  = {0:17.15e}".format(growth_sim))
        if growth_theory != None:
            print("theory     = {0:17.15e}".format(growth_theory))

    fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
    ax  = fig.add_subplot()
    #plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

    plt.xlim(0.0, np.amax(time_orbits))
    # plt.ylim(1e-3,10)
    if yrange != None:
        plt.ylim(yrange[0],yrange[1])

    if logscale == True:
        plt.yscale('log')

#     for i, var in enumerate(data):
#         plt.plot(time_orbits, var, linewidth=2, label=varnames[i])
#         theory_curve = var[g1]*np.exp((time-time[g1])*growth_theory) #In principle could use theory eigenvec. to set the intercept
#         plt.plot(time_orbits[0:2*g2], theory_curve[0:2*g2], color='black', linestyle="dashed", label=varnames[i]+", theory")

    plt.plot(time_orbits, data, linewidth=2, label="simulation")
    if growth_theory != None and var == 'vg':
        theory_curve = data[g1]*np.exp((time-time[g1])*growth_theory) #In principle could use theory eigenvec. to set the intercept
        #theory_curve = data[g1]*np.exp((time-time[g1])*growth_theory*0.01) #If we rescale t by t*st
        # plt.plot(time_orbits[0:2*g2], theory_curve[0:2*g2], color='black', linestyle="dashed", label="theory")
        plt.plot(time_orbits[0:2*g2], theory_curve[0:2*g2], color='black', linestyle="", marker="X", markersize=8, label="theory", markevery=int(len(time_orbits)/20))

    if var ==  'min_eps':
        plt.axhline(y=0, color='r', linestyle='dotted')

    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='lower right', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel(r'$t/P$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(ylab, fontsize=fontsize)

    fname = loc+'/hallSI_maxevol_'+var
    plt.savefig(fname,dpi=150)

def PlotMaxEvol1DCompare(locs, valfven2, labels, title, fname, var='vg', xrange=None, yrange=None,  \
                            logscale=True, avg=1, growth_theory=None, gtheory_labels=None, dgnorm=False):

    #reference alfven speed for each case, stored in an array for later use
    va = [np.sqrt(va2) for va2 in valfven2]

    '''
    Compare 1D evolution across cases
    '''

    fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
    ax  = fig.add_subplot()

    if not yrange:
        plt.ylim(1e-4,1e-1)
    else:
        plt.ylim(yrange[0],yrange[1])

    if logscale  == True:
        plt.yscale('log')

    tend = 0
    for i, loc in enumerate(locs):
        analysis    = np.loadtxt(loc+'/analysis.txt', delimiter=',')
        time        = analysis[:,0]
        max_dvg     = analysis[:,1]/va[i]
        max_eps     = analysis[:,2]
        if dgnorm == True:
            max_eps/= max_eps[0] #normalize by initial value
        rms_vg      = analysis[:,3]
        rms_vgx     = analysis[:,4]
        rms_vgy     = analysis[:,5]
        rms_vgz     = analysis[:,6]
        amflux      = analysis[:,7]
        amfluxd     = analysis[:,8]
        min_eps     = analysis[:,9]

        time_orbits = np.copy(time)/(2.0*np.pi) #convert to orbits
        if var == 'vg':
            data = max_dvg
        elif var == 'epsilon':
            data = max_eps
        elif var == 'rms':
            data = rms_vg
        elif var == 'rmsx':
            data = rms_vgx
        elif var == 'rmsy':
            data = rms_vgy
        elif var == 'rmsz':
            data = rms_vgz
        elif var == 'amflux':
            data = amflux
        elif var == 'amfluxd':
            data = amfluxd
        elif var == 'min_eps':
            data = min_eps

        if avg > 1: #perform a running time average of avg grid points wide
            data = uniform_filter1d(data, size=avg)

        # if pert == True:
        #     data-=data[0]
        # if i == 1:
            # next(ax._get_lines.prop_cycler)#skip the next color in the cycle
        plt.plot(time_orbits, data, linewidth=2, label=labels[i])
        # else:
        #     plt.plot(time_orbits, data, linewidth=2, label=labels[i])#, linestyle="dashed")

        if growth_theory != None and var == 'vg': #and i == len(locs)-1:
            theory_curve = data[0]*np.exp((time-time[0])*growth_theory[i])
            if gtheory_labels !=None:
                lab = gtheory_labels[i]
            else:
                lab = 'theory'
            # plt.plot(time_orbits, theory_curve, linestyle="dashed",markersize=8, label=lab, markevery=int(len(time_orbits)/10))
            plt.plot(time_orbits, theory_curve, linestyle="", marker ="X", markersize=8, label=lab, markevery=int(len(time_orbits)/10))
        maxorbits = np.amax(time_orbits)
        tend      = np.amax([maxorbits, tend])

    if not xrange:
        plt.xlim(0.0, tend)
    else:
        plt.xlim(xrange[0],xrange[1])

    if var == 'vg':
        ylabel = r'$max|\delta v|/V_A$'
    elif var == 'epsilon':
        if dgnorm == False:
            ylabel = r'$max(\epsilon)$'
        else:
            ylabel = r'$max(\epsilon)/\epsilon_0$'
    elif var == 'rms':
        ylabel = r'$rms(\delta v)$'
    elif var == 'rmsx':
        ylabel = r'$rms(\delta v_{x})$'
    elif var == 'rmsy':
        ylabel = r'$rms(\delta v_{y})$'
    elif var == 'rmsz':
        ylabel = r'$rms(\delta v_{z})$'
    elif var == 'amflux':
        ylabel = r'$\overline{\delta v_{dx} \delta v_{gy}}$'
    elif var == 'amfluxd':
        ylabel = r'$\overline{\delta v_{dx} \delta v_{dy}}$'
    elif var == 'min_eps':
        ylabel = r'$min(\epsilon)$'

    if var ==  'min_eps':
        plt.axhline(y=0, color='r', linestyle='dotted')

    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='lower right', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel(r'$t/P$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(ylabel, fontsize=fontsize)

    ax.set_title(title, weight='bold')

    fname = 'hallSI_plot_compare_'+var+'_'+fname
    plt.savefig(fname,dpi=150)

def GetFileName(time_in_orbits, output_info):
    stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info
    #Figure out which set is the desired snapshot in
    nsnap     = int(stop_sim_time/snapshot_dt) #including initial conditions
    gtime     = np.linspace(0, stop_sim_time-snapshot_dt, nsnap)/period #global time axis
    g1        = np.argmin(np.abs(gtime - time_in_orbits)) #index in global time array
    nset      = int(g1/snapshots_per_set) + 1
    # print("time, nset", time_in_orbits, nset)

    fname = "/snapshots/snapshots_s"+str(nset)+".h5"
    return g1, nset, fname

def Plot2DContour(loc, var, time_in_orbits, output_info, disk_info, \
                    clabel, axiscale=None, axilabel=None, arange=None, plotrange=None, \
                    log=None, pert=False, axi=True, norm=None, title=None, aspect=1):

    #physical disk parameters
    eps, va2 = disk_info

    #Read file, need to figure out which set the data is in.
    stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info
    g1, nset, fname   = GetFileName(time_in_orbits, output_info)
    output      = h5py.File(loc+fname, mode='r')

    #Get timestamp
    tarr    = output['scales']['sim_time'][:]/period #convert time axis to orbits
    l1      = g1 - (nset-1)*snapshots_per_set #local index
    tstamp  = tarr[l1]

    #Get data at the timestamp and read axis information
    data    = output['tasks'][var]

    xaxis   = data.dims[1][0][:]
    if axi == False:
        zaxis   = data.dims[3][0][:]
    else:
        zaxis   = data.dims[2][0][:]

    if axiscale != None:
        xaxis /= axiscale[0]
        zaxis /= axiscale[1]

    if arange == None:
        xmin    =-np.amax(xaxis)
        xmax    = np.amax(xaxis)

        zmin    =-np.amax(zaxis)
        zmax    = np.amax(zaxis)
    else:
        xmin, xmax    = arange[0], arange[1]
        zmin, zmax    = arange[0], arange[1]

    data3d  = data[l1]

    if var == 'epsilon': #hydro output is actually Q, need to convert to epsilon
        data3d = eps*np.exp(data3d/eps)

    if norm != None:
        data3d /= norm

    if pert == True:
        data3d -= np.mean(data[0]) #take out initial value (averaged)
    if axi == False:
        data2d  = data3d[:,0,:]
    else:
        data2d  = data3d[:,:]

    if log != None:
        data2d = np.log10(data2d)

    if plotrange == None:
        minv    = np.amin(data2d)
        maxv    = np.amax(data2d)
    else:
        minv    = plotrange[0]
        maxv    = plotrange[1]

    levels  = np.linspace(minv,maxv,nlev)
    clevels = np.linspace(minv,maxv,nclev)

    plt.rc('font',size=fontsize,weight='bold')

    if aspect == 1:
        figsize = (6.5,5)
    if aspect == 4:
        figsize = (4,7)

    fig, ax = plt.subplots(figsize=figsize,constrained_layout=True)
    cp      = plt.contourf(xaxis, zaxis, np.transpose(data2d), levels, cmap=cmap)

    cbar    = plt.colorbar(cp,ticks=clevels,format='%.2f',pad=0)
    # cbar    = plt.colorbar(cp,ticks=clevels,format='%.1f',pad=0)
    if aspect == 4:
        cbar.ax.tick_params(labelsize=16)
    cbar.set_label(clabel)

    ax.set_box_aspect(aspect)

    ax.set_ylim(zmin, zmax)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(1))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    if aspect == 4:
        ax.tick_params(labelsize=16)

    if axilabel != None:
        xlabel = axilabel[0]
        ylabel = axilabel[1]
    else:
        xlabel = r'$x/H_g$'
        ylabel = r'$z/H_g$'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel,labelpad=-1)

    if title !=None:
        ax.set_title(title, weight='bold')
    else:
        ax.set_title(r't={0:3.0f}'.format(tstamp)+"P",weight='bold')

    plt.savefig(loc+'/hallSI_'+var+'2D_'+str(g1).zfill(3),dpi=150)
    plt.close()

def PlotRmsEvol1D(loc, yrange=None, logscale=True, avg=1):
    '''
    Compare rms velocities
    '''

    #read in analysis data
    analysis    = np.loadtxt(loc+'/analysis.txt', delimiter=',')
    time        = analysis[:,0]
    rms_vg      = analysis[:,3]
    rms_vgx     = analysis[:,4]
    rms_vgy     = analysis[:,5]
    rms_vgz     = analysis[:,6]

    time_orbits = np.copy(time)/(2.0*np.pi) #convert to orbits

    if avg > 1: #perform a running time average of avg grid points wide
        rms_vg = uniform_filter1d(rms_vg, size=avg)
        rms_vgx= uniform_filter1d(rms_vgx, size=avg)
        rms_vgy= uniform_filter1d(rms_vgy, size=avg)
        rms_vgz= uniform_filter1d(rms_vgz, size=avg)

    fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
    ax  = fig.add_subplot()

    if yrange != None:
        plt.ylim(yrange[0],yrange[1])

    if logscale == True:
        plt.yscale('log')

    plt.plot(time_orbits, rms_vg, linewidth=2, label=r'$v_g$')
    plt.plot(time_orbits, rms_vgx, linewidth=2, label=r'$v_{gx}$')
    plt.plot(time_orbits, rms_vgy, linewidth=2, label=r'$v_{gy}$')
    plt.plot(time_orbits, rms_vgz, linewidth=2, label=r'$v_{gz}$')

    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='lower right', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel(r'$t/P$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(r'$rms(\delta v_g)$', fontsize=fontsize)

    fname = loc+'/hallSI_rmsevol'
    plt.savefig(fname,dpi=150)

def Plot1DGeostrophicBalance(loc, time_in_orbits, output_info, arange=None, plotrange=None):

    #Read file, need to figure out which set the data is in.
    stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info
    g1, nset, fname   = GetFileName(time_in_orbits, output_info)
    output      = h5py.File(loc+fname, mode='r')

    #Get timestamp
    tarr    = output['scales']['sim_time'][:]/period #convert time axis to orbits
    l1      = g1 - (nset-1)*snapshots_per_set #local index
    tstamp  = tarr[l1]

    #Get data at the timestamp and read axis information
    vgy_snaps     = output['tasks']['vgy']
    W_snaps       = output['tasks']['W']

    xaxis   = vgy_snaps.dims[1][0][:]

    if arange == None:
        xmin    =-np.amax(xaxis)
        xmax    = np.amax(xaxis)
    else:
        xmin, xmax    = arange[0], arange[1]

    #full 3D snapshots
    vgy3D = vgy_snaps[l1]
    W3D   = W_snaps[l1]

    #extract vertically averaged, 1D radial profiles
    vgy1D = np.mean(vgy3D[:,0,:],axis=1)
    W1D   = np.mean(W3D[:,0,:],axis=1)

    #get dW/dx
    dW1Ddx = np.gradient(W1D, xaxis)

    #compare 2*Omegavgy and pressure gradient

    fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
    ax  = fig.add_subplot()

    plt.rc('font',size=fontsize,weight='bold')

    plt.plot(xaxis, 2*vgy1D, linewidth=2, label=r'$2\Omega v_{gy}$')#assuming Omega=1
    # plt.plot(xaxis, dW1Ddx, linestyle="None", marker="X", color="black" ,markersize=8, label=r'$dW/dx$', markevery=int(len(xaxis)/100))
    plt.plot(xaxis, dW1Ddx, linestyle="dashed", linewidth=2, label=r'$dW/dx$')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel(r'$x/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    #plt.ylabel(ylabel, fontsize=fontsize)

    title = r't={0:3.0f}'.format(tstamp)+"P"
    ax.set_title(title, weight='bold')

    fname = 'hallSI_plot_geostrophic'
    plt.savefig(fname,dpi=150)
    plt.close()

def Plot1DEpsilonPressure(loc, time_in_orbits, output_info, disk_info, arange=None, plotrange=None):

    eps, va2 = disk_info

    #Read file, need to figure out which set the data is in.
    stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info
    g1, nset, fname   = GetFileName(time_in_orbits, output_info)
    output      = h5py.File(loc+fname, mode='r')

    #Get timestamp
    tarr    = output['scales']['sim_time'][:]/period #convert time axis to orbits
    l1      = g1 - (nset-1)*snapshots_per_set #local index
    tstamp  = tarr[l1]

    #Get data at the timestamp and read axis information
    dg_snaps     = output['tasks']['epsilon'] #this is actually Q
    dg_snaps     = eps*np.exp(dg_snaps/eps)
    W_snaps      = output['tasks']['W']

    xaxis   = dg_snaps.dims[1][0][:]

    if arange == None:
        xmin    =-np.amax(xaxis)
        xmax    = np.amax(xaxis)
    else:
        xmin, xmax    = arange[0], arange[1]

    #full 3D snapshots
    dg3D = dg_snaps[l1]
    W3D  = W_snaps[l1]

    #extract vertically averaged, 1D radial profiles
    dg1D = np.mean(dg3D[:,0,:],axis=1)
    W1D   = np.mean(W3D[:,0,:],axis=1)

    fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
    ax  = fig.add_subplot()

    plt.rc('font',size=fontsize,weight='bold')

    plt.plot(xaxis, dg1D, linewidth=2, label=r'$\epsilon$')#assuming Omega=1
    plt.plot(xaxis, W1D, linestyle="dashed", linewidth=2, label=r'$W$')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel(r'$x/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    #plt.ylabel(ylabel, fontsize=fontsize)

    title = r't={0:3.0f}'.format(tstamp)+"P"
    ax.set_title(title, weight='bold')

    fname = 'hallSI_plot_epspress'
    plt.savefig(fname,dpi=150)
    plt.close()

def PlotSpaceTime(loc, var, beg, end, output_info, disk_info, \
                    clabel, plotrange=None, log=None, title=''):
    #Some meta data
    eps, va2 = disk_info
    stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info

    #read the first output to get domain information
    output  = h5py.File(loc+'/snapshots/snapshots_s1.h5', mode='r')
    data    = output['tasks'][var]
    xaxis   = data.dims[1][0][:]
    zaxis   = data.dims[3][0][:]

    #empty array to store spacetime plot
    nx      = xaxis.size
    ntime   = snapshots_per_set*(end - beg + 1)
    taxis   = np.zeros(ntime)
    data2D  = np.zeros((ntime, nx))

    #loop through each snapshot in each set of snapshots
    i = 0 #time count
    for n in range(beg, end+1):
        output  = h5py.File(loc+'/snapshots/snapshots_s'+str(n)+'.h5', mode='r')
        tarr    = output['scales']['sim_time'][:]/period
        data    = output['tasks'][var]
        for m in range(0, snapshots_per_set):
            data3D      = data[m]
            taxis[i]    = tarr[m]
            dataXZ      = data3D[:,0,:]
            data2D[i,:] = np.mean(dataXZ,axis=1)
            # print('time=', i, taxis[i])
            i += 1


    tmin    = np.amin(taxis)
    tmax    = np.amax(taxis)

    xmin    =-np.amax(xaxis)
    xmax    = np.amax(xaxis)

    if var == 'epsilon': #output is actually Q, so convert back to epsilon
        data2D = eps*np.exp(data2D/eps)

    if log != None:
        data2D = np.log10(data2D)

    if plotrange == None:
        minv    = np.amin(data2D)
        maxv    = np.amax(data2D)
    else:
        minv    = plotrange[0]
        maxv    = plotrange[1]

    levels  = np.linspace(minv,maxv,nlev)
    clevels = np.linspace(minv,maxv,nclev)

    plt.rc('font',size=fontsize,weight='bold')

    fig, ax = plt.subplots(figsize=(9,4),constrained_layout=False)
    cp      = plt.contourf(taxis, xaxis, np.transpose(data2D), levels, cmap=cmap)

    divider = make_axes_locatable(plt.gca())
    colorbar_axes = divider.append_axes("right", "3%", pad="-5%")
    cbar    = plt.colorbar(cp,ticks=clevels,format='%.3f',cax=colorbar_axes)
    cbar.set_label(clabel)

    ax.set_box_aspect(0.5)
    ax.set_ylim(xmin, xmax)
    ax.set_xlim(tmin, tmax)

    fig.tight_layout()
    plt.subplots_adjust(left=0.13,bottom=0.22)

    ax.set_xlabel(r'$t/P$')
    ax.set_ylabel(r'$x/H_g$',labelpad=-1)

    ax.set_title(title, weight='bold')

    plt.savefig(loc+'/hallSI_'+var+'_SpaceTime.png',dpi=150)
    plt.close()

# def Plot1DDustTrapping(loc, time_in_orbits, output_info, disk_info, arange=None, plotrange=None):
#
#     eps, va2, etahat, Nr2, tstop = disk_info
#
#     #Read file, need to figure out which set the data is in.
#     stop_sim_time, snapshot_dt, snapshots_per_set, period = output_info
#     g1, nset, fname   = GetFileName(time_in_orbits, output_info)
#     output      = h5py.File(loc+fname, mode='r')
#
#     #Get timestamp
#     tarr    = output['scales']['sim_time'][:]/period #convert time axis to orbits
#     l1      = g1 - (nset-1)*snapshots_per_set #local index
#     tstamp  = tarr[l1]
#
#     #Get data at the timestamp and read axis information
#     dg_snaps     = output['tasks']['epsilon']
#     W_snaps      = output['tasks']['W']
#     vgx_snaps    = output['tasks']['vgx']
#     theta_snaps  = output['tasks']['theta']
#
#     xaxis   = dg_snaps.dims[1][0][:]
#
#     if arange == None:
#         xmin    =-np.amax(xaxis)
#         xmax    = np.amax(xaxis)
#     else:
#         xmin, xmax    = arange[0], arange[1]
#
#     #full 3D snapshots
#     dg   = dg_snaps[l1][:,0,:] #this is actually Q
#     dg   = eps*np.exp(dg/eps)
#
#     W    = W_snaps[l1][:,0,:]
#     vgx  = vgx_snaps[l1][:,0,:]
#     theta= theta_snaps[l1][:,0,:]
#
#     #extract vertically averaged, 1D radial profiles
#     dgbar  = np.mean(dg,axis=1)
#     Wbar   = np.mean(W,axis=1)
#     vgxbar = np.mean(vgx,axis=1)
#     thetabar = np.mean(theta,axis=1)
#
#     #calculate the perturbed fields
#     Wp  = W   - np.expand_dims(Wbar,axis=1)
#     dgp = dg  - np.expand_dims(dgbar,axis=1)
#     vgxp= vgx - np.expand_dims(vgxbar,axis=1)
#     thetap= theta - np.expand_dims(thetabar,axis=1)
#
#     dWbardx = np.gradient(Wbar,xaxis)
#     dWpdx   = np.gradient(Wp,xaxis,axis=0)
#     dgpdWpdx_bar = np.mean(dgp*dWpdx,axis=1)
#
#     lhs  = dgbar*vgxbar
#     lhs2 = np.mean(dgp*vgxp,axis=1)
#
#     rhs = -tstop*dgbar*dWbardx
#     rhs2= -tstop*dgpdWpdx_bar
#     rhs3=  tstop*2*etahat*dgbar #assumes H=Omega=1
#     rhs4= -tstop*Nr2*dgbar*thetabar
#     rhs5= -tstop*Nr2*np.mean(dgp*thetap,axis=1)
#
#     fig = plt.figure(figsize=(8,4.5),constrained_layout=True)
#     ax  = fig.add_subplot()
#
#     plt.rc('font',size=fontsize,weight='bold')
#
#     #plt.plot(xaxis, lhs, linewidth=2, label=r'lhs')#assuming Omega=1
#     plt.plot(xaxis, lhs2, linewidth=2, label=r'lhs2')
#     plt.plot(xaxis, rhs, linestyle="dashed", linewidth=2, label=r'rhs')
#     # plt.plot(xaxis, rhs2, linestyle="dashed", linewidth=2, label=r'rhs2')
#     # plt.plot(xaxis, rhs3, linestyle="dashed", linewidth=2, label=r'rhs3')
#     # plt.plot(xaxis, rhs4, linestyle="dashed", linewidth=2, label=r'rhs4')
#     # plt.plot(xaxis, rhs5, linestyle="dashed", linewidth=2, label=r'rhs5')
#
#     lines1, labels1 = ax.get_legend_handles_labels()
#     legend=ax.legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, fontsize=fontsize/1.5, handletextpad=0.1, labelspacing=0.1)
#
#     plt.xticks(fontsize=fontsize,weight='bold')
#     plt.xlabel(r'$x/H_g$',fontsize=fontsize)
#
#     plt.yticks(fontsize=fontsize,weight='bold')
#     #plt.ylabel(ylabel, fontsize=fontsize)
#
#     title = r't={0:3.0f}'.format(tstamp)+"P"
#     ax.set_title(title, weight='bold')
#
#     fname = 'hallSI_plot_dust_trapping'
#     plt.savefig(loc+'/'+fname,dpi=150)
#     plt.close()

if __name__ == "__main__":
    import hallSI_params

    disk_info=[hallSI_params.eps, hallSI_params.va2]

    if hallSI_params.pert == 'random':
        Growth2D = hallSI_params.GrowthRates.twodim(hallSI_params.OneFluidEigenMaxGrowth)
        growth_theory = np.amax(Growth2D)
    else:
        growth_theory   = hallSI_params.OneFluidEigenMaxGrowth(hallSI_params.kx_pert, hallSI_params.kz_pert)

    #compare sim and theory growth rates
    loc             = '.'
    # tinterval       = (200, 600)
    tinterval       = (0, 10)
    PlotMaxEvol1D(loc, disk_info, tinterval,growth_theory,var='vg', yrange=[1e-2,20])
    PlotMaxEvol1D(loc, disk_info, tinterval,var='epsilon',yrange=[1e-1,2e0])
    PlotMaxEvol1D(loc, disk_info, tinterval,var='min_eps',yrange=[0,disk_info[0]],logscale=False)

    time_in_orbits  = 100
    var             ='epsilon'
    clabel          =r'$\epsilon$'
    # var             ='vgz'
    # clabel          =r'$v_{z}$'
    output_info     =(hallSI_params.stop_sim_time, hallSI_params.snapshot_dt, \
                        1, hallSI_params.period)
    #axiscale        =[hallSI_params.Lx, hallSI_params.Lz]
    Plot2DContour(loc, var, time_in_orbits, output_info, disk_info, clabel, aspect=1)
    # Plot1DGeostrophicBalance(loc, time_in_orbits, output_info, arange=None, plotrange=None)
    # Plot1DEpsilonPressure(loc, time_in_orbits, output_info, arange=None, plotrange=None)

    # for n in range(10,400,10):
    #     Plot2DContour(loc, var, n, output_info, clabel, arange=arange)
    # beg = 1
    # end = 101
    # clabel =r'$\langle\epsilon\rangle$'
    # PlotSpaceTime(loc, var, beg, end, output_info, clabel, plotrange=None, log=None, title='')

    # disk_info = (hallSI_params.tstop, hallSI_params.eps, hallSI_params.etahat, hallSI_params.Nr2)
    # Plot1DDustTrapping(loc, time_in_orbits, output_info, disk_info, arange=None, plotrange=None)
