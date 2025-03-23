"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 inclined_porous_convection.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs analysis --cleanup
    $ mpiexec -n 4 python3 plot_slices.py analysis/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 inclined_porous_convection.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs analysis
    $ ln -s analysis/analysis_s1.h5 restart.h5
    $ mpiexec -n 4 python3 inclined_porous_convection.py
=
The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post

import logging
logger = logging.getLogger(__name__)

class flag:
    pass
# Parameters
flag=flag()

# Parameters
flag.La=1

flag.Lx, flag.Ly, flag.Lz = (4*np.pi, np.pi, 1.) #domain size
flag.Nx=128 #grid point number in x
flag.Ny=64
flag.Nz=66 #grid point number in z

#parameter to control simulation and storage time
flag.initial_dt=0.001 #the initial time step
flag.stop_sim_time=100 #The simulation time to stop
flag.post_store_dt=10 #The time step to store the data

#paramter for the initial guess
flag.A_noise=0 #random noise magnitude in the initial condition
flag.restart_t0=1 #if 1, the simulation time will start from zero. Otherwise, will continue the previous one 

# Create bases and domain
x_basis = de.Fourier('x', flag.Nx, interval=(0, flag.Lx), dealias=3/2)
y_basis = de.Fourier('y',flag.Ny, interval=(0,flag.Ly), dealias=3/2)
z_basis = de.Chebyshev('z', flag.Nz, interval=(0, flag.Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
x, y, z = domain.all_grids()

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','u','v','w','uz','vz','wz'])
problem.parameters['La'] = flag.La
 
#For Couette profile of Stokes drift following 
#based on the closed form of asymptotically reduced equations in the equations (29), (31), and (32) of 
#Chini GP, Julien K, Knobloch E. An asymptotically reduced model of turbulent Langmuir circulation. Geophysical and Astrophysical Fluid Dynamics. 2009 Apr 1;103(2-3):179-97.

#We set up the domain size based on the following reference trying to reproduce their results. 
#Zhang Z, Chini GP, Julien K, Knobloch E. Dynamic patterns in the reduced Craik–Leibovich equations. Physics of Fluids. 2015 Apr 1;27(4).
problem.add_equation("dt(u)+dx(p)-La*(dz(uz)+dy(dy(u)))=-v*dy(u)-w*uz")
problem.add_equation("dt(v)+dy(p)-La*(dz(vz)+dy(dy(v)))-z*dy(u)+z*dx(v)=-v*dy(v)-w*vz")
problem.add_equation("dt(w)+dz(p)-La*(dz(wz)+dy(dy(w)))-z*uz+z*dx(w)=-v*dy(w)-w*wz")
problem.add_equation("dy(v) + wz = 0")

problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

problem.add_bc("uz(z='left')=1")
problem.add_bc("uz(z='right')=1")

problem.add_bc("vz(z='left')=0")
problem.add_bc("vz(z='right')=0")

problem.add_bc("w(z='left') = 0")
problem.add_bc("w(z='right') = 0",condition="(ny != 0)")
problem.add_bc("integrate(integrate(p,y),z) = 0", condition="(ny == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

if not pathlib.Path('restart.h5').exists():

    print('Set up initial condition!')
    # Initial conditions
    #x, y, z = domain.all_grids()
    
    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls
    #zb, zt = z_basis.interval
    pert = flag.A_noise * noise * z * (1 - z)
    
    u = solver.state['u']
    u['g'] = np.sin(4.55*x)*(1-z)*z +pert
 
    # Timestepping and output
    dt = flag.initial_dt
    stop_sim_time = flag.stop_sim_time
    fh_mode = 'overwrite'

else:
    # Restart
    print('Restart')
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = flag.stop_sim_time
    fh_mode = 'append'
    if flag.restart_t0:
        solver.sim_time=0
        fh_mode='overwrite'


# Integration parameters
solver.stop_sim_time = flag.stop_sim_time

# Analysis
analysis = solver.evaluator.add_file_handler('analysis', sim_dt=flag.post_store_dt)
analysis.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=flag.initial_dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u','v','w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("integ(sqrt(u*u + v*v+ w*w))/2", name='TKE')
#flow_out = flow_tools.GlobalFlowProperty(solver, cadence=1)
#flow_out.add_property('w*b',name='wb')
           
def print_screen(flag,logger):
    #print the flag onto the screen
    flag_attrs=vars(flag)
    #print(', '.join("%s: %s, \n" % item for item in flag_attrs.items()))
    logger.info(', Attributes: Value,\n,')
    logger.info(', '.join("%s: %s, \n" % item for item in flag_attrs.items()))

def print_file(flag):
    #print the flag onto file
    flag_text=open('./analysis'+'/flag.txt','w+')
    flag_attrs=vars(flag)
    print(', Attributes: 123,\n ------\n-------\n------',file=flag_text)
    print(', test: 123,',file=flag_text)
    print(', '+', '.join("%s: %s, \n" % item for item in flag_attrs.items()),file=flag_text)
    flag_text.close()
    
        
# Main loop
try:
    logger.info('Starting loop')
    print_screen(flag,logger)
    print_file(flag)
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('TKE = %f' %flow.max('TKE'))

    #add check point, only store one snapshot
    checkpoint=solver.evaluator.add_file_handler('checkpoint')
    checkpoint.add_system(solver.state)
    end_world_time = solver.get_world_time()
    end_wall_time = end_world_time - solver.start_time
    solver.evaluator.evaluate_handlers([checkpoint], timestep = flag.initial_dt, sim_time = solver.sim_time, world_time=end_world_time, wall_time=end_wall_time, iteration=solver.iteration)
    post.merge_process_files('checkpoint',cleanup=True)

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()