import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#### Parameters ###
#Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi)
Lx, Ly, Lz = (4.0, 2.0, 4)

Re = 8000 # U_b*H/nu
#Retau = 180 # = u_tau*H/nu
dtype = np.float64
timestepper = d3.RK222
max_timestep = 0.1  # 0.125 to 0.1

#parameters of breaking waves in Sullivan (2004)
#Sullivan PP, McWILLIAMS JC, Melville WK. The oceanic boundary layer driven by wave breaking with stochastic variability. Part 1. Direct numerical simulations. Journal of Fluid Mechanics. 2004 May;507:143-74.
c=2.343
wavelength=c*c*2*np.pi/9.8
chi=0.2
k_b=0.18
T=wavelength/c
#stop_sim_time = 600

mu1=2.1
mu2=5.1
mu3=5
mu4=10
mu5=2
mu6=2

t0=-0.0001
x0=0
y0=0
nx, ny, nz = 128, 100, 96 #grid resolution
#nx, ny, nz=32, 28, 30
coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=3/2)
zbasis = d3.Chebyshev(coords['z'], size=nz, bounds=(-Lz, 0), dealias=1)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
A0 = dist.Field(name='A0', bases=(xbasis,ybasis,zbasis))

u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
#dPdx = -Retau**2/Re**2
dPdx = -2/Re
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
#x, y, z = dist.dealiased_grid(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ez*lift(tau_u1) # Operator representing G
#x_average = lambda A: d3.Average(A,'x')
#xz_average =  lambda A: d3.Average(A,'z')
#xz_average = lambda A: d3.Average(d3.Average(A, 'x'), 'z')

# Problem
problem = d3.IVP([p, u, tau_p, tau_u1, tau_u2], namespace= globals()| locals())
#problem.namespace.update({'t':problem.time})
#problem.namespace.update({problem.time: problem.sim_time_field})

alpha = lambda t: np.maximum(np.minimum((t-t0)/T,1),0)
beta = lambda t: np.maximum(np.minimum((x-x0)/(c*(t-t0)),1),0)
#delta= np.maximum(np.minimum(2*(y-y0)/wavelength, 1),-1)

#This is for 2D breaking waves, such that Y(delta)=1 and we enforce delta=0 everywhere. Use this minimum function will make sure dimension is consistent
delta=np.minimum(y,0)

gamma= lambda t: np.maximum(np.minimum(z/(chi*c*(t-t0)),0),-1)
T_alpha= lambda t: mu1*alpha(t)**2*(np.exp(mu3*(1-alpha(t))**2)-1)
X_beta= lambda t: mu2*beta(t)**2*(1-beta(t))**2*(1+mu4*beta(t)**3)
Y_delta=(1-delta**2)**2*(1+mu5*delta**2)
Z_gamma= lambda t: (1-gamma(t)**2)**2*(1+mu6*gamma(t)**2)

problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2) = -dot(u,grad(u))+A0*ex")
problem.add_equation("u(z=0) = 0") # change from -1 to -0.5
problem.add_equation("u(z=-Lz) = 0") #change from 1 to 0.5
problem.add_equation("integ(p) = 0")

#get the time variable. 
#t=solver.sim_time


# Build Solver
dt = 0.0001 # 0.001
stop_sim_time = 10*T
fh_mode = 'overwrite'
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler('snapshots_breaking_waves', sim_dt=10, max_writes=600)
snapshots.add_task(u, name='velocity')
snapshots.add_task(d3.curl(u), name='vorticity')
snapshots.add_task(A0, name='A0')


#snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=1, max_writes=400)
#snapshots_stress.add_task(xz_average(u),name = 'ubar')
#snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
#snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
#snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')

#snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')


# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=5, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u) # changed threshold from 0.05 to 0.01

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=20) # changed cadence from 10 to 50
flow.add_property(np.sqrt(u@u)/2, name='TKE')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        t=solver.sim_time
        #A0.require_grid_space()   # Switch back to grid space
        A0.preset_scales(1)
        #print('alpha=',alpha(t))
        #print('beta=',beta(t))
        #print('gamma=',gamma(t))
        
        #print('T_alpha=',T_alpha(t))
        #print('X_beta',X_beta(t))
        #print('Z_gamma=',Z_gamma(t))
        A0['g'] = k_b*c*T_alpha(t)*X_beta(t)*Y_delta*Z_gamma(t)/T

        if (solver.iteration-1) % 10 == 0:
            max_TKE = flow.max('TKE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(TKE)=%f' %(solver.iteration, solver.sim_time, timestep, max_TKE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
#finally:
    #solver.log_stats()
