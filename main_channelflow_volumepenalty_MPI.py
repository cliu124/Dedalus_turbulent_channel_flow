import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

#### Parameters ###
#Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi)
Lx, Ly, Lz = (2.4/(2*np.pi), 2.0, 2.4/(2*np.pi)) # non dimensional length

Re = 690 # U_b*H/nu
#Retau = 180 # = u_tau*H/nu
dtype = np.float64
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.1  # 0.125 to 0.1

# Create bases and domain
nx, ny, nz = 32, 20, 30 #54, 129, 42 , 
#nx, ny, nz = 100, 129, 82 #76, 129, 64
#nx, ny, nz = 8, 8, 8 #76, 129, 64
Ny=129
coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64)
#max_nodes = nx * ny * nz  # Max number of nodes
max_nodes = nz*nx*ny  # Max number of nodes

# Creating basis for x, y, z dimensions
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, Lz), dealias=3/2)
#ybasis = d3.Chebyshev(coords['y'], size=ny, bounds=(-Ly/2, Ly/2))
ybasis = d3.Chebyshev(coords['y'], size=ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
print(f"xbasis size = {xbasis.size}, ybasis size = {ybasis.size}, zbasis size = {zbasis.size}")


# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,zbasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
#inv_k = dist.Field(name='inv_k', bases=(xbasis,ybasis,zbasis)) # stiffness function
inv_k = dist.Field(name='inv_k', bases=(xbasis,ybasis,zbasis)) # stiffness function

#local_x, local_y, local_z = dist.local_grids(xbasis, ybasis, zbasis)

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

# convert x, y and z into 1 dimension vectors 
x = x.flatten()
y = y.flatten()
z = z.flatten()

#local_x = local_x.flatten()
#local_y = local_y.flatten()
#local_z = local_z.flatten()

# Access the left edge of the global domain from the bases
global_left_x = xbasis.bounds[0]
global_left_y = ybasis.bounds[0]
global_left_z = zbasis.bounds[0]

# Compute the global coordinates by adding the local grid values to the global left edge
global_x = x + global_left_x
global_y = y + global_left_y
global_z = z + global_left_z

nx_global = xbasis.shape[0] # Global number of points or indices in the x-direction, not local
ny_global = ybasis.shape[0] # Global number of points or indices in the y-direction, not local
nz_global = zbasis.shape[0] # Global number of points or indices in the z-direction, not local


#y_min = np.min(y)    # Get the min value for this local y slice
#y_max = np.max(y)    # Get the max value for this local y slice

# Constructing the global y range using the bounds and local values
global_y = np.linspace(global_left_y, global_left_y + Ly, ny_global)


print("nx_global = ", nx_global)
print("ny_global = ", ny_global)
print("nz_global = ", nz_global)

print(global_left_x)
print(global_left_y)
print(global_left_z)

# FSI cells initalization
# Initialize lists

x_solid = np.zeros(max_nodes)
y_solid = np.zeros(max_nodes)
z_solid = np.zeros(max_nodes)
solid_nodes = np.zeros((max_nodes, 3), dtype=int)

y_solid_symmetric = np.zeros(max_nodes)
z_solid_symmetric = np.zeros(max_nodes)
x_solid_symmetric = np.zeros(max_nodes)
solid_nodes_symmetric = np.zeros((max_nodes, 3), dtype=int)

y_ghost = np.zeros(max_nodes)
z_ghost = np.zeros(max_nodes)
x_ghost = np.zeros(max_nodes)
ghost_nodes = np.zeros((max_nodes, 3), dtype=int)

y_ghost_symmetric = np.zeros(max_nodes)
z_ghost_symmetric = np.zeros(max_nodes)
x_ghost_symmetric = np.zeros(max_nodes)
ghost_nodes_symmetric = np.zeros((max_nodes, 3), dtype=int)

# Substitutions
#dPdx = -Retau**2/Re**2
dPdx = -2/Re
#x, y, z = dist.local_grids(xbasis, ybasis, zbasis)

#x = xbasis.size
#y = ybasis.size
#z = zbasis.size

print("length of vector x =",len(x))
print("length of vector y =",len(y))
print("length of vector x =",len(z))

print("length of global vector x =",len(global_x))
print("length of global vector y =",len(global_y))
print("length of global vector z =",len(global_z))

#print("x grid values:", x)
#print("y grid values:", y)
#print("z grid values:", z)


ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = ybasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ey*lift(tau_u1) # Operator representing G
x_average = lambda A: d3.Average(A,'x')
#xz_average =  lambda A: d3.Average(A,'z')
xz_average = lambda A: d3.Average(d3.Average(A, 'x'), 'z')
#body_force = inv_k*u

#print('length y', len(y))

# for MPI 
ny = len(y)  
 
# compute y sin in streamwise direction
#y_sin = -0.88 + 0.12 * np.sin(2 * np.pi / Lx * x)

y_sin = -0.88 + 0.12 * np.sin(2 * np.pi / Lx * x)

# Print or use y_sin as needed
#print('y sin = ', y_sin)

'''
solid_nodes = np.loadtxt("solid_nodes.csv", delimiter=",")  # Skip header if present
ghost_nodes = np.loadtxt("ghost_nodes.csv", delimiter=",")  # Skip header if present

x_solid = np.loadtxt("x_solid.csv", delimiter=",")  # Skip header if present
y_solid = np.loadtxt("y_solid.csv", delimiter=",")  # Skip header if present
z_solid = np.loadtxt("z_solid.csv", delimiter=",")  # Skip header if present

x_ghost = np.loadtxt("x_ghost.csv", delimiter=",")  # Skip header if present
y_ghost = np.loadtxt("y_ghost.csv", delimiter=",")  # Skip header if present
z_ghost = np.loadtxt("z_ghost.csv", delimiter=",")  # Skip header if present
'''
# adding lines for wavy wall nodes identification

ctr2 = 0
ctr3 = 0

# Nested loops equivalent to MATLAB

for ii in range(nx):
    
    for i in range(nz):
   
        for j in range(ny_global-1):
            
            if (global_y[j] <= y_sin[ii]).all():
                
                # Solid nodes
                y_solid[ctr2] = global_y[j]
                z_solid[ctr2] = z[i]
                x_solid[ctr2] = x[ii]
                solid_nodes[ctr2, :] = [ii, j, i]
                
                #print("global_xsolid_index[",ctr2,"]=",ii)
                #print("global_ysolid_index[",ctr2,"]=",j)
                #print("global_zsolid_index[",ctr2,"]=",i)
                #print("solid_nodes[",ctr2,"]=",solid_nodes[ctr2,:])
                #print(np.shape(solid_nodes))                
                
                # Symmetric solid nodes
                y_solid_symmetric[ctr2] = global_y[ny_global - (j+1)]
                z_solid_symmetric[ctr2] = z[i]
                x_solid_symmetric[ctr2] = x[ii]
                #solid_nodes_symmetric[ctr2, :] = [ii, ny - (j+1), i]
                solid_nodes_symmetric[ctr2, :] = [ii, ny_global-(j+1), i]  # Use global indices
                
                #print("y_solid_symmetric[",ctr2,"]=",y_solid_symmetric[ctr2])
                #print("xsolid_symmetric_index[",ctr2,"]=",ii)
                #print("ysolid_symmetric_index[",ctr2,"]=",ny_global-(j+1))
                #print("zsolid_symmetric_index[",ctr2,"]=",i)
                #input("stop here ...")
                #print("x_solid[",ctr2,"]=",x_solid_symmetric[ctr2])

                ctr2 += 1

            #if (y[j] <= y_sin[ii]).all() and (y[j + 1] >= y_sin[ii]).all(): 
            if (global_y[j] <= y_sin[ii]).all() and (global_y[j + 1] >= y_sin[ii]).all(): 

                # Ghost nodes
                y_ghost[ctr3] = global_y[j]
                z_ghost[ctr3] = z[i]
                x_ghost[ctr3] = x[ii]
                ghost_nodes[ctr3, :] = [ii, j, i]
                
                #print("y_ghost[",ctr3,"]=",y_ghost[ctr3])
                #print("z_ghost[",ctr3,"]=",z_ghost[ctr3])
                #print("x_ghost[",ctr3,"]=",x_ghost[ctr3])
                #print("ghost_nodes[",ctr3,"]=",ghost_nodes[ctr3, :])
                #print(j)

                #input("stop here ...")
                # Symmetric ghost nodes
                y_ghost_symmetric[ctr3] = global_y[ny_global - (j+1)]
                z_ghost_symmetric[ctr3] = z[i]
                x_ghost_symmetric[ctr3] = x[ii]
                
                ghost_nodes_symmetric[ctr3, :] = [ii, ny_global - (j+1), i]
                
                #print("y_ghost_symmetric[",ctr3,"]=",y_ghost[ctr3])
                #print("z_ghost_symmetric[",ctr3,"]=",z_ghost[ctr3])
                #print("x_ghost_symmetric[",ctr3,"]=",x_ghost[ctr3])
                #print("ghost_nodes_symmetric[",ctr3,"]=",ghost_nodes_symmetric[ctr3, :])
                #print(ny-(j+1))
               # input("stop here ...")
               
                ctr3 += 1


solid_nodes2 = np.zeros((len(solid_nodes)+ len(solid_nodes_symmetric), 3), dtype=np.int64)
ghost_nodes2 = np.zeros((len(ghost_nodes) + len(ghost_nodes_symmetric), 3), dtype=np.int64)

#print(np.size(solid_nodes))

#if not isinstance(solid_nodes, np.ndarray):
#    solid_nodes = np.array(solid_nodes)
    
solid_nodes2[0:len(solid_nodes), :] = solid_nodes
solid_nodes2[len(solid_nodes):len(solid_nodes2), :] = solid_nodes_symmetric

ghost_nodes2[0:len(ghost_nodes), :] = ghost_nodes
ghost_nodes2[len(ghost_nodes):len(ghost_nodes2), :] = ghost_nodes_symmetric

#print(ghost_nodes2)
#print(np.size(solid_nodes))
#print(np.size(ghost_nodes))

#print(np.size(solid_nodes2))
#print(np.size(ghost_nodes2))

# Plot the solid and ghost nodes
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(solid_nodes2[:, 2], solid_nodes2[:, 1], solid_nodes2[:, 0], c='b', marker='*')
ax.scatter(ghost_nodes2[:, 2], ghost_nodes2[:, 1], ghost_nodes2[:, 0], c='g', marker='o')
ax.set_xlabel('Z')
ax.set_ylabel('Y')
ax.set_zlabel('X')
ax.view_init(elev=30, azim=60)  # elev is elevation angle, azim is azimuth angle
#plt.show()  # This triggers the plot window to appear.

plt.savefig('FSI sin wave.png')


np.savetxt('solid_nodes_dedalus.csv', solid_nodes2, delimiter=',')
np.savetxt('ghost_nodes_dedalus.csv', ghost_nodes2, delimiter=',')


#input("stop here..")

'''
for i in range(nx):
    for j in range(ny):
        for k in range(nz):

            # Check for solid nodes
            for r in range(len(solid_nodes2)):
                
                if i == solid_nodes2[r, 0] and j == solid_nodes2[r, 1] and k == solid_nodes2[r, 2]:
                    
                    inv_k[i, j, k] = 10**8

            # Check for ghost nodes
            for l in range(len(ghost_nodes2)):
                
                if i == ghost_nodes2[l, 0] and j == ghost_nodes2[l, 1] and k == ghost_nodes2[l, 2]:
                    inv_k[i, j, k] = 10000

'''

# Problem
#problem = d3.IVP([p, u, tau_p, tau_u1, tau_u2], namespace=locals())
problem = d3.IVP([p, u, tau_p, tau_u1, tau_u2], namespace=locals())

problem.add_equation("trace(grad_u) + tau_p = 0")
#problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*ex -dot(u,grad(u))")
problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*ex -dot(u,grad(u)) -inv_k*u")
#problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2) = -dPdx*ex - dot(u, grad(u)) - inv_k*u[0]*ex - inv_k*u[1]*ey - inv_k*u[2]*ez")


problem.add_equation("u(y=-1) = 0") # change from -1 to -0.5
problem.add_equation("u(y=+1) = 0") # change from 1 to 0.5
problem.add_equation("integ(p) = 0")

'''
for r in range(len(solid_nodes2)):
    i, j, k = solid_nodes2[r]
    print(i,j,k)
    print(solid_nodes2[r])
    input("stop here..")
    inv_k['g'][i, j, k] = 10**8  # High stiffness in solid nodes

for l in range(len(ghost_nodes2)):
    i, j, k = ghost_nodes2[l]
    inv_k['g'][i, j, k] = 10000  # Lower stiffness in ghost nodes

'''

print("solid_nodes2 shape:", solid_nodes2.shape)
print("ghost_nodes2 shape:", ghost_nodes2.shape)

'''

for r in range(len(solid_nodes2)):
    i, j, k = solid_nodes2[r]

    # Check if indices are within bounds of inv_k['g']
    if 0 <= i < inv_k['g'].shape[0] and 0 <= j < inv_k['g'].shape[1] and 0 <= k < inv_k['g'].shape[2]:
        print(f"Setting inv_k at ({i}, {j}, {k}) to 10^8")
        inv_k['g'][i, j, k] = 10**8  # High stiffness in solid nodes
    else:
        print(f"Skipping out-of-bounds index ({i}, {j}, {k})")

for l in range(len(ghost_nodes2)):
    i, j, k = ghost_nodes2[l]

    # Check if indices are within bounds of inv_k['g']
    if 0 <= i < inv_k['g'].shape[0] and 0 <= j < inv_k['g'].shape[1] and 0 <= k < inv_k['g'].shape[2]:
        print(f"Setting inv_k at ({i}, {j}, {k}) to 10000")
        inv_k['g'][i, j, k] = 10000  # Lower stiffness in ghost nodes
    else:
        print(f"Skipping out-of-bounds index ({i}, {j}, {k})")
        
'''
# Debug to check if i,j,k is indeed getting captured correctly for the stiffness matrix
#for i, j, k in solid_nodes2:
#    print(f"Current node: ({i}, {j}, {k})")  # Debug output
#    if i != 0 or j != 0 or k != 0:
#        print(f"Stopping: i={i}, j={j}, k={k}") 



for i, j, k in solid_nodes2:
    # Check if indices are within bounds
    if 0 <= i < inv_k['g'].shape[0] and 0 <= j < inv_k['g'].shape[1] and 0 <= k < inv_k['g'].shape[2]:
       # print(f"Setting inv_k at ({i}, {j}, {k}) to 10^8", inv_k['g'][i,j,k])
        inv_k['g'][i, j, k] = (10**2)  # High stiffness in solid nodes 10**8 to 10**2
       # print(f"Setting inv_k at ({i}, {j}, {k}) to 10^2", inv_k['g'][i,j,k])
            #print(f"Stopping: i={i}, j={j}, k={k}") 
               
    else:
        print(f"Skipping out-of-bounds index ({i}, {j}, {k})")


for i, j, k in ghost_nodes2:
    if 0 <= i < inv_k['g'].shape[0] and 0 <= j < inv_k['g'].shape[1] and 0 <= k < inv_k['g'].shape[2]:
        inv_k['g'][i, j, k] = 10  # Lower stiffness in ghost nodes, 10^4 to 10
       # print(f"Setting inv_k at ({i}, {j}, {k}) to 10", inv_k['g'][i,j,k])
    else:
        print(f"Skipping out-of-bounds index ({i}, {j}, {k})")


print(np.size(inv_k))
print(np.shape(inv_k))

print(inv_k['g'])
#input("stopping here..")

print(np.size(u))
print(np.shape(u))

#input("stop here ...")


#if np.all(x == x_ghost) and np.all(y == y_ghost) and np.all(z == z_ghost):
#    problem.add_equation("inv_k = 100") 

#else:
#    problem.add_equation("inv_k = 0") 


#problem.add_equation("body_force(x=x_ghost,y=y_ghost,z=z_ghost) = 100*u") 

#problem.add_equation("inv_k(x=x_solid,y=y_solid,z=z_solid) = 10^6") 

# Build Solver
dt = 0.002 # 0.001
stop_sim_time = 1000 #10000
fh_mode = 'overwrite'
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions (this would be in dedalus 2)
#u = solver.state['u']
#uy = solver.state['uy']

# Random perturbations, initialized globally for same results in parallel
#gshape = domain.dist.grid_layout.global_shape(scales=1)r
#slices = domain.dist.grid_layout.slices(scales=1)
#rand = np.random.RandomState(seed=42)
#noise = rand.standard_normal(gshape)[slices]

# Laminar solution + perturbations damped at walls
#yb, yt = y_basis.interval
#pert =  4e-1 * noise * (yt - y) * (y - yb)

np.random.seed(0)
#u['g'][0] = (1-y**2) + np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation
u['g'][0] = (1 - y[np.newaxis, :, np.newaxis]**2) + np.random.randn(*u['g'][0].shape) * 1e-6 * np.sin(np.pi * (y[np.newaxis, :, np.newaxis] + 1) * 0.5)

#inv_k['g'] = np.zeros_like(inv_k['g'])  # Start with all zeros


#print(np.size(inv_k['g']))

#print(y)
#print(x)


# Initialize inv_k values
#inv_k['g'] = np.zeros_like(inv_k['g'])

# Set inv_k = 100 at ghost nodes
#for j in range(len(x)):
#    for i in range(len(x_ghost)):
#        mask = (x[j] == x_ghost[i]) & (y[j] == y_ghost[i]) & (z[j] == z_ghost[i])
#        inv_k['g'][mask] = 100

#u['g'][1] = np.random.randn(*u['g'][1].shape) * 1e-8*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation

#u.set_scales(1/4, keep_data=True)
#u['g'][0]
#u.set_scales(1, keep_data=True)
#u.differentiate('y', out=uy)

snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=10, max_writes=400)
#snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=0.25)


snapshots.add_task(u, name='velocity')

snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=1, max_writes=400)
snapshots_stress.add_task(xz_average(u),name = 'ubar')
snapshots_stress.add_task(inv_k, name='stiffness')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')

snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')


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
        if (solver.iteration-1) % 10 == 0:
            max_TKE = flow.max('TKE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(TKE)=%f' %(solver.iteration, solver.sim_time, timestep, max_TKE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
