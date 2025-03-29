import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = '.\snapshots_breaking_waves\snapshots_breaking_waves_s1.h5'
# check name of saved variables
with h5py.File(filename, 'r') as file:
    # Recursively list all groups and datasets
    def print_attrs(name, obj):
        print(name)
    file.visititems(print_attrs)

# Read in the data
with h5py.File(filename, 'r') as f:
    # Eg. Access dataset '1.0' in group 'scales/x'
    x = f['scales/x_hash_e879951bed887acb6bcc55bce5faf13406e7c2c3'][:]
    y = f['scales/y_hash_f0fbd0dd64efa218bf9d9c29edf6776573eed9db'][:]
    z = f['scales/z_hash_f785bb960fc9ccf35206fce6efdc3270cf195186'][:]
    velocity = f['tasks/velocity'][:]
    vorticity = f['tasks/vorticity'][:]
    #w = f['tasks/w'][:]

tind=5
yind=1

u=velocity[tind,0,:,yind,:]
w=velocity[tind,2,:,yind,:]

omega_x=vorticity[tind,1,:,yind,:]

def plot_scalar(x,y,scalar,vmin=-1,vmax=1):
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X,Y,scalar.T, shading='gouraud', cmap='bwr',vmin=vmin,vmax=vmax, zorder=1)

def plot_streamline(x,y,vec_x,vec_y):
    X, Y= np.meshgrid(x,y)
    sort_indices = np.argsort(Y[:,0])
    Y_sorted = Y[sort_indices, :]
    X_sorted = X[sort_indices, :]
    vec_x_sorted = vec_x[:, sort_indices]
    vec_y_sorted = vec_y[:, sort_indices]
    speed = np.sqrt(vec_x_sorted**2 + vec_y_sorted**2)
    lw = 1.2 * speed / speed.max()
    plt.streamplot(X_sorted, Y_sorted, vec_x_sorted.T, vec_y_sorted.T,
                   density = 0.7, color ='k', linewidth = lw.T, arrowstyle ='->')
   
X_sorted=x
Y_sorted=z

vec_x_sorted=u
vec_y_sorted=w
    
##interpolation 
from scipy.interpolate import interp1d

# Ensure X remains unchanged
X_uniform = X_sorted  # No changes in x-direction

# Create evenly spaced Y values
Y_uniform = np.linspace(Y_sorted.min(), Y_sorted.max(), len(Y_sorted))

# Interpolate vector fields in the y-direction for each x
vec_x_interp = np.zeros((len(X_sorted), len(Y_uniform)))
vec_y_interp = np.zeros((len(X_sorted), len(Y_uniform)))

for i in range(len(X_sorted)):  # Loop through each x value
    interp_func_x = interp1d(Y_sorted, vec_x_sorted[i, :], kind='linear', fill_value='extrapolate')
    interp_func_y = interp1d(Y_sorted, vec_y_sorted[i, :], kind='linear', fill_value='extrapolate')

    vec_x_interp[i, :] = interp_func_x(Y_uniform)
    vec_y_interp[i, :] = interp_func_y(Y_uniform)    

plt.figure(figsize=(4, 3))
#[y,z,scalar] = z_averaged_scalar('Continuation/Solution1/initial-1/tbest.nc',1)
plot_scalar(x,z,omega_x,np.min(omega_x),-np.min(omega_x))
#print(np.min(scalar),np.max(scalar))
#[y,x,vec_x,vec_y,vec_z] = z_averaged_vec('Continuation/Solution1/initial-1/ubest.nc',1)
plot_streamline(X_uniform,Y_uniform,vec_x_interp,vec_y_interp)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$z$', fontsize=14)
plt.xticks([0,1,2,3,4], ['0', '1', '2', '3', '4'],fontsize=14)
plt.yticks([-3,-2,-1,0],['-3','-2','-1','0'],fontsize=14)
plt.xlim(0, 4)
plt.ylim(-3,0)
plt.savefig('breaking_waves.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#for i in range(np.shape(u)[0]):
    # saveScalarToVTK(x,y,z,u[i],'u','u'+str(i))
    # Merge u, v, w into one data 'velocity'
    #velocity = np.stack((u[i], v[i], w[i]), axis=-1)
    #saveVectorToVTK(x,y,z,velocity,'velocity','velocity'+str(i))