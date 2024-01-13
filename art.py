import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import scipy.spatial as spatial
import matplotlib as mpl
import copy
from pathlib import Path

def make_pointcloud_cube(N=100):
    s = np.linspace(-1,1,N)
    f,g = np.meshgrid(s,s)
    i = f.flatten()
    r = g.flatten()
    o = np.ones(N*N)
    z = -np.ones(N*N)
    
    bottom = np.vstack((i,r,z))
    top = np.vstack((i,r,o))
    left = np.vstack((i,z,r))
    right = np.vstack((i,o,r))
    front = np.vstack((z,i,r))
    back = np.vstack((o,i,r))
    
    return np.hstack((bottom,top,left,right,front,back))


def make_pointcloud_cube_wireframe(N=100):
    s = np.linspace(-1,1,N)
    o = np.ones(N)
    z = -np.ones(N)
    
    bf = np.vstack((s,z,z))
    bb = np.vstack((s,z,o))
    tf = np.vstack((s,o,z))
    tb = np.vstack((s,o,o))
    lb = np.vstack((z,s,z))
    lt = np.vstack((z,s,o))
    rb = np.vstack((o,s,z))
    rt = np.vstack((o,s,o))
    ft = np.vstack((z,z,s))
    fb = np.vstack((z,o,s))
    Bt = np.vstack((o,z,s))
    Bb = np.vstack((o,o,s))

    return np.hstack((bf,bb,tf,tb,lb,lt,rb,rt,ft,fb,Bt,Bb))

    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(rot[0,:],rot[1,:],rot[2,:], marker=',')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    

def make_pointcloud_sphere(N=50):
    s = np.linspace(-1,1,N)
    x,y,z = np.meshgrid(s,s,s)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    r = np.sqrt(x**2+y**2+z**2)
    sphere = np.vstack((x,y,z))
    return sphere[:,r<=1]
    


def rotate_pointcloud(ply, x, y, z):
    # performs tait_bryan rotation - https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    g = np.deg2rad(z) # Z
    b = np.deg2rad(y) # Y
    a = np.deg2rad(x) # X
    tait_bryan = np.array([
        [np.cos(b)*np.cos(g), np.sin(a)*np.sin(b)*np.cos(g) - np.cos(a)*np.sin(g), np.cos(a)*np.sin(b)*np.cos(g) + np.sin(a)*np.sin(g)],
        [np.cos(b)*np.sin(g), np.sin(a)*np.sin(b)*np.sin(g) + np.cos(a)*np.cos(g), np.cos(a)*np.sin(b)*np.sin(g) - np.sin(a)*np.cos(g)],
        [     -np.sin(b)    ,             np.sin(a)*np.cos(b)                    ,             np.cos(a)*np.cos(b)                    ]
    ])
    center = ply
    return np.einsum('ji,ik->jk', tait_bryan,ply)
    
def translate_pointcloud(ply,dist):
    return ply +  np.repeat(np.array(dist)[:,np.newaxis], ply.shape[-1], axis=1)
    
def add_matrices(m1, m2, method='max'):
    if method =='max':
        return np.maximum(m1,m2)
    if method =='add':
        return np.maximum(m1+m2,255)
    
def add_matrixlists(ml1, ml2, method='max'):
    if len(ml1) != len(ml2):
        raise Warning("matrix lists of different length! zipping")
    return [add_matrices(m1,m2, method=method) for m1,m2 in zip(ml1,ml2)]

def add_multiple_matrices(mlist, method='max'):
    out = copy.deepcopy(mlist[0])
    for m in mlist[1:]:
        out = add_matrices(out, m, method=method)
    return out
    
def pointcloud_to_matrix(ply, matrix_shape, tol=0.05, color=[0,255,0]):
    matrix = np.zeros_like(matrix_shape)
    
    x = np.linspace(-1,1, matrix_shape[0])
    y = np.linspace(-1,1, matrix_shape[1])
    z = np.linspace(-1,1, matrix_shape[2])
    xm,ym,zm = np.meshgrid(y,x,z)
    points = np.vstack((xm.flatten(),ym.flatten(),zm.flatten()))
    
    kt = spatial.cKDTree(ply.T)
    dist, index = kt.query(points.T, distance_upper_bound=tol, workers=-1)
    matrix = np.einsum('i,j->ij',np.isfinite(dist), np.array(color))
    return matrix.reshape(matrix_shape).astype('uint8')



def wheel(val, map_name = 'hsv'):
    cmap=mpl.colormaps[map_name]
    norm=mpl.colors.Normalize(vmin=0,vmax=1)
    scalarmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return (255*np.array(scalarmap.to_rgba(val))[...,:3]).astype('uint8')
    

def test_matrix(matrix_shape, bright=1):
    matrix = np.zeros(matrix_shape).astype('uint8')
    val = np.linspace(0,1,matrix_shape[0]*matrix_shape[1])
    col = bright* wheel(val)
    matrix[:,:,0,:] = col.reshape((matrix_shape[0],matrix_shape[1],matrix_shape[3])).astype('uint8')
    
    matrix_list = [matrix]
    for _ in range(matrix_shape[2]):
        matrix = step_up(matrix)
        matrix_list += [matrix]
    return matrix_list
        
def generator_zipup(matrix, xy, color):
    matrix[xy[0],xy[1],0, :] = color
    
def step_up(matrix):
    m = np.roll(matrix, 1, axis = 2)
    m[...,0,-3:] = np.zeros_like(m[...,0,-3:])
    return m.astype(np.uint8)
    
def seed_random_bottom(matrix,color=[0,255,0]):
    if color == 'random':
        color = (np.random.random( 3 ) * 256).astype(np.uint8)
    x = np.random.randint(0,matrix.shape[0])
    y = np.random.randint(0,matrix.shape[1])
    matrix[x,y,0, :] = color
    return matrix
    
def postprocess_antialiasing(matrix):
    raise NotImplementedError

def supersample_antialisaing(matrix_shape:tuple, sample_factor: int, func, **kwargs):
    supersample_shape = tuple(np.array(matrix_shape[:-1])*sample_factor) +(3,)
    tgrid =  np.meshgrid(
        np.linspace(-1,1,matrix_shape[0]),
        np.linspace(-1,1,matrix_shape[1]),
        np.linspace(-1,1,matrix_shape[2])
    )
    target_points = np.vstack([t.flatten() for t in tgrid])
    sample_grid = (
        np.linspace(-1,1,supersample_shape[0]),
        np.linspace(-1,1,supersample_shape[1]),
        np.linspace(-1,1,supersample_shape[2]))

    matrix_list = func(supersample_shape, **kwargs)
    out_list = []
    for matrix in matrix_list:
        r = RegularGridInterpolator(sample_grid, matrix[...,0], method='linear')(target_points.T)
        g = RegularGridInterpolator(sample_grid, matrix[...,1], method='linear')(target_points.T)
        b = RegularGridInterpolator(sample_grid, matrix[...,2], method='linear')(target_points.T)
        out_list += [np.reshape(np.vstack((r,g,b)).T, matrix_shape).astype('uint8')]
    return out_list
    
    
def add_random_noise(matrix, N, color=[0,0,255], shape=None, rand_min_bright=False):
    m = copy.deepcopy(matrix)
    if type(shape)==type(None):
        shape = matrix.shape
    for _ in range(N):
        x = np.random.randint(0,matrix.shape[0]) 
        y = np.random.randint(0,matrix.shape[1]) 
        z = np.random.randint(0,matrix.shape[2])
        if type(rand_min_bright)==float: 
            b = np.random.uniform(rand_min_bright,1.0)
        else:
            b = 1
        m[x,y,z, :] = np.array(b*color).astype('uint8')
    return matrix.astype('uint8')

def lighting_strike(matrix_shape, duration=30, color=[240,200,255]):
    matrix = np.zeros(matrix_shape)
    x = np.random.randint(0,matrix_shape[0]) 
    y = np.random.randint(0,matrix_shape[1]) 
    
    max_bright = np.random.uniform(0.5,1)
    color_arr = max_bright* np.repeat(np.array(color)[:,np.newaxis], matrix_shape[2], axis=1).T
    
    
    # Data from: https://www.researchgate.net/figure/Plot-of-the-PH-light-curve-data-for-the-lightning-event-at-045021-UT-on-14-October_fig1_319345753
    strike_time = [0.0,0.003094132045724507,0.017981792066441793,0.029244031369608958,0.03288201160541693,0.042768106597895095,0.05456423067442806,0.07135171078530833,0.09188762823601526,0.11521379352250305,0.12962496366334997,0.150413843143125,0.17616997701187254,0.20149684198280782,0.22495332104184596,0.22823416190369827,0.2304418305210194,0.23183082202608407,0.25021886188452225,0.2558300196202137,0.2563819367745437,0.2623303772156593,0.2827513119258809,0.2910607313049649,0.3040537809798243,0.31495874408746727,0.3292350011461447,0.34270177971180393,0.36796118814165446,0.3930120945354245,0.4086650782735144,0.41492013935592453,0.417580055877929,0.42011735922586846,0.44302192113057615,0.4610512148386996,0.4815641357413094,0.5057565043394545,0.5315433002723315,0.5577549848081453,0.5788855272882203,0.6047643094134854,0.6304897812181038,0.6550500945858024,0.679334449376336,0.7036188041668696,0.7258794627248584,0.7521875137479364,0.7762265720254344,0.7989164994812361,0.8220356958348503,0.8455841610862764,0.8682127644138187,0.8931716846152007,0.9191117908687252,0.9448679247374727,0.9701028035160197,0.9950310616532718,1.0]
    strike_brighntess = [1.0, 1.0, 0.8702780333534506, 0.7087278267493717, 0.6161616161616168, 0.4949494949494957, 0.3666142324320587, 0.30500648007993814, 0.25042417317148047, 0.1778279035869641, 0.12450657406583154, 0.06397965947427409, 0.02939285113624024, 0.004893861896800189, 0.024348941586945116, 0.13747662719259246, 0.08559641468554346, 0.21940412944330756, 0.2628538074179616, 0.3925543386855832, 0.31617513693909416, 0.4775802425165807, 0.4257000300095317, 0.37309925899544005, 0.3055468989602206, 0.23950771178978858, 0.18935683969964234, 0.13920596760949522, 0.08775809020667147, 0.06974412753061188, 0.1061323321362515, 0.17314427329118942, 0.2424242424242431, 0.26841240161514435, 0.22610532355880153, 0.182871813136261, 0.128829925108084, 0.08343473916441724, 0.0481273723193425, 0.03680431006582019, 0.07694971260103678, 0.03948067023483404, 0.005614420403842857, 0.0008706061595376013, 0.0008706061595376013, 0.0008706061595376013, 0.0008706061595376013, 0.0008706061595376013, 0.004893861896800189, 0.04236290426300293, 0.06974412753061188, 0.0870375316996288, 0.1259476910799151, 0.11009540392498351, 0.07118524454469721, 0.03659843620666514, 0.014261122488351319, 0.0, 0.0]
    brightness = np.interp(np.linspace(0,1,duration), strike_time, strike_brighntess)
    
    matrix_list=[]
    for b in brightness:
        matrix[x,y,:,:] = np.array(b*color_arr).astype('uint8')
        matrix_list += [matrix.astype('uint8')]
    return matrix_list

def generate_perlin_noise_3d(shape, res=[3,3,1]):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    gradients[-1] = gradients[0]
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def generate_perlin_noise_1d(len, amplitude=1, frequency=1, octaves=1):
    # result = 0
    # x = np.arange(len)
    
    # n = np.random.uniform(-1,1, size=len+4)
    
    
    # nm2 = n[:-4]
    # nm1 = n[1:-3]
    # n0 = n[2:-2]
    # np1 = n[3:-1]
    # np2 = n[4:]

    # def __cosine_interp(a, b, x):
    #     x2 = (1 - np.cos(x * np.pi)) / 2
    #     return a * (1 - x2) + b * x2
        
    # for o in range(octaves):
    #     xo = x * frequency*2**o
    #     result += amplitude/(2**o) * __cosine_interp()
    pass

    
    
def generate_clouds(matrix_shape, height, periods=[4,4,1], color=[255,255,255], duration=100):
    cloud_shape = matrix_shape[:2]+ (height,)
    rem = periods[2] - (height+duration)%periods[2] # add remainder so period fits integer
    cloudset_shape = matrix_shape[:2]+ (height+duration+rem,)
    noise = 0.5*generate_perlin_noise_3d(cloudset_shape, periods) +0.5 ## bring to range 0 to 1
    matrix_list = []
    for i in range(duration):
        matrix = np.einsum('ijk,c->ijkc', noise[:,:,i:i+height], color)
        pad = ((0,0), (0,0), (matrix_shape[2] - cloud_shape[2],0), (0,0))
        matrix_list += [np.pad(matrix, pad_width=pad, mode='constant', constant_values=0).astype('uint8')]
    return matrix_list
    
    
def load_ply(dir):
    with open(dir,'r') as file:
        lines = file.readlines()
        i=0
        while i < 100:
            if lines[i] == "end_header\n":
                break
            i+=1
    ply = np.loadtxt(dir,skiprows=i+1).T
    return ply/max( np.abs(np.min(ply)), np.max(ply))
    
    
def render_particles(pos, color, matrix_shape):
    indexset = np.round( (pos+1)/2 *np.array(matrix_shape[:3]), 0).astype(int)
    indexset = np.maximum(indexset,0)

        
    matrix = np.zeros(matrix_shape)
    if len(indexset.shape)<2:
        indexset = np.minimum(indexset, np.array(matrix_shape[:-1])-1)
        matrix[indexset[0],indexset[1],indexset[2],:] = color
    else:
        for k in range(3):
            indexset[:,k] = np.minimum(indexset[:,k], matrix_shape[k]-1)
        for j,i in enumerate(indexset):
            matrix[i[0],i[1],i[2],:] = color[j,:]
    return matrix.astype('uint8')
     
    
def propagate_particles(pos, vel, alive=None, dt=1, g=-0.01, a=[0,0,0]):
    pos = pos + vel*dt
    if len(vel.shape) <2:
        vel = vel+ dt*np.array([0,0,g]) + dt*np.array(a)
    else:
        vel = vel + dt*np.tile([0,0,g], (pos.shape[0],1))
    if not type(alive)==type(None):
        alive += -dt
    return pos, vel, alive

def trim_dead(pos,vel,alive,col, alive_time_limit=-10):
    out_of_bounds = np.where(np.sum(np.abs(pos)>1, axis=-1))[0]
    timeout = np.where(alive < alive_time_limit)[0]
    dead = np.unique(np.append(timeout,out_of_bounds))
    if len(pos.shape) > 1:
        pos = np.delete(pos, dead, axis=0)
        vel = np.delete(vel, dead, axis=0)
        col = np.delete(col, dead, axis=0)
        alive = np.delete(alive, dead, axis=0)
    return pos, vel, alive, col

def wall_bounce(pos,vel, elastic=1):
    for i in np.where(np.sum(np.abs(pos)>1, axis=-1))[0]:
        if len(vel.shape) <2:
            for j in np.where(np.abs(pos) >1):
                vel[j] *= -elastic
                pos[j] = np.sign(pos[j])
        else:
            for j in np.where(np.abs(pos[i,:]) >1)[0]:
                vel[i, j] *= -elastic
                pos[i, j] = np.sign(pos[i, j])
    return pos, vel
    
def color_points(val, map="rainbow"):
    return  wheel((val+1)/2, map_name=map)
    
def explode_missile(pos, vel, alive, col, dt, num_stars=10, vel_burst=0.1, crand="mix"):
    fuse = (alive+dt >0) * (alive<=0)
    apogee = (vel[...,2] <=0) * (alive>=0)
    boundary = (alive>=0)*((pos[...,0] <=-1) + (pos[...,0] >=1) + (pos[...,1]<=-1) + (pos[...,1]>=1))
    explode = np.where(fuse+apogee+boundary)[0]
    for i in explode:
        if len(pos.shape) <2:
            pos = np.vstack((pos, np.tile(pos, (num_stars,1)) ))
            vel = np.vstack((vel, vel_burst*np.random.uniform(-1,1,size=(num_stars,3)) ))
        else:
            pos = np.vstack((pos, np.tile(pos[i,:], (num_stars,1)) ))
            vel = np.vstack((vel, np.tile(vel[i,:], (num_stars,1)) + vel_burst*np.random.uniform(-1,1,size=(num_stars,3)) ))
        if crand == "all":
            col = np.vstack((col, wheel(np.random.uniform(size=(num_stars)))    ))
        if crand == "single":
            col = np.vstack((col, np.tile(wheel(np.random.uniform(), (num_stars,1)))    ))
        if crand == "mix":
            if np.random.uniform() <0.3:
                col = np.vstack((col, wheel(np.random.uniform(size=(num_stars)))    ))
            else:
                col = np.vstack((col, np.tile(wheel(np.random.uniform()), (num_stars,1)))    )


            
        alive = np.append(alive, np.zeros(num_stars))
    pos = np.delete(pos, explode, axis=0)
    vel = np.delete(vel, explode, axis=0)
    col = np.delete(col, explode, axis=0)
    alive = np.delete(alive, explode, axis=0)
    return pos, vel, alive, col
    

def new_missile(pos, vel, alive, col, alive_time=10, spread=0.05, brightness=0.5):
    pos = np.vstack((pos,[ 0.5*np.random.normal(), 0.5*np.random.normal(), -1 ]  ))
    vel = np.vstack((vel,[ spread*np.random.normal(), spread*np.random.normal(), np.random.uniform(0.4, 0.62) ]))
    col = np.vstack((col, (brightness*wheel(np.random.uniform(0,1),map_name='hsv')).astype('uint8')  ))
    alive = np.append(alive,alive_time)
    return pos, vel, alive, col

def twinkle(matrix_shape, inds, start=0, ramp_up=10, dwell=2, ramp_down=10, color=[112, 102, 79], duration=100):
    matrix = np.zeros(matrix_shape)
    brightness = np.interp(
        np.arange(duration), 
        np.append(np.cumsum([0,start, ramp_up, dwell, ramp_down]),duration),
        [0,0,1,1,0,0])
    
    matrix_list = []
    for b in brightness:            
        matrix[inds[0],inds[1],inds[2]] = b * np.array(color)
        matrix_list += [matrix.astype('uint8')]
    return matrix_list
    