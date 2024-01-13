import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import scipy.spatial as spatial
import matplotlib as mpl
from pathlib import Path


import art

def makeset_wave(matrix_shape, frames=100, coloring='gist_rainbow'):
    x = np.linspace(-1,1, matrix_shape[0])
    y = np.linspace(-1,1, matrix_shape[1])
    xm,ym = np.meshgrid(x,y)

    r = np.sqrt(xm**2 +ym**2)
    base_value = np.cos(r*8)*np.exp(-np.abs(r**2)**1.0)
    
    
    matrix_list = []
    for t in np.linspace(0,2*np.pi,frames):
        value = base_value*np.cos(t)
        zcurr = (matrix_shape[-2]*((value+1)/2)).astype(int) ## z index of wave
        
        if isinstance(coloring, str): # if cmap key is passed
            cmap = plt.get_cmap(coloring)
            norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
            ScalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
            color = (ScalarMap.to_rgba(value)[...,0:3] *256).astype("uint8") # remove alpha (always just 1)
            
        else: # assume rgb color is passed
            color = np.tile(coloring, matrix_shape[:-2]+(1,)) 
        
        matrix = np.zeros(matrix_shape).astype("uint8")
        for ix in range(matrix_shape[0]):
            for iy in range(matrix_shape[1]):
                matrix[ix,iy,zcurr[ix,iy],:] = color[ix,iy,:]
        matrix_list += [matrix]
    return matrix_list


def cube_rotates(matrix_shape):
    cube = 0.65*art.make_pointcloud_cube_wireframe()
    cuber = art.rotate_pointcloud(cube, 45,20,5)
    # cubec = translate_pointcloud()
    sphere = art.make_pointcloud_sphere()
    cube2 = 0.7*cuber
    
    matrix_list = []
    for a in np.linspace(0,360,200):
        cuber = art.rotate_pointcloud(cuber, 0,0.5,2)
        cube2 = art.rotate_pointcloud(cube2, -1,0,-3)
        sph = (0.1 + 0.2*np.sin(a/10))*sphere
        
        m1 = art.pointcloud_to_matrix(cuber, matrix_shape, tol=0.1)
        m2 = art.pointcloud_to_matrix(cube2, matrix_shape, tol=0.075, color=[255,0,0])
        m3 = art.pointcloud_to_matrix(sph, matrix_shape, tol=0.075, color=[0,0,255])
        
        matrix = art.add_matrices(m3,art.add_matrices(m1,m2))
        matrix_list += [matrix]
    return matrix_list



def thunderstorm(matrix_shape, lighting_freq=0.3, duration=100):
    cloudset = art.generate_clouds(matrix_shape, height=3, periods=[5,5,5], color=[100,100,100], duration=duration)

    lightning_flag= False
    matrix_list = []
    for i in range(duration):
        blank = np.zeros(matrix_shape)
        rain = art.add_random_noise(blank, 100, color=[0,0,50])
        scene = art.add_matrices(rain,cloudset[i])
        
        if lightning_flag:
            # propagate lighting strike
            scene = art.add_matrices(scene,lightning[j])
            j += 1
            if j == len(lightning):
                lightning_flag= False 

        else:
            # chance of new lighting strike
            if np.random.uniform() < lighting_freq:
                lightning = art.lighting_strike(matrix_shape, color=[200,190,255], duration = np.random.randint(10,50))
                scene = art.add_matrices(scene,lightning[0])
                j = 1
                lightning_flag= True        
        matrix_list += [scene]
    return matrix_list

def face_bounce(matrix_shape, axis=0, color=[0,255,0]):
    face = art.load_ply(Path.joinpath(Path(__file__).parent ,"faceMesh.ply"))
    pos = art.rotate_pointcloud(face,0,0,90).T
    
    n = pos.shape[0]
    vel = np.zeros((n,3))
    vel[:,axis] = -np.ones(n)*2/(matrix_shape[axis])
    col = color*np.ones((n,3)).astype('uint8')
    
    matrix_list = []
    for i in range(matrix_shape[axis]*2):
        pos, vel, _  = art.propagate_particles(pos, vel, dt=1, g=0.0)
        pos, vel = art.wall_bounce(pos, vel)
        matrix_list += [art.render_particles(pos, col, matrix_shape)]
    return matrix_list

def face_sweep(matrix_shape, duration=50):
    face = art.load_ply(Path.joinpath(Path(__file__).parent ,"faceMesh.ply"))
    # face = art.rotate_pointcloud(face,90,0,0)
    face = art.translate_pointcloud(face,[0,-0.5,0])
    matrix_list = []
    for i in range(duration):
        face = art.translate_pointcloud(face,[0,0.1,0])
        matrix_list += [art.pointcloud_to_matrix(face, matrix_shape, tol=0.15)]
    return matrix_list

def head_rotate(matrix_shape, duration=75):
    face = art.load_ply   (Path.joinpath(Path(__file__).parent ,"female_head.ply"))
    matrix_list = []
    for i in range(duration):
        face = face = art.rotate_pointcloud(face,0,0,360/duration)
        matrix_list += [art.pointcloud_to_matrix(face, matrix_shape, tol=0.15, color=list(art.wheel(i/duration,map_name='rainbow')))]
    return matrix_list

def underwater(matrix_shape, duration=100):
    s1 = art.generate_clouds(matrix_shape, height=2, periods=[10,10,10], color=[70,217,250], duration=duration)
    s2 = art.generate_clouds(matrix_shape, height=2, periods=[10,10,10], color=[56,93,255], duration=duration)
    s3 = art.generate_clouds(matrix_shape, height=2, periods=[10,10,10], color=[57,247,244], duration=duration)
    
    vol = art.generate_clouds(matrix_shape, height=matrix_shape[2], periods=[10,10,10], color=[0,0,140], duration=duration)
    return [art.add_multiple_matrices([s1[i],s2[i],s3[i],vol[i]]) for i in range(duration)]
        

def fireworks(matrix_shape, duration=100, spawn_rate=0.25, dt=0.5):
    pos = np.array([0,0, -1])
    vel = np.array([0,0,0.5])
    alive = np.array([10.0])
    col = np.array([151, 15, 255]).astype('uint8')
    matrix_list = []
    for i in range(duration):
        if np.random.uniform(0,1) < spawn_rate*dt:
            pos, vel, alive, col = art.new_missile(pos, vel, alive, col,
                spread=0.05, brightness=0.2, alive_time=np.random.uniform(2,10))
            
        pos, vel, alive, col = art.propagate_particles(pos, vel, alive, dt=dt, g=-0.1)
        pos, vel, alive, col = art.explode_missile(pos, vel, alive, col,dt, num_stars=20, vel_burst=0.1)
        pos, vel, alive, col = art.trim_dead(pos, vel, alive, col, alive_time_limit=-50)
        matrix_list += [art.render_particles(pos, col, matrix_shape)]
    return matrix_list
        
def bounce_wave(matrix_shape, axis=0, height=2.0, size_factor = 5.0):
    ms = list(matrix_shape[:-1])
    del ms[axis]
    s0 = np.linspace(-1,1,int(size_factor*ms[0]))
    s1 = np.linspace(-1,1,int(size_factor*ms[1]))
    p0,p1 = np.meshgrid(s0,s1)
    pos = -np.ones((int(size_factor**2*np.product(ms)),3))
    ind = [0,1,2]
    del ind[axis]
    pos[:,ind[0]] = p0.flatten()
    pos[:,ind[1]] = p1.flatten()
    
    r = np.sqrt(pos[:,ind[0]]**2 + pos[:,ind[1]]**2)
    s = 0.5
    z = np.exp(-0.5*(r/s)**2)
    plt.plot(r,z,'.')
    pos[:,axis] = height*(z - np.min(z)) -1
    
    vel = np.zeros((int(size_factor**2*np.product(ms)),3))
    vel[:,axis] = np.ones(int(size_factor**2*np.product(ms)))*2/(matrix_shape[axis])
    
    matrix_list = []
    for i in range(matrix_shape[axis]*2):
        pos, vel, _  = art.propagate_particles(pos, vel, dt=1, g=0.0)
        pos, vel = art.wall_bounce(pos, vel)
        col = art.color_points(pos[:,axis], map="rainbow")
        matrix_list += [art.render_particles(pos, col, matrix_shape)]
    return matrix_list

      
def spiral_wave(matrix_shape, axis=0, rotations=1.5, points=500):
    theta = np.linspace(0,rotations*2*np.pi,points)
    r = 1 - 1.0* theta/(rotations*2*np.pi)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    pos = -1*np.ones((points,3))
    ind = [0,1,2]
    del ind[axis]
    pos[:,ind[0]] =x
    pos[:,ind[1]] =y
    pos[:,axis] =r**2
    
    vel = np.zeros((points,3))
    vel[:,axis] = np.ones(points)*2/(matrix_shape[axis])
    
    matrix_list = []
    for i in range(matrix_shape[axis]*2):
        pos, vel, _  = art.propagate_particles(pos, vel, dt=1, g=0.0)
        pos, vel = art.wall_bounce(pos, vel)
        col = art.color_points(pos[:,axis], map="rainbow")
        matrix_list += [art.render_particles(pos, col, matrix_shape)]
    return matrix_list

def particles_in_box(matrix_shape, num_particles=10, duration=300, max_speed=0.1):
    pos = np.zeros((num_particles,3))
    vel = max_speed*np.random.uniform(-1,1,size=(num_particles,3))
    col = (np.random.random( size=(num_particles,3) ) * 256).astype(np.uint8)
    matrix_list = []
    for i in range(duration):
        pos, vel, _  = art.propagate_particles(pos, vel, dt=1, g=0.0)
        pos, vel = art.wall_bounce(pos, vel)
        matrix_list += [art.render_particles(pos, col, matrix_shape)]
    return matrix_list
        
        
def starfield(matrix_shape, duration=200, num_twinkles=150):
    matrix_list = duration*art.generate_clouds(matrix_shape, height=matrix_shape[2], periods=[10,10,10], color=[34, 0, 48], duration=1)
    # matrix_list = [np.zeros(matrix_shape).astype('uint8') for _ in range(duration)]
    for _ in range(num_twinkles):
        inds = [np.random.randint(0,matrix_shape[i]) for i in range(3)]
        twinkle = art.twinkle(matrix_shape, inds, 
                              start=np.random.randint(0,duration), 
                              ramp_up=np.random.uniform(5,50),
                              dwell =np.random.uniform(1,50),
                              ramp_down=np.random.uniform(5,50), 
                              color=[112, 102, 79], duration=duration)
        matrix_list =  art.add_matrixlists(matrix_list,twinkle)
    return matrix_list


def angler_fish(matrix_shape, duration=200, col=[100, 102, 81], a=0.1, dt=0.2):
    fish = art.load_ply(Path.joinpath(Path(__file__).parent ,"angler fish.ply"))
    fish = 5*art.translate_pointcloud(fish,[0,-1,0])
    fish = art.translate_pointcloud(fish,[0,-1.1,0])

    matrix_list = []
    
    t_appear = 0.5
    d_appear = 1.3
    t_dash = 0.9
    d_dash = 7.5
    
    dt_appear = duration*(t_dash- t_appear)
    dt_dash = duration*(1-t_dash)

    speed = np.interp(np.arange(duration),
                      [0, t_appear*duration, (t_appear+1e-3)*duration, t_dash*duration, (t_dash+1e-3)*duration, duration],
                      [0, 0,                  d_appear/dt_appear,       d_appear/dt_appear,  d_dash/dt_dash,        d_dash/dt_dash])
    
    fade = np.interp(np.arange(duration),
                     [0, t_appear*duration,t_dash*duration, duration],
                     [0, 0,              1,                 1])
    
    pos = np.array([0,0,0])
    vel = np.array([0,0,0])
    for s,f in zip(speed,fade):
        fish = art.translate_pointcloud(fish,[0,s,0])
        mf = art.pointcloud_to_matrix(fish, matrix_shape, color=[int(f*255),0,0])
        pos, vel, _  = art.propagate_particles(pos, vel, g=0, dt=dt, a=np.random.uniform(-a,a, size=3))
        point = art.render_particles(pos, col, matrix_shape)
        matrix_list += [art.add_matrices(point,mf)]
    return matrix_list
    
    
    
    
# DONE Fireworks
# TODO pulsing points
# DONE linear wave bounce
# TODO Stars/nebula
# TODO multilightning
# TODO  helix
# TODO review https://github.com/MaltWhiskey/Mega-Cube/blob/master/Software/LED%20Display/src/core/Graphics.h
# TODO review https://github.com/MaltWhiskey/Mega-Cube/tree/master/Software/LED%20Display/src/space