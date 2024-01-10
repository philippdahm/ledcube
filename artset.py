import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import scipy.spatial as spatial
import matplotlib as mpl

import art

def makeset_wave(matrix_shape, frames=100, coloring='hsv'):
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