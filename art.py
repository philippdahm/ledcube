import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator

def generator_zipup(matrix, xy, color):
    matrix[xy[0],xy[1],0, :] = color
    
def step_up(matrix):
    m = np.roll(matrix, 1, axis = 2)
    m[...,0,-3:] = np.zeros_like(m[...,0,-3:])
    m[:,:,0,-3:]
    return m
    
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