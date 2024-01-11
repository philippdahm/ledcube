import time
from rpi_ws281x import *
import argparse

import numpy as np
import matplotlib.pyplot as plt

import drivers
import art
import artset
from pathlib import Path
import time



        
        

# if __name__ == '__main__':

    
#     cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
#     shape = (12,2,13) #(12,12,28)
#     matrix_shape = shape +(3,)
#     size = [0.1,0.1,0.001] # m side lengths of cube
#     n_channels= 1
#     config = { ## Default Config
#                 "pins": [18],
#                 "freq_hz" : 800000,
#                 "dma" : 10,
#                 "PWM channel" : [0],
#                 "strip type" : None,
#                 "connection" : 'scan'
#             }
#     neop = drivers.Neopixel(matrix_shape, n_channels, config=config)
#     # neop = drivers.Visualise(matrix_shape, n_channels, size)
    
#     matlist = art.test_matrix(matrix_shape)
#     while True:
#         neop.animate(matlist, wait_ms=25, method="24bit_single")
        
#     while True:
#         matrix = (np.random.random( matrix_shape ) * 256).astype(np.uint8)
#         start = time.time()
#         neop.display(matrix, method="24bit_single")
#         dt = (time.time() - start)
#         print(f"{dt*1e3:0.1f}ms \t {1/dt:0.2f}HZ")
#         time.sleep(1.0)
    
    
    
    
    
if __name__ == '__main__':
    cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
    shape = (20,20,20) #(12,12,28)
    matrix_shape = shape +(3,)
    size = [3,3,1.4] # m side lengths of cube
    n_channels= 2
    
    vis = drivers.Visualise(matrix_shape, n_channels, size, fps=20)
    
    
    matrix_list = artset.fireworks(matrix_shape)
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test_fireworks.gif") ,dpi = 100)

    matrix_list = artset.underwater(matrix_shape, duration=100)
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test_underwater.gif") )

    matrix_list = artset.face_sweep(matrix_shape)
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test_head.gif") )

    matrix_list = artset.thunderstorm(matrix_shape,lighting_freq=0.5, duration=100)
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test_thunderstorm.gif") )
    
    if 0: # Cube
        cube = 0.65*art.make_pointcloud_cube_wireframe()
        cuber = art.rotate_pointcloud(cube, 45,20,5)
        # cubec = art.translate_pointcloud()
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
        
        vis.save_animated(matrix_list, Path.joinpath(cdir,"test_cube.gif") )

        vis.display(matrix)
        cubec = art.translate_pointcloud(cuber, [0.2,0,0])
    
    matrix = np.zeros(matrix_shape)
    
    matrix = matrix.astype(np.uint8)
    # matrix_list = []
    # for i in range(int(3*vis.fps)):
    #     # print(matrix[0,0,:, 1])
    #     matrix_list += [matrix]
    #     # vis.display_raw_static(matrix)
    #     matrix = art.propagate_zipup(matrix)
    #     if i %3 ==0:
    #         matrix = art.seed_random_bottom(matrix, color='random')
    
    slowdown = 0.5
    matrix_list = art.makeset_wave(matrix_shape, frames=int(vis.fps/slowdown), coloring=[255,0,255])

    a = vis.animate(matrix_list)
    # plt.show()
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test_wave2.gif") )
    
    matrix_list2 = art.supersample_antialisaing(matrix_shape, 10, art.makeset_wave, frames=int(vis.fps/slowdown), coloring=[255,0,255])
    a = vis.animate(matrix_list2)
    # plt.show()
    vis.save_animated(matrix_list2, Path.joinpath(cdir,"test_wave_supersampled.gif") )
    
    
    ### TODO: preprocess frames & load
    # TODO: blend mutliple sets together (blend function)