import time
from rpi_ws281x import *
import argparse

import numpy as np
import matplotlib.pyplot as plt

import drivers
import art
from pathlib import Path

if __name__ == '__main__':
    cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
    shape = (20,20,20) #(12,12,28)
    matrix_shape = shape +(3,)
    size = [3,3,1.4] # m side lengths of cube
    n_channels= 2
    
    vis = drivers.Visualise(matrix_shape, n_channels, size, fps=30)
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