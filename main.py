import time
from rpi_ws281x import *
import argparse

import numpy as np
import matplotlib.pyplot as plt

import drivers
import art
from pathlib import Path
import time


if __name__ == '__main__':
    cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
    shape = (12,12,13) #(12,12,28)
    matrix_shape = shape +(3,)
    size = [0.1,0.1,0.001] # m side lengths of cube
    n_channels= 1
    config = { ## Default Config
                "pins": [18,],
                "freq_hz" : 800000,
                "dma" : 10,
                "PWM channel" : [0],
                "strip type" : None,
                "connection" : 'scan'
            }
    neop = drivers.Neopixel(matrix_shape, n_channels, config=config)
    
    while True:
        matrix = (np.random.random( matrix_shape ) * 256).astype(np.uint8)
        start = time.time()
        neop.display(matrix, method="24bit_single")
        dt = (time.time() - start)
        print(f"{dt*1e3:0.1f}ms \t {1/dt:0.2f}HZ")
        time.sleep(1.0)
    
    
    
    
    
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