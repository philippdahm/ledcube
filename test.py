import time
# from rpi_ws281x import *
import argparse

import numpy as np
import matplotlib.pyplot as plt

import drivers
import art
import artset
from pathlib import Path
import time

import serial
import numpy as np

def test_echo(serial_addr, N=1000, size=1*12*12*3):
    print(f"init ports {serial_addr}")
    ports = [serial.Serial(addr,
                baudrate=115200, ## 115200 #460800
                bytesize=8,
                timeout=2,
                ) for addr in serial_addr]
    
    data = np.random.random_integers(0,255,size=size).astype("uint8")
    result = {a:[] for a in serial_addr}
    print(f"testing ports ...")
    for i in range(1000):
        for addr, port in zip(serial_addr, ports):
            port.write(data.tobytes()+b"\n") 
            d = port.readline()[:-1]
            resp = np.frombuffer(d, dtype="uint8")
            result[addr] += [resp != data]   

    for k,v in result.items():
        print(f"{k} : {np.sum(v)/N/size:0.1e}")
    print(result)
            
        
        


        
        

if __name__ == '__main__':
    test_echo(['/dev/ttyACM1','/dev/ttyACM3'])

    # LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53

    cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
    shape = (12,12,13) #(12,12,28)
    matrix_shape = shape +(3,)
    size = [0.1,0.1,0.001] # m side lengths of cube
    config = { ## Default Config
                "pins": [18, 21, 12], #10
                "freq_hz" : 800000,
                "dma" : 10,
                "PWM channel" : [0,0,0],
                "strip type" : None,
                "connection" : 'scan'
            }
    
    n_channels= len(config["pins"])
    neop = drivers.NeopixelRpi(matrix_shape, n_channels, config=config)
    # neop = drivers.Visualise(matrix_shape, n_channels, size)
    
    matlist = art.test_matrix(matrix_shape, n_channels=n_channels)

    print(f"running test with {n_channels} channels on pins: {config['pins']}")
    while True:
        neop.animate(matlist, wait_ms=25, method="24bit_single")



#     while True:
#         matrix = (np.random.random( matrix_shape ) * 256).astype(np.uint8)
#         start = time.time()
#         neop.display(matrix, method="24bit_single")
#         dt = (time.time() - start)
#         print(f"{dt*1e3:0.1f}ms \t {1/dt:0.2f}HZ")
#         time.sleep(1.0)
