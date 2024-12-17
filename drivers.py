

from webcolors import rgb_to_hex

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import prod
        
import json
from pathlib import Path
import time




class Driver:
    def __init__(self, matrix_shape:Tuple, n_channels:int, **kwargs):
    ## Matrix shape: 3 array following: x strings, y strings, LED in string, colour (RGB)
    ## output_channel_indices: list of channels, each channel has an xy array of indices to which each string connects to. nan means no connection
    ## physical_size: list of 3 physical dimensions [m] in x (rows), y (columns) z (string height)
            
        self.matrix_shape = matrix_shape
        self.n_channels = n_channels

        
    
    def display(self, matrix):
        if matrix.shape != self.matrix_shape:
            raise ValueError("Input matrix does not match setup")
        pass
    
    def _connect_rows(self, rows, connection="scan"):
        if connection == 'scan':
            return np.vstack(rows) # then stacks the rows (x)
        
        elif connection == 'snake':
            rows[1::2,:,:,:] = rows[1::2,::-1,:,:] # Reverses the column (y) order of every second row
            return np.vstack(rows)  # then stacks the rows (x)
        
        else:
            raise KeyError(f"Connection type '{connection}' not defined")
        
    def mat2channels(self, matrix, connection="scan"):
        if matrix.shape[0]%self.n_channels != 0:
            raise ValueError(f"number of rows ({matrix.shape[0]}) does not divide nicely into number of channels ({self.n_channels})")
    
        # new_shape =  (matrix.shape[0], matrix.shape[1]*matrix.shape[2], matrix.shape[3]) #stack strings (z) within columns (y) direction 
        # collapsed_rowstrings = np.reshape(matrix, new_shape)
        
        row_sets = np.split(matrix, self.n_channels, axis=0) # split into groups of rows
        
        channels = []
        for rows in row_sets:
            collapsed_rows = self._connect_rows(rows, connection=connection) ## Connects the various rows (x) together (collapsed onto xy)
            channels += [np.vstack(collapsed_rows)] ## stacks each string (z) in series of each row-colum point(xy)

        return channels

    
class Visualise(Driver):
    def __init__(self, matrix_shape, n_channels, physical_size, fps=1, **kwargs):
        super().__init__(matrix_shape, n_channels, **kwargs)
        
        self.physical_size = physical_size
        
        self.fig=None
        self.ax=None
        self.fps=fps
        
        x,y,z = np.indices(matrix_shape[:-1])
        self.x = x.flatten()*physical_size[0]/np.max(x)
        self.y = y.flatten()*physical_size[1]/np.max(y)
        self.z = z.flatten()*physical_size[2]/np.max(z)

    def display(self, matrix):
        self.display_raw_static(matrix)
    
    def _setup(self):
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.fig.tight_layout()
        self.ax.xaxis.set_pane_color('k')
        self.ax.yaxis.set_pane_color('k')
        self.ax.zaxis.set_pane_color('k')
        
        
    def _update(self,matrix):
        colors = np.apply_along_axis(rgb_to_hex, -1, matrix)
        # self.ax.clear()
        return self.ax.scatter(self.x, self.y, self.z, marker=',', c=colors.flatten())
        
    
    def display_raw_static(self,matrix):
        if type(self.fig)==type(None):
            self._setup()
        self._update(matrix)
        plt.show()
        
    def animate(self, matrix_list, **kwargs):
        self._setup()
        scat= self._update(matrix_list[0])
        
        def animate(i):
            colors = np.apply_along_axis(rgb_to_hex, -1, matrix_list[i])
            scat.set(color=colors.flatten())
            return scat,
        
        return animation.FuncAnimation(self.fig, animate, interval=1000/self.fps, blit=True, frames=len(matrix_list))
        
    def save_animated(self, matrix_list, save_dir="", bitrate=50, dpi=100):
        ani =self.animate( matrix_list)
        writer = animation.PillowWriter(fps=self.fps, bitrate=bitrate)
        ani.save(save_dir, writer=writer, dpi=dpi)
            
        
        

class NeopixelRpi(Driver):
    def __init__(self, matrix_shape, n_channels, config=None, **kwargs):
        import rpi_ws281x
        import _rpi_ws281x as ws
        super().__init__(matrix_shape, n_channels, **kwargs)
        
        if type(config)==type(None):
            config = { ## Default Config
                "pins": [18],
                "freq_hz" : 800000,
                "dma" : 10,
                "PWM channel" : [0],
                "strip type" : None,
                "connection" : 'scan'
            }
        if len(config["pins"]) != n_channels:
            raise ValueError("Config doesnt have same number of channels as pins")
        if len(config["PWM channel"]) != n_channels:
            raise ValueError("Config doesnt have same number of channels as PWMs")
        
        self.config = config
        
        n_leds = int(prod(self.matrix_shape[:-1])/n_channels)
        self.strips = [rpi_ws281x.PixelStrip(
                num= n_leds,
                pin=self.config["pins"][i],
                freq_hz=self.config["freq_hz"],
                dma=self.config["dma"],
                channel=self.config["PWM channel"][i],
                strip_type = self.config["strip type"]
            ) for i in range(self.n_channels)]
        
        self.check_flag = False
            
    def rgb_to_24bit(self, matrix):
        ## Using rpi_ws281x.py line 14 to cast 3 by 255int into 24bit colour
        ## ... should mean arbitrary front dimensions, last needs to be 3 (RGB)
        m = matrix.astype('uint32') ## Need to cast to bigger dtype otherwise overflows!
        return (m[...,0] << 16) | (m[...,1] << 8) | m[...,2]
        
    def _check_display(self,matrix):
        channels = self.mat2channels(matrix, connection=self.config['connection'])
        if len(channels) != len(self.strips):
            raise ValueError(f"Differnt number of channels and LED strips")
        for n,c in enumerate(channels):
            if len(self.strips[n]) != len(c):
                raise RuntimeError(f"Channel {n} has a missmatch in number of LEDs on allocated strip")
        for strip in self.strips:
            strip.begin()
            
    def scale_brightness_to_perceived(self, matrix, inverse=False):
        if inverse:
            return (matrix * np.log10(1+matrix/255*9)).astype('uint8')
        else:
            return (matrix * (1-np.exp(-matrix*0.02))).astype('uint8')

        
    def display(self,matrix, method="color_single"):
        matrix = self.scale_brightness_to_perceived(matrix)
        if not self.check_flag:
            self._check_display(matrix)
            self.check_flag = True
        if method=="color_single": # Works. 84ms, spike to 153ms
            self._display_color_single(np.flip(matrix, axis=2))
        elif method=="24bit_single": # Doesnt work,  color needs to be some overwritten type 'uint32_t'
            self._display_24bit_single(np.flip(matrix, axis=2))
        elif method=="color_array": # Doesnt work. deeper in C++ code :(
            self._display_color_array(np.flip(matrix, axis=2))
        elif method=="24bit_array": # Doesnt work. deeper in C++ code :(
            self._display_24bit_array(np.flip(matrix, axis=2))
        else:
            raise KeyError(f"Method {method} not defined")
        
    def _display_color_single(self, matrix):
        channels = self.mat2channels(matrix, connection=self.config['connection'])
        for strip, channel in zip(self.strips, channels):
            for n, rgb in enumerate(channel):
                strip.setPixelColorRGB(n, *rgb)  #rgb[0], rgb[1], rgb[2])
            strip.show()
                
    def _display_24bit_single(self, matrix):
        channels =  self.mat2channels(matrix, connection=self.config['connection'])
        for strip, channel in zip(self.strips, channels):
            # for n, c24b in enumerate(self.rgb_to_24bit(channel)):
            #     strip.setPixelColor(n, c24b)
            for n in range(channel.shape[0]):
                strip.setPixelColorRGB(n, channel[n,0],channel[n,1],channel[n,2])
            strip.show()
            
    def _display_color_array(self, matrix):
        channels = self.mat2channels(matrix, connection=self.config['connection'])
        for strip, channel in zip(self.strips, channels):
            ## attempting to set values directly like line 155 rpi_ws2811x.py
            colors = [rpi_ws281x.Color(channel[i,0],channel[i,1], channel[i,2]) for i in range(channel.shape[0])]
            strip._leds =  colors
            strip.show()
            
    def _display_24bit_array(self, matrix):
        channels =  self.mat2channels(matrix, connection=self.config['connection'])     
        for strip, channel in zip(self.strips, channels):
            ## attempting to set values directly like line 155 rpi_ws2811x.py
            strip._leds =  self.rgb_to_24bit(channel)
            strip.show()
            
    def animate(self, matrix_list, wait_ms=50, method="color_single"):
        for m in matrix_list:
            self.display(m,method=method)
            time.sleep(wait_ms/1000.0)
    

class NeopixelSerial(Driver):
    def __init__(self, matrix_shape, serial_list, config=None, **kwargs):
        import serial

        self.serial_list=serial_list
        self.n_channels = len(serial_list)

        self.serial_addr = np.unique(serial_list)
        self.ports = [serial.Serial(chan,
                               baudrate=19200,
                               bytesize=8,
                               timeout=5,
                               ) for chan in self.serial_addr]
        
        self.write_config(matrix_shape)
        

    def write_config(self, matrix_shape):
        for port in self.ports:
            print(f"configuring port {port.port}...")
            nch = int(self.serial_list.count(port.port))
            leds_per_ch = int(np.sum(matrix_shape[:-1])/self.n_channels)
            
            ack_flag = False
            while not ack_flag:
                port.write(nch.to_bytes()+leds_per_ch.to_bytes()+b'\n')
                response = bytearray(port.readline())
                if len(response)>=2:
                    if response[0]==nch and response[1]==leds_per_ch:
                        print(f"configuration succesful")
                        ack_flag = True
                    else:
                        print(f"wrong response... {response}")
                else:
                    print(f"No response, retrying... {response}")
        
        
    def write_channel(self, data, channel, channel_per_serial=8):
        port_index = np.where(self.serial_list[channel]==self.serial_addr)[0][0]
        ## Can make this a general map later rather than assume these are in order
        channel_on_board = channel%channel_per_serial 
        self.ports[port_index].write(channel_on_board.to_bytes() + data.tobytes() + b"\n")

    def read_channel(self, port, size):  ## TODO: code to go onto SCORPIO
        d = bytearray(port.readline())
        # d = bytearray(int(7).to_bytes() + data.tobytes() + b"\n") ## Test String
        channel = d[0]
        del d[-1]  ## may not nead to do if readline removes eol char
        return np.frombuffer(d, dtype="uint8", offset=1).reshape(size)

    def close(self):
        for p in self.ports:
            p.close()

    def display(self, matrix):
        ## break matrix into channels
        channels = self.mat2channels(matrix)

        ## write each channel to serial
        for c,data in enumerate(channels):
            self.write_channel(data, c)


    def animate(self, matrix_list, wait_ms=0, **kwargs):
        for m in matrix_list:
            self.display(m)
            time.sleep(wait_ms/1000.0)




if __name__ == '__main__':
    cdir = Path(__file__).parent #"/home/tycho/Documents/art/ledcube/"
    
    shape = (4,3,5) #(12,12,28)
    matrix_shape = shape +(3,)
    size = [3,3,1.4] # m side lengths of cube
    n_channels= 2
    
    vis = Visualise(matrix_shape, n_channels, size, fps=5)
    
    with open( Path.joinpath(cdir, 'default_driver_config.json'), 'r') as f:
        config = json.load(f)

    neop = NeopixelRpi(matrix_shape, n_channels, config=config)
    
    # with open(Path.joinpath(cdir, 'default_driver_config.json'), 'w') as f:
    #     json.dump(neop.config, f)
        
    matrix_list = []
    matrix = (np.random.random( matrix_shape ) * 256).astype(np.uint8)
    
    scan_channels = vis.mat2channels(matrix, connection='scan')
    snake_channels = vis.mat2channels(matrix, connection='snake')
    
    for i in range(10):
        matrix = (0.9*matrix).astype(np.uint8)
        matrix_list += [matrix]
        # vis.display_raw_static(matrix)
        neop.display(matrix, method="24bit_array")
    
    a = vis.animate(matrix_list)
    plt.show()
    vis.save_animated(matrix_list, Path.joinpath(cdir,"test.gif") )
    

    
