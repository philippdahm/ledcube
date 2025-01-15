import time
import usb_cdc
import board
import rainbowio
import adafruit_ticks
import math
from adafruit_led_animation.helper import PixelMap
from adafruit_neopxl8 import NeoPxl8
import random
import digitalio


class Driver_RP2040:
    def __init__(self, num_strands, strand_length, brightness=1, auto_write=False):
        self.strand_length = strand_length
        self.num_pixels = num_strands * strand_length
        self.num_strands = num_strands

        # Make the object to control the pixels
        self.strands = [
            NeoPxl8(
                board.NEOPIXEL0,
                self.strand_length*2,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL2,
                self.strand_length*2,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL4,
                self.strand_length*2,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL6,
                self.strand_length*2,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            )
            ]

    def write_strand(self, channel, data):
        ## set all RGB values of leds (uint8)
        strand = self.strands[int(channel/2)]
        istart = channel%2 * self.strand_length
        #print(f"Channel {channel} PIO {int(channel/2)}:  data{data[0:3]}")
        #print("writing")
        n_data = int(len(data)/3)
        n_strand = len(strand)
        # if n_data > n_strand:
        #     print(f"data ({n_data}) shorter than strand ({n_strand})")
        for i in range(n_data):
            #print(istart+i)
            strand[istart+i] = (data[3*i],data[3*i+1],data[3*i+2])


    def show_all(self):
        for s in self.strands:
            s.show()


def parse_data(d):
    channel = d[0]  #first number is channel index
    return channel, d[1:]


def run_driver(com, name=""):
    
    init_flag = False
    ## infinite loop
    while True:
        ## wait for first configuration command
        d = bytearray(com.readline())

        ## reset init flag if data received doesnt make sense. go back to waiting for init.
        if len(d) <2:
            init_flag = False 
            print(f"waiting for init{name}... {d}")

        ## if not yet initialised and data makes sense: initialise
        elif not init_flag:
            num_strands = int(d[0])
            strand_length =int.from_bytes(d[1:-1], "little")
            #Send ack to host
            driver = Driver_RP2040(num_strands, strand_length)
            print(f"initialised with {num_strands} strands of {strand_length} leds")
            com.write(num_strands.to_bytes(1,1)+strand_length.to_bytes(1,1)+b"\n")
            init_flag = True
            print(f"initialised with {num_strands} strands of {strand_length} leds")


        ## if initialised: read, display then send ack
        else:
            channel, data = parse_data(d)
            driver.write_strand(channel, data)
            
            ## once last channel has been written, display leds
            if channel == (num_strands-1):
                driver.show_all()

            ## send ack
            # print(channel.to_bytes(1,1))
            com.write(channel.to_bytes(1,1)+b"\n")
            print("")

def run_driver_simple(com, name="", num_strands=6, strand_length=12*13):
    com.reset_output_buffer()
    com.reset_input_buffer()
    driver = Driver_RP2040(num_strands, strand_length)
    print(f"simple init{name} with num_strands{num_strands}, strand_length{strand_length}")
    while True:
        d = bytearray(com.readline())
        if len(d)>2:
            channel, data = parse_data(d)
            try:
                driver.write_strand(channel, data)
            except:
                print(f"error writing {channel}, {len(data)}")
            
            ## once last channel has been written, display leds
            if channel == (num_strands-1):
                driver.show_all()
            # time.sleep(0.001)
            ## send ack
            # print(channel.to_bytes(1,1))
            com.write(channel.to_bytes(1,1)+b"\n")
            com.flush()
        else:
            print(f"{name} waiting.. {d}")



def _random_colour(chan, rows, perstring, tc):
    return [random.randint(0,255) for i in range(3*rows*perstring)]

def _random_string(chan, rows, perstring, tc):
    data = []
    for i in range(rows):
        col = [random.randint(0,255) for i in range(3)]
        data += col*perstring
    return data


def _raindrop(chan, rows, perstring, tc):
    data = []
    for i in range(rows*perstring):
        if (i-int(tc*60))%5 == 0:
            data += [100, 100, 255]
        else:
            data += [0,0,0]
    return data
    
def _plane(chan, rows, perstring, tc):
    if (chan-int(tc*40))%5 == 0:
        return [random.randint(0,255) for i in range(3)]*rows*perstring
    else:
        return [0]*3*rows*perstring


def _pulse(chan, rows, perstring, tc):
    b = (math.sin(tc*3.14159*2*5)+1)/2
    return [int(0*b),int(255*b),int(255*b)]*perstring*rows
    
def test_driver(num_strands, rows = 12, perstring = 13, duration =20):
    strand_length = rows * perstring
    driver = Driver_RP2040(num_strands, strand_length)

    animations = {
        'random' : _random_colour,
        'string' : _random_string,
        'raindrop': _raindrop,
        'plane': _plane,
        'pulse': _pulse,
        }
    while True:
        for name, ani in animations.items():
            ti = time.monotonic()
            print(name)
            tc = 0
            while tc < 1 :
                tc = (time.monotonic()-ti)/duration
                for i in range(num_strands):
                    data = ani(i, rows, perstring, tc)
                    driver.write_strand(i, data)
                driver.show_all()
                time.sleep(0.0001)



def echo(com):
    while True:
        d = com.readline()
        com.write(d)

def echo_pro(com):
    while True:
        d = com.readline()
        ch,da = parse_data(d)
        com.write(ch.to_bytes(1,1)+d)
