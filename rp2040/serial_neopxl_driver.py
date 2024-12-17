import usb_cdc
import board
import rainbowio
import adafruit_ticks
from adafruit_led_animation.helper import PixelMap
from adafruit_neopxl8 import NeoPxl8



class Driver_RP2040:
    def __init__(self, num_strands, strand_length, brightness=1, auto_write=False):
        self.strand_length = strand_length
        self.num_pixels = num_strands * strand_length
        self.num_strands = num_strands

        # Make the object to control the pixels
        self.strands = [
            NeoPxl8(
                board.NEOPIXEL0,
                self.strand_length,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL2,
                self.strand_length,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL4,
                self.strand_length,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            ),
            NeoPxl8(
                board.NEOPIXEL6,
                self.strand_length,
                num_strands=2,
                auto_write=auto_write,
                brightness=brightness,
            )
            ]
        
    def write_strand(self, channel, data):
        ## set all RGB values of leds (uint8)
        strand = self.strands[int(channel/2)]
        istart = channel%2 * self.strand_length
        for i in range(int(len(data)/3)):
            strand[istart+i] = (data[3*i],data[3*i+1],data[3*i+2])

    def show_all(self):
        for s in self.strands:
            s.show()


def read_channel(com):
    d = bytearray(com.readline())
    print(d)
    channel = d[0]  #first number is channel index
    return channel, d[1:]


def run_driver(com):
    ## wait for first configuration command
    init_flag = False
    while not init_flag:
        d = bytearray(com.readline())
        if len(d) >=2:
            num_strands = d[0]
            strand_length =d[1]
            #Send ack to host
            com.write(num_strands.to_bytes(1,1)+strand_length.to_bytes(1,1)+b"\n")
            driver = Driver_RP2040(num_strands, strand_length)
            init_flag = True
        else:
            print(f"waiting for init... {d}")
        

    ## infinite loop to read and display data
    while True:
        channel, data = read_channel(com)

        driver.write_strand(channel, data)
        
        ## once last channel has been written, display leds
        if channel == (num_strands-1):
            driver.show_all()

