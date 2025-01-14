import board
from digitalio import DigitalInOut, Direction
import time
from adafruit_neopxl8 import NeoPxl8
from adafruit_led_animation.helper import PixelMap
import usb_cdc


import animation
import serial_neopxl_driver

#led = DigitalInOut(board.LED)
#led.direction = Direction.OUTPUT

com = usb_cdc.data


while True:
    print("Hello, CircuitPython!")
    ## test!
    # serial_neopxl_driver.test_driver(
    #     num_strands=8,
    #     rows=12,
    #     perstring = 13,
    #     duration=20
    # )

    # run!
    # print(com)
    # d =com.readline()
    # led.value = True
    # time.sleep(0.5)
    # led.value = False
    # print(d)
    # com.write(d)
    # data = com.readline()
    # com.write(data)
    #led.value = True

    #led.value = False
    # time.sleep(0.1)
    serial_neopxl_driver.echo(com)
    
    serial_neopxl_driver.run_driver_simple(com, name='1')

