import board
import digitalio
import time
from adafruit_neopxl8 import NeoPxl8
from adafruit_led_animation.helper import PixelMap
import usb_cdc


import animation
import serial_neopxl_driver

#led = digitalio.DigitalInOut(board.LED)
#led.direction = digitalio.Direction.OUTPUT

com = usb_cdc.data

while True:
    print("Hello, CircuitPython!")
    #led.value = True
    #time.sleep(5)
    #led.value = False
    #time.sleep(0.1)

    ## test!
    # serial_neopxl_driver.test_driver(
    #     num_strands=8,
    #     rows=12,
    #     perstring = 13,
    #     duration=20
    # )

    # run!
    # com = usb_cdc.data
    # print(com)
    # d =com.readline()
    # led.value = True
    # time.sleep(0.5)
    # led.value = False
    # print(d)
    # com.write(b"hello\n")
    # time.sleep(0.5)
    serial_neopxl_driver.run_driver(com, name='2')

