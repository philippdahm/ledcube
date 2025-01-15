import usb_cdc
usb_cdc.enable(console=True, data=True)
usb_cdc.data.timeout = 2.0 # set data in timeout in seconds
usb_cdc.data.write_timeout = 2.0 # set data out timeout in seconds


