import serial


com = serial.Serial("/dev/ttyACM1",
            baudrate=19200,
            bytesize=8,
            timeout=5,
            )

while True:
    com.write(b"Hello Ima  Pi\n")
    data = com.readline()
    print(data)





