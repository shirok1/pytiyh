import serial

do_log = True


def check_command_response(data, name):
    if data[1] == 0x02:
        if do_log:
            print(name + " success! " + str(data))
    elif data[1] == 0xEE:
        if do_log:
            print(name + " failed! " + str(data))
    else:
        if do_log:
            print(name + " received unknown response: " + str(data))


class Emm:
    def __init__(self, port):
        self.port = port
        self.ser = serial.Serial(port, 38400, timeout=1)

    def read_encoder(self):
        self.ser.write(b'\x01\x30\x6B')
        data = self.ser.read(4)
        return int.from_bytes(data[1:3], byteorder='big', signed=False)

    def read_pulse(self):
        self.ser.write(b'\x01\x33\x6B')
        data = self.ser.read(6)
        return int.from_bytes(data[1:5], byteorder='big', signed=False)

    def read_position(self):
        self.ser.write(b'\x01\x36\x6B')
        data = self.ser.read(6)
        return int.from_bytes(data[1:5], byteorder='big', signed=False)

    def read_angle(self):
        return self.read_position() * 360 / 65536

    # def read_delta(self):
    #     pass

    def read_enable(self):
        self.ser.write(b'\x01\x3A\x6B')
        data = self.ser.read(3)
        return bool(data[1])

    def read_stuck(self):
        self.ser.write(b'\x01\x3E\x6B')
        data = self.ser.read(3)
        return bool(data[1])

    def read_auto_zero(self):
        self.ser.write(b'\x01\x3F\x6B')
        data = self.ser.read(3)
        return bool(data[1])

    def modify_step(self, step: int):
        self.ser.write(b'\x01\x84' + step.to_bytes(1, "big") + b'\x6B')
        data = self.ser.read(3)
        check_command_response(data, "modify_step")

    def modify_serial_address(self, address: int):
        self.ser.write(b'\x01\xAE' + address.to_bytes(1, "big") + b'\x6B')
        data = self.ser.read(3)
        check_command_response(data, "modify_serial_address")

    def set_enable(self, enable: bool):
        self.ser.write(b'\x01\xF3' + int(enable).to_bytes(1, "big") + b'\x6B')
        data = self.ser.read(3)
        check_command_response(data, "set_enable")

    def set_speed_mode(self, ccw: bool, speed: int, acc: int):
        ccw_speed_compound = (ccw << 12) + speed & 0x04FF
        self.ser.write(b'\x01\xF6'
                       + ccw_speed_compound.to_bytes(2, "big")
                       + acc.to_bytes(1, "big") + b'\x6B')
        data = self.ser.read(3)
        check_command_response(data, "set_speed_mode")

    def save_speed_mode(self):
        self.ser.write(b'\x01\xFF\xC8\x6B')
        data = self.ser.read(3)
        check_command_response(data, "save_speed_mode")

    def clear_speed_mode(self):
        self.ser.write(b'\x01\xFF\xCA\x6B')
        data = self.ser.read(3)
        check_command_response(data, "clear_speed_mode")

    def set_position_mode(self, ccw: bool, speed: int, acc: int, pulse: int):
        ccw_speed_compound = (ccw << 12) + speed & 0x04FF
        self.ser.write(b'\x01\xFD'
                       + ccw_speed_compound.to_bytes(2, "big")
                       + acc.to_bytes(1, "big")
                       + pulse.to_bytes(3, "big") + b'\x6B')
        data = self.ser.read(3)
        check_command_response(data, "set_position_mode")


if __name__ == '__main__':
    emm = Emm('/dev/ttyS0')
    print(emm.read_encoder())
    print(emm.read_pulse())
    print(emm.read_position())
    print(emm.read_angle())
    print(emm.read_enable())
    print(emm.read_stuck())
    print(emm.read_auto_zero())
