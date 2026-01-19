import os
import logging
import serial
from itertools import cycle, chain
from collections import deque

import numpy as np
from numba import jit, types

from serial_comm.serial_comm import (
    connect_to_arduino, send_data_to_arduino, receive_data_from_arduino
)

# Set up logging
logger = logging.getLogger(__name__)
LOG_FORMAT = '%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s'
filename = os.path.basename(__file__)
os.path.splitext(os.path.basename(__file__))
logging.basicConfig(
    filename=os.path.splitext(filename)[0] + '.log',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format=LOG_FORMAT
)

# TODO: Need to split into two
NUM_LEDS = 798

COMMAND_LC = np.array(list(b'LC'), dtype=np.uint8)
COMMAND_SN = np.array(list(b'SN'), dtype=np.uint8)

B2 = 32
BLACK = np.ones(3, dtype='uint8')
WHITE = np.full_like(BLACK, B2)
RED = np.array([B2, 0, 0], dtype='uint8')
GREEN = np.array([0, B2, 0], dtype='uint8')
BLUE = np.array([0, 0, B2], dtype='uint8')
YELLOW = np.array([B2, B2, 0], dtype='uint8')
MAGENTA = np.array([B2, 0, B2], dtype='uint8')
CYAN = np.array([0, B2, B2], dtype='uint8')

# Arduino communication
BAUD_RATE = 57600

# Serial ports of Teensy devices
# Find these by running ls /dev/tty.* from command line
# SERIAL_PORTS = {
#     49: '/dev/cu.usbmodem12745401',
#     50: '/dev/cu.usbmodem6862001'
# }
# Usually, 
#  - TEENSY1 is on usb port 1275401
#  - TEENSY2 is on usb port 6862001
# Raspberry Pi uses the /dev/ttyACM* naming scheme 
# Find these by running ls /dev/tty.* from command line
SERIAL_PORTS = [
    '/dev/ttyACM0',
    '/dev/ttyACM1'
]
# Usually (but not always),
#  - TEENSY1 is on usb port '/dev/ttyACM1'
#  - TEENSY2 is on usb port '/dev/ttyACM0'


# LEDs per strip
LEDS_PER_STRIP = {
    'TEENSY1': [100, 100, 98, 100, 100, 100, 100, 100],
    'TEENSY2': [99, 99, 99, 100, 100, 100, 100, 98]
}


def set_led(i, rgb):
    """Command L1"""
    assert len(rgb) == 3
    return np.array((76, 49, i // 256 % 256, i % 256, *rgb), dtype=np.uint8)


@jit(
    [
        types.uint8[:, :](types.int32[:]), 
        types.uint8[:, :](types.int64[:])
    ], 
    nopython=True
)
def make_idx_array(leds):
    idx = np.empty((leds.shape[0], 2), dtype=np.uint8)
    for i in range(leds.shape[0]):
        idx[i, 0] = leds[i] // 256 % 256
        idx[i, 1] = leds[i] % 256
    #idx = np.array([(i // 256 % 256, i % 256) for i in leds], dtype=np.uint8)
    return idx


def set_leds(leds, rgb_array):
    """Command LN"""
    n = rgb_array.shape[0]
    assert rgb_array.shape[1] == 3
    idx = make_idx_array(leds)
    return np.concatenate(
        [
            (76, 78, n // 256 % 256, n % 256),
            np.hstack((idx, rgb_array)).flatten()
        ]
    ).astype(np.uint8)


def set_all_leds(rgb_array):
    """Command LA"""
    assert rgb_array.shape[0] == NUM_LEDS
    assert rgb_array.shape[1] == 3
    return np.concatenate(
        [
            (76, 65),
            rgb_array.flatten()
        ]
    ).astype(np.uint8)


def set_leds_one_colour(leds, rgb):
    """Command CN"""
    n = leds.shape[0]
    assert len(rgb) == 3
    idx = make_idx_array(leds)
    return np.concatenate(
        [
            (67, 78, n // 256 % 256, n % 256, *rgb),
            idx.flatten()
        ]
    ).astype(np.uint8)


def set_all_leds_one_colour(rgb):
    """Command CA"""
    assert len(rgb) == 3
    return np.array((67, 65, *rgb), dtype=np.uint8)


class Display1593():

    def __init__(
        self, 
        ports=SERIAL_PORTS, 
        baud_rate=BAUD_RATE, 
        leds_per_strip=LEDS_PER_STRIP
    ):
        self.ports = ports
        self.baud_rate = baud_rate
        self.first_led_of_strip = {
            name: np.cumsum([0] + leds) for name, leds in leds_per_strip.items()
        }
        self.max_leds_per_strip = max(chain.from_iterable(leds_per_strip.values()))
        self.leds_per_strip = {
            name: np.array(leds) for name, leds in leds_per_strip.items()
        }

    def connect(self):
        connections = {}
        for port in self.ports:
            conn = serial.Serial(port, baudrate=self.baud_rate)
            logger.info(f'Connected to port {port}.')

            status, message = connect_to_arduino(conn)
            if status == 0:
                worker_name = message
            else:
                raise Exception(message)
            logger.info(f"Hello from: {worker_name}")
            connections[worker_name] = conn

        self.connections = connections

    def clear_all(self):
        for name, ser in self.connections.items():
            send_data_to_arduino(ser, COMMAND_LC)
        logger.info(f'clear_all called.')
    
    def show_now(self):
        for name, ser in self.connections.items():
            send_data_to_arduino(ser, COMMAND_SN)
        logger.info(f'show_now called.')

    def disconnect(self):
        for name, conn in self.connections.items():
            conn.close()
            logger.info(f'Closed connection to {name}.')

    def __enter__(self):
        """Enter context manager method"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager method"""
        self.disconnect()
        return False
