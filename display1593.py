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

# LED setup
NUMBER_OF_LEDS = {
    'TEENSY1': 798,
    'TEENSY2': 795
}


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


def _board_leds(leds, rgb_array, led_idx):
    """Filter led ids into separate lists for each board."""
    board_leds = [[], []]
    rgb_arrays = [[], []]
    for i, led in enumerate(leds):
        if led < led_idx[0]:
            board_leds[0].append(led)
            rgb_arrays[0].append(rgb_array[i])
        elif led < led_idx[1]:
            board_leds[1].append(led - led_idx[0])
            rgb_arrays[1].append(rgb_array[i])
        else:
            raise ValueError("invalid led id")
    for i, arrays in enumerate(rgb_arrays):
        if len(arrays) > 0:
            rgb_arrays[i] = np.stack(arrays)
        else:
            rgb_arrays[i] = np.empty((0, 3), dtype='uint8')
    return board_leds, rgb_arrays


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
        number_of_leds=NUMBER_OF_LEDS
    ):
        self.ports = ports
        self.baud_rate = baud_rate
        self.board_names = list(number_of_leds.keys())
        self.leds_per_board = list(number_of_leds.values())
        self.first_led_of_strip = list(np.cumsum([0] + self.leds_per_board))
        self.led_idx = list(np.cumsum(self.leds_per_board))
        self._connections = []

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

        if set(connections.keys()) != set(self.board_names):
            raise ValueError(
                f"board name mismatch, expected {self.board_names}, "
                f"got {list(connections.keys())}"
            )

        # Store connections in same order as expected board names
        self._connections = []
        for name in self.board_names:        
            self._connections.append(connections[name])

    def clear_all(self):
        # Command LC
        for ser in self._connections:
            send_data_to_arduino(ser, COMMAND_LC)
        logger.info(f'Method clear_all called.')

    def set_led(self, i, rgb):
        if i < 0:
            raise ValueError("invalid led id")
        assert len(rgb) == 3
        if i < self.led_idx[0]:
            led_id = i
            conn = self._connections[0]
        elif i < self.led_idx[1]:
            led_id = i - self.led_idx[0]
            conn = self._connections[1]
        else:
            raise ValueError("invalid led id")
        # Command L1
        cmd = np.array(
            (76, 49, led_id // 256 % 256, led_id % 256, *rgb), dtype=np.uint8
        )
        send_data_to_arduino(conn, cmd)
        logger.info(f'Method set_led called.')

    def set_leds(self, leds, rgb_array):
        assert rgb_array.shape[1] == 3
        board_leds, rgb_arrays = _board_leds(leds, rgb_array, self.led_idx)
        for leds, rgb_array, conn in zip(board_leds, rgb_arrays, self._connections):
            if len(leds) == 0:
                continue
            n = rgb_array.shape[0]
            idx = make_idx_array(np.array(leds, dtype='int32'))
            # Command LN
            cmd = np.concatenate(
                [
                    (76, 78, n // 256 % 256, n % 256),
                    np.hstack((idx, rgb_array)).flatten()
                ]
            ).astype(np.uint8)
            send_data_to_arduino(conn, cmd)
        logger.info(f'Method set_leds called with {n} leds.')

    def show_now(self):
        # Command SN
        # TODO: In future this will be synchronized by comms between boards
        for ser in self._connections:
            send_data_to_arduino(ser, COMMAND_SN)
        logger.info(f'Method show_now called.')

    def disconnect(self):
        while len(self._connections) > 0:
            conn = self._connections.pop()
            conn.close()
            logger.info(f'Closed connection to {conn.port}.')

    def __enter__(self):
        """Enter context manager method"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager method"""
        self.disconnect()
        return False
