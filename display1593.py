from itertools import chain
import numpy as np
from numba import jit, types


# TODO: Need to split into two
NUM_LEDS = 798


COMMAND_LC = np.array(list(b'LC'), dtype="uint8")
COMMAND_SN = np.array(list(b'SN'), dtype="uint8")

B2 = 32
BLACK = np.ones(3, dtype='uint8')
WHITE = np.full_like(BLACK, B2)
RED = np.array([B2, 0, 0], dtype='uint8')
GREEN = np.array([0, B2, 0], dtype='uint8')
BLUE = np.array([0, 0, B2], dtype='uint8')
YELLOW = np.array([B2, B2, 0], dtype='uint8')
MAGENTA = np.array([B2, 0, B2], dtype='uint8')
CYAN = np.array([0, B2, B2], dtype='uint8')


def set_led(i, rgb):
    """Command L1"""
    return np.array((76, 49, i // 256 % 256, i % 256, *rgb), dtype="uint8")


@jit(
    [
        types.uint8[:, :](types.int32[:]), 
        types.uint8[:, :](types.int64[:])
    ], 
    nopython=True
)
def make_idx_array(leds):
    idx = [(i // 256 % 256, i % 256) for i in leds]
    return np.array(idx, dtype="uint8")


def set_leds(leds, rgb_array):
    """Command LN"""
    n = rgb_array.shape[0]
    idx = make_idx_array(leds)
    return np.hstack(
        [
            (76, 78, n // 256 % 256, n % 256),
            np.hstack((idx, rgb_array)).flatten()
        ]
    )


def set_all_leds(rgb_array):
    """Command LA"""
    assert rgb_array.shape[0] == NUM_LEDS
    return np.hstack(
        [
            (76, 65),
            rgb_array.flatten()
        ]
    )


def clear_all_leds():
    """Command LC"""
    return COMMAND_LC


def show_now():
    """Command SN"""
    return COMMAND_SN
