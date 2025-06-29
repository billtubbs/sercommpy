from itertools import chain
import numpy as np
from numba import jit, types


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


def clear_all_leds():
    """Command LC"""
    return COMMAND_LC


def show_now():
    """Command SN"""
    return COMMAND_SN
