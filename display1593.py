from itertools import chain
import numpy as np
import numba as nb


# TODO: Need to split into two
NUM_LEDS = 798


COMMAND_LC = np.array(list(b'LC'), dtype="uint8")
COMMAND_SN = np.array(list(b'SN'), dtype="uint8")


def set_led(i, rgb):
    """Command L1"""
    return np.array((76, 49, i // 256 % 256, i % 256, *rgb), dtype="uint8")


@nb.jit
def make_idx_array(leds: nb.int64[:]):
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
