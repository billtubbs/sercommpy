"""Python module to facilitate data communication with a connected device
via serial USB connection.

Bill Tubbs
May 2025

"""

import numpy as np
import numba as nb
from itertools import chain


MY_NAME = "HostComputer"
START_MARKER = 254
END_MARKER = 255
SPECIAL_BYTE = 253
MAX_PACKAGE_LEN = 8192


def send_data_to_arduino(ser, data):
    global START_MARKER, END_MARKER
    # Length includes 2 bytes to transmit length value
    #length_bytes = (data.shape[0] + 2).to_bytes(length=2, byteorder='big')
    ser.write(chain.from_iterable([
        [START_MARKER],
        #encode_data(length_bytes),  # TODO: Remove this
        encode_data(data),
        [END_MARKER]
    ]))


def receive_data_from_arduino(ser):
    global START_MARKER, END_MARKER
    # Read data until the start character is found
    bytes_seq = ser.read_until(
        bytes([START_MARKER]), size=MAX_PACKAGE_LEN * 2 + 1
    )
    assert bytes_seq[-1] == START_MARKER, "No start marker found"
    # Read data until the end marker is found
    bytes_seq = ser.read_until(
        bytes([END_MARKER]), size=MAX_PACKAGE_LEN * 2 + 1
    )
    assert bytes_seq[-1] == END_MARKER, \
        f"No end marker found after {MAX_PACKAGE_LEN * 2 + 1} bytes read"
    # Decode and convert to numpy array
    bytes_seq = decode_bytes(bytes_seq[:-1])  # omit end marker
    assert bytes_seq.shape[0] <= MAX_PACKAGE_LEN, \
        f"More than {MAX_PACKAGE_LEN} data bytes in package"
    # n_bytes = int.from_bytes(bytes_seq[0:2], byteorder='big')
    return bytes_seq


@nb.njit()
def encode_data(data: nb.uint8[:]) -> nb.uint8[:]:
    # TODO: Could this be converted to return bytes?
    global SPECIAL_BYTE
    data_out: nb.uint8[:] = []
    for x in data:
        if x >= SPECIAL_BYTE:
            data_out.append(SPECIAL_BYTE)
            data_out.append(x - SPECIAL_BYTE)
        else:
            data_out.append(x)
    return np.array(data_out, dtype='uint8')


@nb.njit()
def decode_bytes(bytes_seq: nb.uint8[:]) -> nb.uint8[:]:
    # TODO: Could this be converted to accept bytes?
    global SPECIAL_BYTE
    data_out: nb.uint8[:] = []
    n = 0
    while n < len(bytes_seq):
        x = bytes_seq[n]
        if x == SPECIAL_BYTE:
            n += 1
            x = SPECIAL_BYTE + bytes_seq[n]
        data_out.append(x)
        n += 1
    return np.array(data_out, dtype='uint8')
