"""Python module to facilitate data communication with a connected device
via serial USB connection.

Bill Tubbs
May 2025

"""
import time
import numpy as np
import numba as nb
from numba import jit, types
from itertools import chain


MY_NAME = "HostComputer"
START_MARKER = 254
END_MARKER = 255
SPECIAL_BYTE = 253
MAX_PACKAGE_LEN = 8192


# Define the array types
readonly_uint8_array = types.Array(types.uint8, 1, 'C', readonly=True)
writable_uint8_array = types.Array(types.uint8, 1, 'C')



def connect_to_arduino(ser, timeout_time=10, hello_message=b'My name is '):
    # Wait for the initial hello message from the Arduino
    t0 = time.time()
    while (time.time() - t0) < timeout_time:
        if ser.in_waiting > 0:
            data_received = receive_data_from_arduino(ser)
            if np.array_equal(data_received[:2], [0, 0]):
                message_bytes = bytes(data_received[2:])
                assert message_bytes.startswith(hello_message)
                message = message_bytes.removeprefix(
                    hello_message
                ).decode('utf')
                status = 0
                break
            else:
                status, message = 2, "No hello message in data received"
        if (time.time() - t0) >= timeout_time:
            status, message = 1, "Timeout"
    return status, message


def send_data_to_arduino(ser, data):
    global START_MARKER, END_MARKER
    # TODO: Make this non-blocking
    ser.write(chain.from_iterable([
        [START_MARKER],
        encode_data(data.astype(np.uint8)),
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
    # Convert to numpy array and decode, omitting end marker
    data = np.frombuffer(bytes_seq[:-1], dtype=np.uint8)
    data = decode_bytes(data)
    assert data.shape[0] <= MAX_PACKAGE_LEN, \
        f"More than {MAX_PACKAGE_LEN} data bytes in package"
    return data


@jit([writable_uint8_array(readonly_uint8_array), 
      writable_uint8_array(writable_uint8_array)], nopython=True)
def encode_data(data):
    # TODO: Could this be converted to return bytes?
    global SPECIAL_BYTE
    data_out: nb.uint8[:] = []
    for x in data:
        if x >= SPECIAL_BYTE:
            data_out.append(SPECIAL_BYTE)
            data_out.append(x - SPECIAL_BYTE)
        else:
            data_out.append(x)
    return np.array(data_out, dtype=np.uint8)


@jit([writable_uint8_array(readonly_uint8_array), 
      writable_uint8_array(writable_uint8_array)], nopython=True)
def decode_bytes(data_in):
    # TODO: Could this be converted to accept bytes?
    global SPECIAL_BYTE
    data_out: nb.uint8[:] = []
    n = 0
    while n < len(data_in):
        x = data_in[n]
        if x == SPECIAL_BYTE:
            n += 1
            x = SPECIAL_BYTE + data_in[n]
        data_out.append(x)
        n += 1
    return np.array(data_out, dtype=np.uint8)
