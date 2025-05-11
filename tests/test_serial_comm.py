import numpy as np
from serial_comm.serial_comm import encode_data, decode_bytes


def test_serial_comm():
    data = [0, 64, 65, 253, 254, 255]
    encoded_data = encode_data(data)
    decoded_data = decode_bytes(encoded_data)
    assert np.array_equal(decoded_data, data)
