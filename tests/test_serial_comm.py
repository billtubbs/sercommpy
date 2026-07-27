import numpy as np

from serial_comm.serial_comm import decode_data, encode_data


def test_serial_comm():
    data = np.array([0, 64, 65, 252, 253, 254, 255], dtype="uint8")
    encoded_data = encode_data(data)
    assert np.array_equal(
        encoded_data,
        np.array([0, 64, 65, 252, 253, 0, 253, 1, 253, 2], dtype="uint8"),
    )
    decoded_data = decode_data(encoded_data)
    assert np.array_equal(decoded_data, data)
