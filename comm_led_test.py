from itertools import cycle
import numpy as np
import serial
from serial_comm.serial_comm import (
    connect_to_arduino, send_data_to_arduino, receive_data_from_arduino
)
import logging
import os
import time


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


def connect(address="/dev/tty.usbmodem112977801", baud=57600):
    ser = serial.Serial(address, baud)
    return ser


def run_test(ser):

    test_data = [
        np.array(list(b'LC'), dtype="uint8"),
        np.array(list(b'SN'), dtype="uint8"),
        np.array(list(b'L1') + [0, 0, 32, 4, 32], dtype="uint8"),
        np.array(list(b'SN'), dtype="uint8"),
        np.array(
            list(b'LN') + [
                0, 4,
                0, 0, 32, 4, 10,
                0, 2, 4, 10, 32,
                0, 4, 10, 32, 4,
                0, 6, 10, 4, 32
            ],
            dtype="uint8"
        ),
        np.array(list(b'SN'), dtype="uint8"),
        np.array(list(b'LA') + 7 * [32, 4, 32], dtype="uint8"),
        np.array(list(b'SN'), dtype="uint8"),
        np.array(list(b'LC'), dtype="uint8"),
        np.array(list(b'SN'), dtype="uint8"),
    ]

    # Calculate check-sums to check data transmission
    test_data_cycle = cycle(test_data)

    status, message = connect_to_arduino(ser)
    if status == 0:
        worker_name = message
    else:
        raise Exception(message)
    logger.info(f"Hello from: {worker_name}")

    time.sleep(1.0)
    t_start = time.time()
    logger.info("Test start")

    waiting_for_response = False
    n_iter = len(test_data)
    i_iter = 0
    while i_iter < n_iter:

        if ser.in_waiting == 0 and waiting_for_response is False:
            try:
                data = next(test_data_cycle)
            except IndexError:
                break
            logger.info(f"Sending Test {i_iter} data...")
            send_data_to_arduino(ser, data)
            logger.info("Data sent.")
            waiting_for_response = True

        if ser.in_waiting > 0:
            logger.info("Receiving data...")
            data_received = receive_data_from_arduino(ser)
            logger.info("Data received.")
            if np.array_equal(data_received[:2], [0, 0]):
                # Debug message from Arduino
                logger.info(f"Debug message: {data_received[2:].tobytes()}")
            else:
                # Process data from Arduino - data integrity checks
                assert data_received.shape[0] == 6
                num_bytes_received = (
                    int(data_received[0]) * 256 + int(data_received[1])
                )
                data_sum = (
                    int(data_received[2]) * 16777216
                    + int(data_received[3]) * 65536
                    + int(data_received[4]) * 256
                    + int(data_received[5])
                )
                assert num_bytes_received == data.shape[0]
                assert data_sum == data.sum()
                logger.info(f"Test {i_iter} complete.")
                i_iter += 1
                waiting_for_response = False
                time.sleep(0.5)

    t_end = time.time()
    logger.info(f"Elapsed time: {t_end - t_start:.3f}s for {n_iter} tests.")
    logger.info(f"Cycle time: {(t_end - t_start) * 1000 / n_iter:.0f}ms.")


def main():
    logger.info('='*35)
    logger.info(f'{filename} started.')
    ser = connect()
    logger.info("Connected to Arduino.")
    run_test(ser)
    logger.info(f"{filename} complete.")
    ser.close()


if __name__ == "__main__":
    main()
