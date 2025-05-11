from collections import deque
import numpy as np
import serial
from src.serial_comm import (
    send_data_to_arduino, 
    receive_data_from_arduino,  
)
import logging
import os

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

logger.info(f'{filename} started')

ser = serial.Serial("/dev/tty.usbmodem112977801", 57600)
logger.info("Connected to Arduino.")

test_data = [
    (1, np.arange(0, 4794, dtype="uint8")),
    (2, np.arange(0, 4794, dtype="uint8")),
    (3, np.random.randint(256, size=4794, dtype="uint8")),
    (4, np.random.randint(256, size=4794, dtype="uint8"))
]

# Add check-sums
test_data = deque([(i, data, np.sum(data)) for i, data in test_data])

waiting_for_response = False
responses = []
while True:

    if ser.in_waiting == 0 and waiting_for_response is False:
        try:
            i, data, check_sum = test_data.popleft()
        except IndexError:
            break
        send_data_to_arduino(ser, data)
        waiting_for_response = True
        logger.info(f"Test data {i} sent.")

    if ser.in_waiting > 0:
        num_bytes, data_received = receive_data_from_arduino(ser)
        if num_bytes == 0:
            logger.info(f"Debug message: {data_received.tobytes()}")
        else:
            num_bytes_received = (
                int(data_received[0]) * 256 + int(data_received[1])
            )
            data_sum = (
                int(data_received[2]) * 256 ** 3
                + int(data_received[3]) * 256 ** 2
                + int(data_received[4]) * 256
                + int(data_received[5])
            )
            logger.info(
                f"Bytes received: {num_bytes_received}, check-sum: {check_sum}"
            )
            assert num_bytes_received == data.shape[0] + 2
            assert data_sum == check_sum
            waiting_for_response = False

ser.close()

logger.info(f"{filename} complete")
