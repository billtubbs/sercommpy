from itertools import cycle
from collections import deque
import numpy as np
import serial
from serial_comm.serial_comm import (
    connect_to_arduino, send_data_to_arduino, receive_data_from_arduino
)
from display1593 import *
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
# Usually, 
#  - TEENSY1 is on usb port '/dev/ttyACM1'
#  - TEENSY2 is on usb port '/dev/ttyACM0'



def move_pointer_commands(i1, i2):
    return [
        np.array(list(b'L1') + [0, i1, 0, 0, 0], dtype="uint8"),
        np.array(list(b'L1') + [0, i2, 20, 32, 0], dtype="uint8")
    ]
    

def manual_testing(ser):

    status, message = connect_to_arduino(ser)
    if status == 0:
        worker_name = message
    else:
        raise Exception(message)
    logger.info(f"Hello from: {worker_name}")

    # # Clear display
    # data = np.array(list(b'LC'), dtype="uint8")
    # send_data_to_arduino(ser, data)

    # for i in range(50):
    #     data = np.array(list(b'L1') + [0, i, 32, 4, 32], dtype="uint8")
    #     send_data_to_arduino(ser, data)

    # # Show LED updates now
    # np.array(list(b'SN'), dtype="uint8")
    # send_data_to_arduino(ser, data)

    waiting_for_response = False
    command_queue = deque()
    led_id = 0
    while True:

        # Check for new key presses
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    led_id_new = (led_id - 1) % 100
                    command_queue.extendleft(
                        move_pointer_commands(led_id, led_id_new)
                    )
                    led_id = led_id_new
                    logger.info("LEFT key pressed.")
                if event.key == pygame.K_RIGHT:
                    led_id_new = (led_id + 1) % 100
                    command_queue.extendleft(
                        move_pointer_commands(led_id, led_id_new)
                    )
                    led_id = led_id_new
                    logger.info("RIGHT key pressed.")

        if ser.in_waiting == 0 and waiting_for_response is False:
            if len(command_queue) > 0:
                data = command_queue.pop()
                send_data_to_arduino(ser, data)
                logger.info("Command sent.")
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


def run_test(ser):

    logger.info("Preparing test data...")
    test_data = []
    test_data.append(clear_all_leds())

    test_data.append(show_now())
    
    # TEENSY1
    # 0 100
    # 1 100
    # 2 98
    # 3 100
    # 4 100
    # 5 100
    # 6 100
    # 7 100

    # TEENSY2
    # 0 99
    # 1 99
    # 2 99
    # 3 100
    # 4 100
    # 5 100
    # 6 100
    # 7 98

    # String start number
    s = 700
    
    # Light first 95 leds
    leds = np.arange(s, s + 95)
    rgb_array = 16 * np.ones((95, 3))
    test_data.append(set_leds(leds, rgb_array))
    
    test_data.append(show_now())
    
    # Light 5 more leds
    leds = np.arange(s + 95, s + 100)
    rgb_5 = np.array([
        (32, 0, 0), (0, 32, 0), (0, 0, 32), (32, 32, 0), (32, 0, 32)
    ], dtype='uint8')
    test_data.append(set_leds(leds, rgb_5))

    test_data.append(show_now())

    # Iterator to cycle through test data
    test_data_cycle = cycle(test_data)

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
                #time.sleep(0.5)

    t_end = time.time()
    logger.info(f"Elapsed time: {t_end - t_start:.3f}s for {n_iter} tests.")
    logger.info(f"Cycle time: {(t_end - t_start) * 1000 / n_iter:.0f}ms.")


def open_serial_connections(ports):
    connections = {}
    for port in SERIAL_PORTS:
        conn = serial.Serial(port, baudrate=BAUD_RATE)
        logger.info(f'Connected to port {port}.')
        
        status, message = connect_to_arduino(conn)
        if status == 0:
            worker_name = message
        else:
            raise Exception(message)
        logger.info(f"Hello from: {worker_name}")
        connections[worker_name] = conn

    return connections


def close_serial_connections(connections):
    for name, conn in connections.items():
        conn.close()
        logger.info(f'Closed connection to {name}.')


def main():
    logger.info('='*35)
    logger.info(f'{filename} started.')
    connections = open_serial_connections(SERIAL_PORTS)
    run_test(connections['TEENSY1'])
    #run_test(connections['TEENSY2'])
    #manual_testing(connections['TEENSY1'])
    close_serial_connections(connections)
    logger.info(f'{filename} ended.')


if __name__ == "__main__":
    main()
