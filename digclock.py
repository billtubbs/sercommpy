# Code to load digit data and light
# leds

import numpy as np
import pickle
from datetime import datetime
from display1593 import Display1593


with open('digdata.pickle', 'rb') as handle:
    dig_data = pickle.load(handle)

print("data for %d digit segments unpickled." % (len(dig_data)))

d_chars = {
        0: [0, 1, 2, 4, 5, 6],
        1: [5, 6],
        2: [1, 2, 3, 4, 5],
        3: [2, 3, 4, 5, 6],
        4: [0, 3, 5, 6],
        5: [0, 2, 3, 4, 6],
        6: [0, 1, 2, 3, 4, 6],
        7: [2, 5, 6],
        8: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 2, 3, 4, 5, 6],
        10: [7]
        }

# Initialise display memory
smem = [0] * 1593
smem_prev = [-1] * 1593

# Digits
#  dig_data[1]
#  dig_data[2]
#  dig_data[3]
#  dig_data[4]
# Points:
#  dig_data[0]

# brightness values
bcycle = {
    0: 9,
    1: 9,
    2: 9,
    3: 9,
    4: 9,
    5: 9,
    6: 8,
    7: 5,
    8: 3,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 2,
    18: 5,
    19: 8,
    20: 9,
    21: 9,
    22: 9,
    23: 9
}

# Get current time
t = datetime.now().time()
hr, min = (t.hour, t.minute)
d4, d3 = (hr // 10), (hr % 10)
d2, d1 = (min // 10), (min % 10)

# Connect to Teensies
dis = Display1593()
dis.connect()

# Set display dimmer level
bness = bcycle[hr % 24]

# Set colour for points
for n in range(2):
    for i, x in dig_data[0][n].items():
        smem[i] = x // bness

# Set colour for digit 4
for n in d_chars[d4]:
    for i, x in dig_data[1][n].items():
        smem[i] += x // bness

# Set colour for digit 3
for n in d_chars[d3]:
    for i, x in dig_data[2][n].items():
        smem[i] += x // bness

# Set colour for digit 2
for n in d_chars[d2]:
    for i, x in dig_data[3][n].items():
        smem[i] += x // bness

# Prepare empty array for RGB color vector
rgb = np.zeros(3, dtype="uint8")

while True:

    # Set colour for digit 1
    for n in d_chars[d1]:
        for i, x in dig_data[4][n].items():
            smem[i] += x // bness

    for i in range(1593):
        if smem[i] != smem_prev[i]:
            rgb[:] = (smem[i], 0, 0)
            dis.set_led(i, rgb)
            smem_prev[i] = smem[i]

    t = datetime.now().time()
    m = t.minute
    hr = t.hour
    s = t.second

    dis.show_now()
    print("%2d:%2d " % (hr, m))

    while t.minute == m:
        while datetime.now().time().second == s:
            pass
        for n in range(2):
            for i, x in dig_data[0][n].items():
                rgb[:] = ((s % 2) * x // bness, 0, 0)
                dis.set_led(i, rgb)
        t = datetime.now().time()
        s = t.second

    m = (m + 1) % 60
    if m == 0:
        hr = (hr + 1) % 24

    # Set value for digits 1
    d1 = (min % 10)

    if d1 == 0:

        # Set value for digits 1
        d2 = (min / 10)

        for n in range(7):
            for i, x in dig_data[3][n].items():
                smem[i] = 0

        for n in d_chars[d2]:
            for i, x in dig_data[3][n].items():
                smem[i] = x // bness

    if min == 0:

        # Set display dimmer level
        bness = bcycle[hr % 24]

        # Set values for digits 3 and 4
        d4, d3 = (hr / 10), (hr % 10)

        for n in range(7):
            for i, x in dig_data[2][n].items():
                smem[i] = 0

        for n in d_chars[d3]:
            for i, x in dig_data[2][n].items():
                smem[i] = x // bness

        for n in range(7):
            for i, x in dig_data[1][n].items():
                smem[i] = 0

        for n in d_chars[d4]:
            for i, x in dig_data[1][n].items():
                smem[i] = x // bness

    for n in range(7):
        for i, x in dig_data[4][n].items():
            smem[i] = 0