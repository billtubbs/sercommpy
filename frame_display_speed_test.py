# Simple test script

import time
import numpy as np

from display1593 import Display1593

dis = Display1593()
dis.connect()

# Prepare led data
N_FRAMES = 20
data = []
for col in range(N_FRAMES):
   data.append(np.full((1593, 3), col, dtype="uint8"))

display_times = []
t0 = time.time()
for d in data:
    dis.set_all_leds(d)
    dis.show_now()
    display_times.append(time.time() - t0)

print("Display times (ms):")
t_prev = 0.0
for i, t in enumerate(display_times, start=1):
    print(f"{i:02d}, {t * 1000:.1f}, {(t - t_prev) * 1000:.1f}")
    t_prev = t

dis.clear_all()
dis.show_now()
