import time
from math import sin, cos
# from scan import *
import numpy as np
import cython_process_array
import cython_simulator

if __name__ == "__main__":
    bot_state = (6000.0, 6000.0, 0)
    cython_process_array.main(bot_state)