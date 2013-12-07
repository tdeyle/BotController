import time
from math import sin, cos
# from scan import *
import numpy as np
import cython_process_array
import cython_simulator

if __name__ == "__main__":

    WORLD_WIDTH = 12000   #// in millimeters
    WORLD_HEIGHT = 6000   #// in millimeters
    SCREEN_WIDTH = 1200   #// in px
    SCREEN_HEIGHT = 600   #// in px
    CELL_SIZE = 240       #// in millimeters
    LPS_WIDTH = 20
    LPS_HEIGHT = 20
    GPS_WIDTH = WORLD_WIDTH/CELL_SIZE
    GPS_HEIGHT = WORLD_HEIGHT/CELL_SIZE

    NUM_ROWS = WORLD_WIDTH/CELL_SIZE
    NUM_COLS = WORLD_HEIGHT/CELL_SIZE

    #---------------------------------------
    # Cython Implementation with 'C'
    #---------------------------------------
    before_cy = time.clock()

    LPS = np.arange(LPS_WIDTH*LPS_HEIGHT, dtype=np.float64).reshape((LPS_HEIGHT,LPS_WIDTH))
    GPS = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))
    cython_process_array.cy_initArray(GPS)


    sim_map = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))

    cython_simulator.cy_buildMap(sim_map, NUM_ROWS, NUM_COLS, CELL_SIZE)
    print sim_map

    b = np.arange(360, dtype=np.int32)
    b[0:360] = 8

    # print b

    # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    cython_process_array.cy_processArray(GPS, LPS, b, 48.0, 11.0, 0.0)
    np.set_printoptions(linewidth=350, threshold='nan', precision=2)
    
    after_cy = time.clock()

    #-------------------------------------
    # Printing results
    #-------------------------------------
    print "cy array", after_cy-before_cy

    print LPS
    print ""
    print GPS
