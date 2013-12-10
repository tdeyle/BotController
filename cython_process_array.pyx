# Where is the bot location going to go?

import cython
import time
import cython_simulator
from libc.math cimport M_PI, sin, cos
import numpy as np
cimport numpy as np

cdef extern from "process_array.h":
    void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double origx, double origy, double theta)
    void initialize(double *GPS_arr)

cdef extern from "array_parameters.h":
    cdef int WORLD_WIDTH
    cdef int WORLD_HEIGHT
    cdef int SCREEN_WIDTH
    cdef int SCREEN_HEIGHT
    cdef int CELL_SIZE 

    cdef int GPS_WIDTH
    cdef int GPS_HEIGHT
    cdef int LPS_WIDTH
    cdef int LPS_HEIGHT

    cdef int NUM_ROWS
    cdef int NUM_COLS

    cdef int MAX_RANGE


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_processArray(np.ndarray[double, ndim=2, mode="c"] input_GPS, np.ndarray[double, ndim=2, mode="c"] input, np.ndarray[int, ndim=1, mode="c"] dist, double origx, double origy, double theta):
    process(&input_GPS[0,0], &input[0,0], &dist[0], origx, origy, theta)

def cy_initArray(np.ndarray[double, ndim=2, mode="c"] input_GPS):
    initialize(&input_GPS[0,0])

def measureDistance(np.ndarray[int, ndim=1, mode="c"] dist_arr, np.ndarray[double, ndim=2, mode="c"] sim_map_arr, double bot_x, double bot_y, double theta):
    # cdef int angle, length
    cdef double degrees

    print "Entering"

    for angle in xrange(360):
        local_angle = (theta + angle % 360.0)
        # print "local_angle", local_angle
        degrees = angle * M_PI / 180.0

        for length in xrange(MAX_RANGE):

            # print "length: ", length, "degrees: ", degrees

            current_x = bot_x + (cos(degrees) * length)
            current_y = bot_y + (sin(degrees) * length)

            grid_x = int(current_x / CELL_SIZE)
            grid_y = int(current_y / CELL_SIZE)

            # print "angle: ", local_angle, "current X,Y: ", current_x, ",", current_y, "grid X,y: ", grid_x, ",", grid_y, sim_map_arr[grid_y, grid_x]
            
            if grid_y >= WORLD_HEIGHT/CELL_SIZE or grid_y < 0 or grid_x >= WORLD_WIDTH/CELL_SIZE or grid_x < 0:
                dist_arr[int(local_angle)] = length
                # print "Over bounds"
                break
            elif sim_map_arr[grid_y,grid_x] == 1.0:
                dist_arr[int(local_angle)] = length
                # print "Hit @ ", length
                # break 
            elif length == MAX_RANGE - 1:
                dist_arr[int(local_angle)] = MAX_RANGE-1
                # print "No hit", local_angle


def main(bot_state):
    cdef double botx, boty, theta
    
    botx, boty, theta = bot_state

    before_cy = time.clock()

    LPS = np.arange(LPS_WIDTH*LPS_HEIGHT, dtype=np.float64).reshape((LPS_HEIGHT,LPS_WIDTH))
    np.ascontiguousarray(LPS)

    GPS = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))
    np.ascontiguousarray(GPS)
    cy_initArray(GPS)

    sim_map = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))
    distance = np.arange(360, dtype=np.int32)

    cython_simulator.cy_buildMap(sim_map, NUM_ROWS, NUM_COLS, CELL_SIZE)
    print sim_map

    # cython_simulator.cy_scanDistance(distance, sim_map, botx, boty, theta)
    measureDistance(distance, sim_map, botx, boty, theta)
    print "-----------------------------"
    print distance, botx, boty
    # b = np.arange(360, dtype=np.int32)
    # b[0:360] = 8

    # print b

    # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    cy_processArray(GPS, LPS, distance, botx, boty, 0.0)
    np.set_printoptions(linewidth=350, threshold='nan', precision=2)
    
    after_cy = time.clock()

    #-------------------------------------
    # Printing results
    #-------------------------------------
    print "cy array", after_cy-before_cy

    print LPS
    print ""
    print GPS