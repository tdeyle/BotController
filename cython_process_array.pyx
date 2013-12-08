import cython
import time
import cython_simulator
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


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_processArray(np.ndarray[double, ndim=2, mode="c"] input_GPS, np.ndarray[double, ndim=2, mode="c"] input, np.ndarray[int, ndim=1, mode="c"] dist, double origx, double origy, double theta):
    process(&input_GPS[0,0], &input[0,0], &dist[0], origx, origy, theta)

def cy_initArray(np.ndarray[double, ndim=2, mode="c"] input_GPS):
    initialize(&input_GPS[0,0])

def main():
    before_cy = time.clock()

    LPS = np.arange(LPS_WIDTH*LPS_HEIGHT, dtype=np.float64).reshape((LPS_HEIGHT,LPS_WIDTH))
    GPS = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))
    cy_initArray(GPS)

    sim_map = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))

    cython_simulator.cy_buildMap(sim_map, NUM_ROWS, NUM_COLS, CELL_SIZE)
    print sim_map

    b = np.arange(360, dtype=np.int32)
    b[0:360] = 8

    # print b

    # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    cy_processArray(GPS, LPS, b, 10.0, 11.0, 0.0)
    np.set_printoptions(linewidth=350, threshold='nan', precision=2)
    
    after_cy = time.clock()

    #-------------------------------------
    # Printing results
    #-------------------------------------
    print "cy array", after_cy-before_cy

    print LPS
    print ""
    print GPS
