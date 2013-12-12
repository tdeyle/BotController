# Where is the bot location going to go?
#!python
#cython: cdivision(True), boundscheck(False), wraparound(False)
import cython
import time
import cython_simulator
from libc.math cimport M_PI, sin, cos, fabs, lrint, lround, fmax
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

    cdef int GPS_WIDTH_CELLS
    cdef int GPS_HEIGHT_CELLS
    cdef int LPS_WIDTH_CELLS
    cdef int LPS_HEIGHT_CELLS

    cdef int LPS_ORIGINx
    cdef int LPS_ORIGINy

    cdef int NUM_ROWS
    cdef int NUM_COLS

    cdef int MAX_RANGE
    cdef int SENSOR_FOV

    cdef int OCCUPIED
    cdef int UNOCCUPIED
    cdef int UNKNOWN


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_processArray(np.ndarray[double, ndim=2, mode="c"] input_GPS, np.ndarray[double, ndim=2, mode="c"] input, np.ndarray[int, ndim=1, mode="c"] dist, double origx, double origy, double theta):
    process(&input_GPS[0,0], &input[0,0], &dist[0], origx, origy, theta)

def cy_initArray(np.ndarray[double, ndim=2, mode="c"] input_GPS):
    initialize(&input_GPS[0,0])

cdef void initializeLPS(np.ndarray[double, ndim=2, mode="c"] LPS_arr):

    LPS_arr[0:] = 0.5

cdef void measureDistance(np.ndarray[int, ndim=1, mode="c"] dist_arr, np.ndarray[double, ndim=2, mode="c"] sim_map_arr, double bot_x, double bot_y, double theta):
    # cdef int angle, length
    cdef double local_angle_rad, current_x, current_y
    cdef int x, y
    cdef int angle, length, local_angle, grid_x, grid_y

    x = (int)(bot_x);
    y = (int)(bot_y);

    print "Entering"

    for angle in xrange(SENSOR_FOV):
        local_angle = int(theta + angle) % 360
        local_angle_rad = <double>(local_angle * M_PI / 180.0)

        for length in xrange(MAX_RANGE):

            current_x = x + cos(local_angle_rad) * length
            current_y = y + sin(local_angle_rad) * length

            grid_x = int(current_x / CELL_SIZE)
            grid_y = int(current_y / CELL_SIZE)

            if grid_y >= WORLD_HEIGHT/CELL_SIZE or grid_y < 0 or grid_x >= WORLD_WIDTH/CELL_SIZE or grid_x < 0:
                dist_arr[local_angle] = length
                break
            elif sim_map_arr[grid_y, grid_x] == 1.0:
                dist_arr[local_angle] = length
                break 
            elif length == MAX_RANGE - 1:
                dist_arr[local_angle] = MAX_RANGE-1

cdef void detectHits(np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[int, ndim=1, mode='c'] dist_arr, double theta, double origx, double origy):
    cdef int senseObstacle
    cdef int i, dist, arc, offx, offy
    cdef double hitx, hity

    arc = 0
    offx = 0
    offy = 0

    for i in xrange(SENSOR_FOV):
        dist = dist_arr[i]
        if dist < MAX_RANGE - 1:
            senseObstacle = True
        else:
            senseObstacle = False
        hitx = cos(i*M_PI/180) * dist + origx
        hity = sin(i*M_PI/180) * dist + origy

        print "At ", i, "degrees, ", hitx, hity

        assignOccupancy(LPS_arr, offx, offy, hitx, hity, origx, origy, arc, senseObstacle)

cdef void assignOccupancy(np.ndarray[double, ndim=2, mode='c'] LPS_arr, double offx, double offy, double hitx, double hity, double origx, double origy, double arc, int senseObstacle):
    cdef double rise, run, stepx, stepy, current_cell_x, current_cell_y
    cdef int steps, step

    rise = hity - origy
    if fabs(rise) < 0.1:
        rise = 0.0

    run = hitx - origx
    if fabs(run) < 0.1:
        run = 0.0

    steps = lrint(lround(fmax(fabs(rise/CELL_SIZE), fabs(run/CELL_SIZE))))

    if steps == 0:
        LPS_arr[int(LPS_ORIGINy/CELL_SIZE), int(LPS_ORIGINx/CELL_SIZE)] = OCCUPIED
        return

    stepx = run/steps
    stepy = rise/steps

    if fabs(stepx) > CELL_SIZE:
        stepx = CELL_SIZE
        if run < 0:
            stepx *= -1

    if fabs(stepy) > CELL_SIZE:
        stepy = CELL_SIZE
        if rise < 0:
            stepy *= -1

    current_cell_x, current_cell_y = (origx, origy)

    LPS_arr[int(current_cell_y), int(current_cell_y)] = UNOCCUPIED

    print rise, run, steps, stepx, stepy, current_cell_x, current_cell_y

    for step in range(steps):
        current_cell_x += stepy
        current_cell_y += stepx

        LPS_arr[int(current_cell_y), int(current_cell_y)] = UNOCCUPIED

        if senseObstacle == True:
            LPS_arr[int(hity/CELL_SIZE), int(hitx/CELL_SIZE)] = OCCUPIED
        else:
            LPS_arr[int(hity/CELL_SIZE), int(hitx/CELL_SIZE)] = UNOCCUPIED

def main(bot_state):
    cdef double fBotx, fBoty, fTheta
    
    fBotx, fBoty, fTheta = bot_state

    np.set_printoptions(linewidth=600, threshold='nan', precision=2)
    
    # before_cy = time.clock()

    LPS = np.arange(LPS_WIDTH_CELLS*LPS_HEIGHT_CELLS, dtype=np.float64).reshape(LPS_HEIGHT_CELLS,LPS_WIDTH_CELLS)
    
    np.ascontiguousarray(LPS)

    GPS = np.arange(GPS_WIDTH_CELLS*GPS_HEIGHT_CELLS, dtype=np.float64).reshape(GPS_HEIGHT_CELLS,GPS_WIDTH_CELLS)
    np.ascontiguousarray(GPS)
    cy_initArray(GPS)

    sim_map = np.arange(GPS_WIDTH_CELLS*GPS_HEIGHT_CELLS, dtype=np.float64).reshape(GPS_HEIGHT_CELLS,GPS_WIDTH_CELLS)
    
    cython_simulator.cy_buildMap(sim_map, GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, CELL_SIZE)
    
    distance = np.arange(SENSOR_FOV, dtype=np.int32)

    print sim_map

    before = time.clock()
    measureDistance(distance, sim_map, fBotx, fBoty, fTheta)

    LPS[0:] = 0.5

    detectHits(LPS, distance, fTheta, LPS_ORIGINx, LPS_ORIGINy)
    
    print "Cython: ", time.clock() - before

    print distance
    # print "Bot Location: ", botx, boty

    print LPS
    # print GPS
    # # b = np.arange(360, dtype=np.int32)
    # # b[0:360] = 8

    # # print b

    # # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    # cy_processArray(GPS, LPS, distance, fBotx, fBoty, fTheta)
    
    # after_cy = time.clock()

    # #-------------------------------------
    # # Printing results
    # #-------------------------------------
    # # print "cy array", after_cy-before_cy

    # print "LPS: "
    # print LPS
    # print ""
    # print "GPS: "
    # print GPS