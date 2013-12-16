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

cdef void detectHits(np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[int, ndim=1, mode='c'] dist_arr, double theta):
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
        hitx = cos(theta+i*M_PI/180) * dist + LPS_ORIGINx
        hity = sin(theta+i*M_PI/180) * dist + LPS_ORIGINy

        assignOccupancy(LPS_arr, offx, offy, hitx, hity, arc, senseObstacle)

cdef void assignOccupancy(np.ndarray[double, ndim=2, mode='c'] LPS_arr, int offx, int offy, double hitx, double hity, int arc, int senseObstacle):
    cdef double rise, run, stepx, stepy, fcurrent_x, fcurrent_y
    cdef int steps, step, cell_hitx, cell_hity, current_cell_x, current_cell_y

    rise = (hity - LPS_ORIGINy) / CELL_SIZE #hity/CELL_SIZE - LPS_ORIGINy/CELL_SIZE
    if abs(rise) < 0.1:
        rise = 0.0

    run = (hitx - LPS_ORIGINx) / CELL_SIZE #hitx/CELL_SIZE - LPS_ORIGINx/CELL_SIZE
    if abs(run) < 0.1:
        run = 0.0

    steps = int(round(max(abs(rise), abs(run))))

    # if steps == 0:
    #     LPS_arr[int(LPS_ORIGINy/CELL_SIZE), int(LPS_ORIGINx/CELL_SIZE)] = OCCUPIED
    #     return

    stepx = run/steps
    stepy = rise/steps

    if abs(stepx) > CELL_SIZE:
        stepx = CELL_SIZE
        if run < 0:
            stepx *= -1

    if abs(stepy) > CELL_SIZE:
        stepy = CELL_SIZE
        if rise < 0:
            stepy *= -1

    fcurrent_cell_x, fcurrent_cell_y = (LPS_ORIGINx/CELL_SIZE, LPS_ORIGINy/CELL_SIZE)
    current_cell_x, current_cell_y = (int(fcurrent_cell_x), int(fcurrent_cell_y))

    LPS_arr[current_cell_y, current_cell_x] = 2 # UNOCCUPIED
    cell_hitx, cell_hity = (int(hitx/CELL_SIZE), int(hity/CELL_SIZE))

    # print "Rise: ", rise, "run: ", run, "steps: ", steps, "stepx: ", stepx, "stepy: ", stepy, "current cells: ", current_cell_x, current_cell_y

    for step in range(steps):
        fcurrent_cell_x += stepx
        fcurrent_cell_y += stepy

        current_cell_x = int(fcurrent_cell_x)
        current_cell_y = int(fcurrent_cell_y)

        LPS_arr[current_cell_y, current_cell_x] = UNOCCUPIED

        if senseObstacle == True:
            LPS_arr[cell_hity, cell_hitx] = OCCUPIED
        else:
            LPS_arr[cell_hity, cell_hitx] = UNOCCUPIED

def main(bot_state):
    cdef double fBotx, fBoty, fTheta
    
    fBotx, fBoty, fTheta = bot_state

    np.set_printoptions(linewidth=600, threshold='nan', precision=2)
    
    before_cy = time.clock()

    LPS = np.arange(LPS_WIDTH_CELLS*LPS_HEIGHT_CELLS, dtype=np.float64).reshape(LPS_HEIGHT_CELLS,LPS_WIDTH_CELLS)
    
    np.ascontiguousarray(LPS)

    GPS = np.arange(GPS_WIDTH_CELLS*GPS_HEIGHT_CELLS, dtype=np.float64).reshape(GPS_HEIGHT_CELLS,GPS_WIDTH_CELLS)
    np.ascontiguousarray(GPS)
    cy_initArray(GPS)

    sim_map = np.arange(GPS_WIDTH_CELLS*GPS_HEIGHT_CELLS, dtype=np.float64).reshape(GPS_HEIGHT_CELLS,GPS_WIDTH_CELLS)
    
    cython_simulator.cy_buildMap(sim_map, GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, CELL_SIZE)
    
    distance = np.arange(SENSOR_FOV, dtype=np.int32)

    before = time.clock()
    measureDistance(distance, sim_map, fBotx, fBoty, fTheta)

    LPS[0:] = 0.5

    detectHits(LPS, distance, fTheta)
    
    print "Cython: ", time.clock() - before

    # print distance
    # print "Bot Location: ", botx, boty

    # print LPS
    # print sim_map
    # print GPS
    # # b = np.arange(360, dtype=np.int32)
    # # b[0:360] = 8

    # # print b

    # # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    # cy_processArray(GPS, LPS, distance, fBotx, fBoty, fTheta)
    
    after_cy = time.clock()

    # #-------------------------------------
    # # Printing results
    # #-------------------------------------
    # print "cy array", after_cy-before_cy

    print "LPS: "
    print LPS
    # print ""
    # print "GPS: "
    # print GPS