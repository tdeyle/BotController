# Where is the bot location going to go?
#!python
#cython: cdivision(True), boundscheck(False), wraparound(False)
import cython
import time
import cython_simulator
from libc.math cimport M_PI, sin, cos, fabs, lrint, lround, fmax, abs
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange

cdef extern from "process_array.h":
    void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double origx, double origy, double theta)
    void initialize(double *GPS_arr)
    void detectHits(double *LPS_arr, int *dist_arr, double theta)
    
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
    cdef int LPS_ORIGIN_CELLx
    cdef int LPS_ORIGIN_CELLy

    cdef int NUM_ROWS
    cdef int NUM_COLS

    cdef int MAX_RANGE
    cdef int SENSOR_FOV

    cdef int OCCUPIED
    cdef int UNOCCUPIED
    cdef int UNKNOWN

    cdef double max_occupied
    cdef double max_empty

cdef extern from "process_GPS.h":
    void updateFromLPS(double *LPS_arr, double *GPS_arr, double origx, double origy, double theta)

def cy_processArray(np.ndarray[double, ndim=2, mode="c"] input_GPS, np.ndarray[double, ndim=2, mode="c"] input_LPS, np.ndarray[int, ndim=1, mode="c"] dist, double origx, double origy, double theta):
    process(&input_GPS[0,0], &input_LPS[0,0], &dist[0], origx, origy, theta)

def cy_initArray(np.ndarray[double, ndim=2, mode="c"] input_GPS):
    initialize(&input_GPS[0,0])

def c_detectHits(np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[int, ndim=1, mode='c'] dist_arr, double theta):
    detectHits(&LPS_arr[0,0], &dist_arr[0], theta)

def c_updateFromLPS(np.ndarray[double, ndim=2, mode='c'] input_LPS, np.ndarray[double, ndim=2, mode='c'] input_GPS, double origx, double origy, double theta):
    updateFromLPS(&input_LPS[0,0], &input_GPS[0,0], origx, origy, theta)

cdef void initializeLPS(np.ndarray[double, ndim=2, mode="c"] LPS_arr):

    LPS_arr[0:] = 0.5

cdef void measureDistance(np.ndarray[int, ndim=1, mode="c"] dist_arr, np.ndarray[double, ndim=2, mode="c"] sim_map_arr, double bot_x, double bot_y, double theta):
    # cdef int angle, length
    cdef double local_angle_rad, current_x, current_y
    cdef int x, y
    cdef int angle, length, local_angle, grid_x, grid_y

    x = lrint(bot_x);
    y = lrint(bot_y);

    print "Entering"
    
    for angle in xrange(SENSOR_FOV):
    # for angle in prange(SENSOR_FOV, nogil=True, num_threads=10):
        local_angle = <int>(theta + angle) % 360
        local_angle_rad = <double>(local_angle * M_PI / 180.0)

        for length in xrange(MAX_RANGE):
        # for length in prange(MAX_RANGE, num_threads=10):
            current_x = x + cos(local_angle_rad) * length
            current_y = y + sin(local_angle_rad) * length

            grid_x = lrint(current_x / CELL_SIZE)
            grid_y = lrint(current_y / CELL_SIZE)

            if grid_y >= WORLD_HEIGHT/CELL_SIZE or grid_y < 0 or grid_x >= WORLD_WIDTH/CELL_SIZE or grid_x < 0:
                dist_arr[local_angle] = length
                break
            elif sim_map_arr[grid_y, grid_x] == 1.0:
                dist_arr[local_angle] = length
                break 
            elif length == MAX_RANGE - 1:
                dist_arr[local_angle] = MAX_RANGE-1

cdef void cy_detectHits(np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[int, ndim=1, mode='c'] dist_arr, double theta):
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

        cy_assignOccupancy(LPS_arr, offx, offy, hitx, hity, arc, senseObstacle)

cdef void cy_assignOccupancy(np.ndarray[double, ndim=2, mode='c'] LPS_arr, int offx, int offy, double hitx, double hity, int arc, int senseObstacle):
    cdef double rise, run, stepx, stepy, fcurrent_cell_x, fcurrent_cell_y
    cdef int steps, step, cell_hitx, cell_hity, current_cell_x, current_cell_y

    rise = (hity - LPS_ORIGINy) / CELL_SIZE
    if fabs(rise) < 0.1:
        rise = 0.0

    run = (hitx - LPS_ORIGINx) / CELL_SIZE
    if fabs(run) < 0.1:
        run = 0.0

    steps = lrint(lround(fmax(fabs(rise), fabs(run))))

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

    fcurrent_cell_x, fcurrent_cell_y = (LPS_ORIGINx/CELL_SIZE, LPS_ORIGINy/CELL_SIZE)
    current_cell_x, current_cell_y = (lrint(fcurrent_cell_x), lrint(fcurrent_cell_y))

    LPS_arr[current_cell_y, current_cell_x] = UNOCCUPIED
    cell_hitx, cell_hity = (lrint(hitx/CELL_SIZE), lrint(hity/CELL_SIZE))

    # print "Rise: ", rise, "run: ", run, "steps: ", steps, "stepx: ", stepx, "stepy: ", stepy, "current cells: ", current_cell_x, current_cell_y

    for step in xrange(steps):
        fcurrent_cell_x += stepx
        fcurrent_cell_y += stepy

        current_cell_x = lrint(fcurrent_cell_x)
        current_cell_y = lrint(fcurrent_cell_y)

        LPS_arr[current_cell_y, current_cell_x] = UNOCCUPIED

        if senseObstacle == True:
            LPS_arr[cell_hity, cell_hitx] = OCCUPIED
        else:
            LPS_arr[cell_hity, cell_hitx] = UNOCCUPIED

cdef void cy_updateFromLPS(np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[double, ndim=2, mode='c'] GPS_arr, double origx, double origy, double theta):
    
    # Declare boundary variables
    cdef int LPS_lower_boundsX, LPS_lower_boundsY, LPS_upper_boundsX, LPS_upper_boundsY
    cdef int GPS_lower_boundsX, GPS_lower_boundsY, GPS_upper_boundsX, GPS_upper_boundsY
    cdef int Lower_boundaryFlagX, Lower_boundaryFlagY, Upper_boundaryFlagX, Upper_boundaryFlagY

    # Declare bot location variables
    cdef int botx, bot_y

    cdef int active_rows, active_cols

    cdef int GPS_skip, LPS_skip
    cdef int GPSidx, LPSidx

    botx = lrint(origx/CELL_SIZE)
    boty = lrint(origy/CELL_SIZE)

    # Check bot location values
    if botx < 0 or botx > GPS_WIDTH_CELLS:
        print "Invalid x location"
        return

    if boty < 0 or boty > GPS_HEIGHT_CELLS:
        print "Invalid y location"
        return

    # Setup initial boundary values
    LPS_lower_boundsX = 0
    LPS_lower_boundsY = 0
    GPS_lower_boundsX = botx - LPS_ORIGIN_CELLx
    GPS_lower_boundsY = boty - LPS_ORIGIN_CELLy

    LPS_upper_boundsX = LPS_WIDTH_CELLS - 1
    LPS_upper_boundsY = LPS_HEIGHT_CELLS - 1
    GPS_upper_boundsX = botx + LPS_ORIGIN_CELLx - 1
    GPS_upper_boundsY = boty + LPS_ORIGIN_CELLy - 1

    # Trip the boundary flags if LPS is over the limits of the GPS
    Lower_boundaryFlagX = botx - LPS_ORIGIN_CELLx
    Lower_boundaryFlagY = boty - LPS_ORIGIN_CELLy
    Upper_boundaryFlagX = botx + LPS_ORIGIN_CELLx
    Upper_boundaryFlagY = boty + LPS_ORIGIN_CELLy

    print "Bot Location:", botx, boty
    print "NumCols: ", GPS_WIDTH_CELLS, "NumRows: ", GPS_HEIGHT_CELLS
    print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY, "GPS_upper_boundsX: ", GPS_upper_boundsX, "GPS_upper_boundsY: ", GPS_upper_boundsY
    print "LPS_lower_boundsX: ", LPS_lower_boundsX, "LPS_lower_boundsY: ", LPS_lower_boundsY, "LPS_upper_boundsX: ", LPS_upper_boundsX, "LPS_upper_boundsY: ", LPS_upper_boundsY
    print "LowerBoundaryFlagX: ", Lower_boundaryFlagX, "LowerBoundaryFlagY: ", Lower_boundaryFlagY
    print "UpperBoundaryFlagX: ", Upper_boundaryFlagX, "UpperBoundaryFlagY: ", Upper_boundaryFlagY

    if Lower_boundaryFlagX < 0:
        LPS_lower_boundsX = abs(Lower_boundaryFlagX)
        GPS_lower_boundsX = 0
        print "Lower_boundaryFlagX hit"

    if Lower_boundaryFlagY < 0:
        LPS_lower_boundsY = abs(Lower_boundaryFlagY)
        GPS_lower_boundsY = 0
        print "Lower_boundaryFlagY hit"

    if Upper_boundaryFlagX > GPS_WIDTH_CELLS - 1:
        LPS_upper_boundsX = GPS_WIDTH_CELLS - 1 - botx + LPS_ORIGIN_CELLx
        GPS_upper_boundsX = GPS_WIDTH_CELLS - 1
        print "Upper_boundaryFlagX hit"

    if Upper_boundaryFlagY > GPS_HEIGHT_CELLS - 1:
        LPS_upper_boundsY = GPS_HEIGHT_CELLS - 1 - boty + LPS_ORIGIN_CELLy
        GPS_upper_boundsY = GPS_HEIGHT_CELLS - 1
        print "Upper_boundaryFlagY hit"

    print "------------After Boundary Check-----------"
    print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY
    print "GPS_upper_boundsX: ", GPS_upper_boundsX, "GPS_upper_boundsY: ", GPS_upper_boundsY

    active_rows = LPS_upper_boundsY - LPS_lower_boundsY + 1
    active_cols = LPS_upper_boundsX - LPS_lower_boundsX + 1

    GPSidx = GPS_lower_boundsX + (GPS_lower_boundsY * GPS_WIDTH_CELLS)
    LPSidx = LPS_lower_boundsX + (LPS_lower_boundsY * LPS_WIDTH_CELLS)

    GPS_skip = GPS_WIDTH_CELLS + active_cols
    LPS_skip = GPS_HEIGHT_CELLS + active_rows

    print "---------------------------------------------"
    print "active_rows: ", active_rows, "active_cols: ", active_cols
    print "GPS_skip: ", GPS_skip, "GPSidx: ", GPSidx
    print "LPS_skip: ", LPS_skip, "LPSidx: ", LPSidx

    cdef int x, y, row, col

    GPSx = GPS_lower_boundsX
    GPSy = GPS_lower_boundsY

    LPSx = LPS_lower_boundsX
    LPSy = LPS_lower_boundsY

    for row in range(active_rows):
        for col in range(active_cols):
            if LPS_arr[LPSx, LPSy] == 0.5:
                GPS_arr[GPSx, GPSy] = GPS_arr[GPSx, GPSy]
            else:
                GPS_arr[GPSx, GPSy] = cy_getProb(GPS_arr[GPSx,GPSy], lrint(LPS_arr[LPSx,LPSy]))

            GPSx += 1
            LPSx += 1
        GPSy += 1
        LPSy += 1
        GPSx = GPS_lower_boundsX
        LPSx = LPS_lower_boundsY

cdef double cy_getProb(double prior_occ, int obstacle_is_sensed):
    cdef double POcc, PEmp, inv_prior, new_prob

    inv_prior = 1.0 - prior_occ

    POcc = max_occupied
    PEmp = max_empty
    new_prob = prior_occ

    if obstacle_is_sensed == 1:
        new_prob = (POcc * prior_occ) / ((POcc * prior_occ) + (PEmp * inv_prior))
    elif obstacle_is_sensed == 0:
        new_prob = (PEmp * prior_occ) / ((PEmp * prior_occ) + (POcc * inv_prior))

    return new_prob

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
    
    measureDistance(distance, sim_map, fBotx, fBoty, fTheta)
    print "measureDistance Done"

    LPS[0:] = 0.5

    # cy_detectHits(LPS, distance, fTheta)
    # before = time.clock()
    
    cy_detectHits(LPS, distance, fTheta)
    cy_updateFromLPS(LPS, GPS, fBotx, fBoty, fTheta)

    # print "C: ", time.clock() - before_cy

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
    
    print "cython: ", time.clock() - before_cy

    # #-------------------------------------
    # # Printing results
    # #-------------------------------------
    print distance
    print "LPS: "
    print LPS
    print ""
    print "GPS: "
    print GPS