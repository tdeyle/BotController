#!python
#cython: cdivision(True), boundscheck(False), wraparound(False)
import cython
import time
import cython_simulator
from libc.math cimport M_PI, sin, cos, fabs, lrint, lround, fmax, abs, ceil
import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdint cimport uint32_t, int32_t

ctypedef float real_t
ctypedef uint32_t uint_t
ctypedef int32_t int_t

cdef extern from "array_parameters.h":
    cdef int_t WORLD_WIDTH
    cdef int_t WORLD_HEIGHT
    cdef int_t SCREEN_WIDTH
    cdef int_t SCREEN_HEIGHT
    cdef int_t CELL_SIZE 

    cdef int_t GPS_WIDTH_CELLS
    cdef int_t GPS_HEIGHT_CELLS
    cdef int_t LPS_WIDTH_CELLS
    cdef int_t LPS_HEIGHT_CELLS

    cdef int_t LPS_ORIGINx
    cdef int_t LPS_ORIGINy
    cdef int_t LPS_ORIGIN_CELLx
    cdef int_t LPS_ORIGIN_CELLy

    cdef int_t NUM_ROWS
    cdef int_t NUM_COLS

    cdef int_t MAX_RANGE
    cdef int_t SENSOR_FOV

    cdef int_t OCCUPIED
    cdef int_t UNOCCUPIED
    cdef int_t UNKNOWN

    cdef real_t max_occupied
    cdef real_t max_empty

cdef class Mapping:
    cdef public LPS, GPS, distance, sim_map
    cdef real_t fBotx, fBoty, fTheta
    cdef int_t debug

    def __init__(self, debug, bot_state=(0,0,0)):
        self.debug = debug

        # Declare LPS
        self.LPS = self.setArray(LPS_HEIGHT_CELLS, LPS_WIDTH_CELLS, np.float64)
        
        # Declare GPS
        self.GPS = self.setArray(GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, np.float64)

        # Declare self.distance
        self.distance = np.arange(SENSOR_FOV, dtype=np.int32)

        # Declare Sim_Map
        self.sim_map = self.setArray(GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, np.float64)
        cython_simulator.cy_buildMap(self.sim_map, GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, CELL_SIZE)

        # Declare current position
        self.fBotx, self.fBoty, self.fTheta = bot_state
        
    def setPosition(self, bot_state):
        self.fBotx, self.fBoty, self.fTheta = bot_state

    def setArray(self, rows, cols, type_of_number):
        temp = np.arange(rows*cols, dtype=type_of_number).reshape(cols, rows)
        np.ascontiguousarray(temp)
        temp[0:] = 0.5
        return temp

    def update_pos(self, x, y, theta):
        self.setPosition((x, y, theta))
        self.measureDistance(self.distance, x, y, theta)
        self.LPS[0:] = 0.5
        self.cy_detectHits(self.distance, self.LPS, theta)
        self.cy_updateFromLPS(x, y, theta, self.LPS, self.GPS)
     
    cdef void initializeLPS(self):
        self.LPS[0:] = 0.5

    cdef void measureDistance(self, np.ndarray[int, ndim=1, mode="c"] dist_arr, double x, double y, double theta):
        cdef real_t local_angle_rad, current_x, current_y
        cdef int_t angle, length, local_angle, grid_x, grid_y

        # if self.debug is True: print "Entering"
        
        for angle in xrange(SENSOR_FOV):
        # for angle in prange(SENSOR_FOV, nogil=True, num_threads=10):
            local_angle = <int_t>(theta + angle) % 360
            local_angle_rad = <real_t>(local_angle * M_PI / 180.0)

            for length in xrange(MAX_RANGE):
            # for length in prange(MAX_RANGE, num_threads=10):
                current_x = x + cos(local_angle_rad) * length
                current_y = y + sin(local_angle_rad) * length

                grid_x = lrint(ceil(current_x / CELL_SIZE))
                grid_y = lrint(ceil(current_y / CELL_SIZE))

                if grid_y >= WORLD_HEIGHT/CELL_SIZE or grid_y < 0 or grid_x >= WORLD_WIDTH/CELL_SIZE or grid_x < 0:
                    dist_arr[local_angle] = length
                    break
                elif self.sim_map[grid_y, grid_x] == 1:
                    dist_arr[local_angle] = length
                    break 
                elif length == MAX_RANGE-1:
                    dist_arr[local_angle] = MAX_RANGE-1

        if self.debug is True: print dist_arr

    cdef void cy_detectHits(self, np.ndarray[int, ndim=1, mode="c"] dist_arr, np.ndarray[double, ndim=2, mode='c'] LPS_arr, double theta):
        cdef int_t senseObstacle
        cdef int_t i, dist, arc, offx, offy
        cdef real_t hitx, hity

        cdef real_t rise, run, stepx, stepy, fcurrent_cell_x, fcurrent_cell_y
        cdef int_t steps, step, cell_hitx, cell_hity, current_cell_x, current_cell_y

        arc = 0
        offx = 0
        offy = 0

        for i in xrange(SENSOR_FOV):
            dist = dist_arr[i]
            if dist == MAX_RANGE - 1:
                senseObstacle = False
            else:
                senseObstacle = True
            hitx = cos(theta + i*M_PI/180) * dist + LPS_ORIGINx
            hity = sin(theta + i*M_PI/180) * dist + LPS_ORIGINy

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

            # if self.debug is True: print "Rise: ", rise, "run: ", run, "steps: ", steps, "stepx: ", stepx, "stepy: ", stepy, "current cells: ", current_cell_x, current_cell_y

            for step in xrange(steps):
                fcurrent_cell_x += stepx
                fcurrent_cell_y += stepy

                current_cell_x = lrint(ceil(fcurrent_cell_x))
                current_cell_y = lrint(ceil(fcurrent_cell_y))

                LPS_arr[current_cell_y, current_cell_x] = UNOCCUPIED

            if senseObstacle is True:
                LPS_arr[cell_hity, cell_hitx] = OCCUPIED
            elif senseObstacle is False:
                LPS_arr[cell_hity, cell_hitx] = UNOCCUPIED

    cdef void cy_updateFromLPS(self, double x, double y, double theta, np.ndarray[double, ndim=2, mode='c'] LPS_arr, np.ndarray[double, ndim=2, mode='c'] GPS_arr):
        
        # Declare boundary variables
        cdef int_t LPS_lower_boundsX, LPS_lower_boundsY, LPS_upper_boundsX, LPS_upper_boundsY
        cdef int_t GPS_lower_boundsX, GPS_lower_boundsY, GPS_upper_boundsX, GPS_upper_boundsY
        cdef int_t Lower_boundaryFlagX, Lower_boundaryFlagY, Upper_boundaryFlagX, Upper_boundaryFlagY

        # Declare bot location variables as integers
        cdef int_t botx, bot_y

        cdef int_t active_rows, active_cols

        cdef int_t GPS_skip, LPS_skip 
        cdef int_t GPSidx, LPSidx

        # if self.debug is True:
        #     print "origx, y: ", self.fBotx, self.fBoty
        #     print "GPS_WIDTH_CELLS, HEIGHT: ", GPS_WIDTH_CELLS, GPS_HEIGHT_CELLS

        botx = lrint(x/CELL_SIZE)
        boty = lrint(y/CELL_SIZE)

        # Check bot location values
        if botx < 0 or botx > GPS_WIDTH_CELLS:
            # if self.debug is True: print "Invalid x location"
            return

        if boty < 0 or boty > GPS_HEIGHT_CELLS:
            # if self.debug is True: print "Invalid y location"
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

        # if self.debug is True:
        #     print "Bot Location:", botx, boty
        #     print "NumCols: ", GPS_WIDTH_CELLS, "NumRows: ", GPS_HEIGHT_CELLS
        #     print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY, "GPS_upper_boundsX: ", GPS_upper_boundsX, "GPS_upper_boundsY: ", GPS_upper_boundsY
        #     print "LPS_lower_boundsX: ", LPS_lower_boundsX, "LPS_lower_boundsY: ", LPS_lower_boundsY, "LPS_upper_boundsX: ", LPS_upper_boundsX, "LPS_upper_boundsY: ", LPS_upper_boundsY
        #     print "LowerBoundaryFlagX: ", Lower_boundaryFlagX, "LowerBoundaryFlagY: ", Lower_boundaryFlagY
        #     print "UpperBoundaryFlagX: ", Upper_boundaryFlagX, "UpperBoundaryFlagY: ", Upper_boundaryFlagY

        if Lower_boundaryFlagX < 0:
            LPS_lower_boundsX = abs(Lower_boundaryFlagX)
            GPS_lower_boundsX = 0
            # if self.debug is True: print "Lower_boundaryFlagX hit"

        if Lower_boundaryFlagY < 0:
            LPS_lower_boundsY = abs(Lower_boundaryFlagY)
            GPS_lower_boundsY = 0
            # if self.debug is True: print "Lower_boundaryFlagY hit"

        if Upper_boundaryFlagX > GPS_WIDTH_CELLS - 1:
            LPS_upper_boundsX = GPS_WIDTH_CELLS - 1 - botx + LPS_ORIGIN_CELLx
            GPS_upper_boundsX = GPS_WIDTH_CELLS - 1
            # if self.debug is True: print "Upper_boundaryFlagX hit"

        if Upper_boundaryFlagY > GPS_HEIGHT_CELLS - 1:
            LPS_upper_boundsY = GPS_HEIGHT_CELLS - 1 - boty + LPS_ORIGIN_CELLy
            GPS_upper_boundsY = GPS_HEIGHT_CELLS - 1
            # if self.debug is True: print "Upper_boundaryFlagY hit"

        # if self.debug is True:
        #     print "------------After Boundary Check-----------"
        #     print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY
        #     print "GPS_upper_boundsX: ", GPS_upper_boundsX, "GPS_upper_boundsY: ", GPS_upper_boundsY

        active_rows = LPS_upper_boundsY - LPS_lower_boundsY + 1
        active_cols = LPS_upper_boundsX - LPS_lower_boundsX + 1

        GPSidx = GPS_lower_boundsX + (GPS_lower_boundsY * GPS_WIDTH_CELLS)
        LPSidx = LPS_lower_boundsX + (LPS_lower_boundsY * LPS_WIDTH_CELLS)

        GPS_skip = GPS_WIDTH_CELLS + active_cols
        LPS_skip = GPS_HEIGHT_CELLS + active_rows

        # if self.debug is True:
        #     print "---------------------------------------------"
        #     print "active_rows: ", active_rows, "active_cols: ", active_cols
        #     print "GPS_skip: ", GPS_skip, "GPSidx: ", GPSidx
        #     print "LPS_skip: ", LPS_skip, "LPSidx: ", LPSidx

        cdef int_t GPSx, GPSy, LPSx, LPSy, row, col

        GPSx = GPS_lower_boundsX
        GPSy = GPS_lower_boundsY

        LPSx = LPS_lower_boundsX
        LPSy = LPS_lower_boundsY

        for row in range(active_rows):
            for col in range(active_cols):
                if LPS_arr[LPSy, LPSx] == 0.5:
                    GPS_arr[GPSy, GPSx] = GPS_arr[GPSy, GPSx]
                else:
                    GPS_arr[GPSy, GPSx] = self.cy_getProb(GPS_arr[GPSy,GPSx], lrint(LPS_arr[LPSy,LPSx]))

                GPSx += 1
                LPSx += 1
            GPSy += 1
            LPSy += 1
            GPSx = GPS_lower_boundsX
            LPSx = LPS_lower_boundsX

        GPS_arr[boty, botx] = 2

    cdef real_t cy_getProb(self, double prior_occ, int obstacle_is_sensed):
        cdef real_t POcc, PEmp, inv_prior, new_prob

        inv_prior = 1.0 - prior_occ

        POcc = max_occupied
        PEmp = max_empty
        new_prob = prior_occ

        if obstacle_is_sensed == 1:
            new_prob = (POcc * prior_occ) / ((POcc * prior_occ) + (PEmp * inv_prior))
        elif obstacle_is_sensed == 0:
            new_prob = (PEmp * prior_occ) / ((PEmp * prior_occ) + (POcc * inv_prior))

        return new_prob

def main():
    cdef int i
    stuff = Mapping(False, (1000, 1000, 0))
    np.set_printoptions(linewidth=600, threshold='nan', precision=2, suppress=True)
    
    before = time.clock()
    stuff.update_pos(1000+i*200, 1000+i*200, 0)
    print time.clock() - before
    # print stuff.LPS, stuff.GPS

    # print stuff.LPS
    print stuff.GPS