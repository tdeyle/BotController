import time
from math import sin, cos
# from scan import *
import numpy as np
import cython_process_array
import cython_simulator

def python_list_making():
    pyarr = [[0 for x in xrange(20)] for y in xrange(20)] 

    for x in range(20):
        for y in range(20):
            pyarr[x][y] = 0

    # print len(pyarr), len(pyarr[0]), pyarr[0]
    
    return pyarr

def print_python_list(pyarr):
    for x in range(20):
        for y in range(20):
            print pyarr[x][y]

def python_set_cell(cell, row, col, value):
    cell[int(row)][int(col)] = value

def detectHits(LPS, distance, theta, x, y):
    offx = 0
    offy = 0
    arc = 0
    dist = 0

    for i in range(360):
        dist = distance[i]
        
        if dist < 5:
            senseObstacle = True
        else:
            senseObstacle = False
        hitx = cos(theta) * dist
        hity = sin(theta) * dist

        computeOccupancy(LPS, offx, offy, hitx, hity, x, y, arc, senseObstacle, i)

def computeOccupancy(LPS, offx, offy, hitx, hity, origx, origy, arc, senseObstacle, i):
    colScaleMM = 1.0
    rowScaleMM = 1.0

    rise = hity - origy
    if abs(rise) < 0.1: rise = 0
    run = hitx - origx
    if abs(run) < 0.1: run = 0

    steps = int(round(max(abs(rise/rowScaleMM), abs(run/colScaleMM))))

    if steps == 0:
        python_set_cell(LPS, hitx, hity, 1.0)
        return

    stepx = run / float(steps)

    if abs(stepx) > colScaleMM:
        stepx = colScaleMM
        if run < 0:
            stepx *= -1

    stepy = rise / float(steps)

    if abs(stepy) > rowScaleMM:
        stepy = colScaleMM
        if rise < 0.0:
            stepy *= -1.0

    currx = origx
    curry = origy

    for step in range(steps):
        curry += stepy
        currx += stepx
        python_set_cell(LPS, currx, curry, 0.0)
        if senseObstacle:
            python_set_cell(LPS, hitx, hity, 1.0)
        else:
            python_set_cell(LPS, hitx, hity, 0.0)

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

    #----------------------------------------
    # Python Implementation
    #----------------------------------------
    before_py = time.clock()

    L_SONAR_distance = []
    C_SONAR_distance = []
    R_SONAR_distance = []
    for i in range(360):
        L_SONAR_distance.append(3)
        C_SONAR_distance.append(3)
        R_SONAR_distance.append(3)
    
    a = python_list_making() # Add variables to pass for rows and columns
    b = python_list_making() # Add variables to pass for rows and columns
    c = python_list_making() # Add variables to pass for rows and columns
    
    detectHits(a, L_SONAR_distance, 0, 10, 10)
    detectHits(b, C_SONAR_distance, 0, 10, 10)
    detectHits(c, R_SONAR_distance, 0, 10, 10)
    
    after_py = time.clock()

    #---------------------------------------
    # Cython Implementation with 'C'
    #---------------------------------------
    before_cy = time.clock()

    Left = np.arange(LPS_WIDTH*LPS_HEIGHT, dtype=np.float64).reshape((LPS_HEIGHT,LPS_WIDTH))
    # Center = np.arange(20*20, dtype=np.float64).reshape((20,20))
    # Right = np.arange(20*20, dtype=np.float64).reshape((20,20))
    GPS = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_HEIGHT,GPS_WIDTH))
    cython_process_array.cy_initArray(GPS)

    b = np.arange(360, dtype=np.int32)
    b[0:360] = 8

    # print b

    # Add assigning values to the GPS and LPS arrays here.... GPS[0:...] = 0.5

    cython_process_array.cy_processArray(GPS, Left, b, 48.0, 11.0, 0.0)
    np.set_printoptions(linewidth=350, threshold='nan', precision=2)
    # cython_process_array.cy_processArray(GPS, Center, b, 0.0, 0.0, 0.0)
    # cython_process_array.cy_processArray(GPS, Right, b, 0.0, 0.0, 0.0)

    after_cy = time.clock()

    #-------------------------------------
    # Printing results
    #-------------------------------------
    print "py array", after_py-before_py
    print "cy array", after_cy-before_cy

    print Left
    print ""
    print GPS

    # sim_map = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_WIDTH,GPS_HEIGHT))

    # cython_simulator.cy_buildMap(sim_map, NUM_ROWS, NUM_COLS, CELL_SIZE)
    # print sim_map