import cython
import numpy as np
cimport numpy as np

cdef extern from "simulator.h":
    void createMap(double *sim_map_arr, int num_row, int num_cols, int cell_size)
    void scanDistance(double *dist_arr, double *sim_map_arr, double bot_x, double bot_y, double theta)
    
def cy_buildMap(np.ndarray[double, ndim=2, mode="c"] input_sim_map, int num_rows, int num_cols, int cell_size):
    """
    Builds a map that has x rows and y columns with cell size
    """
    createMap(&input_sim_map[0,0], num_rows, num_cols, cell_size)

def cy_scanDistance(np.ndarray[double, ndim=1, mode="c"] input_dist, np.ndarray[double, ndim=2, mode="c"] input_sim_map, int x, int y, int theta):
    """
    Takes a measurement between the xy coords of the robot and obstacles around the robot in 1 degree increments and returns an array
    """
    scanDistance(&input_dist[0], &input_sim_map[0,0], x, y, theta)
