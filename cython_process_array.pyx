import cython
import numpy as np
cimport numpy as np

cdef extern from "process_array.h":
	void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double origx, double origy, double theta)
	void initialize(double *GPS_arr)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_processArray(np.ndarray[double, ndim=2, mode="c"] input_GPS, np.ndarray[double, ndim=2, mode="c"] input, np.ndarray[int, ndim=1, mode="c"] dist, double origx, double origy, double theta):
	process(&input_GPS[0,0], &input[0,0], &dist[0], origx, origy, theta)

def cy_initArray(np.ndarray[double, ndim=2, mode="c"] input_GPS):
	initialize(&input_GPS[0,0])