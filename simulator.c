#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "process_array.h"
#include "simulator.h"

void createMap(double *sim_map_arr, int num_row, int num_col, int cell_size)
{
	// Given an array address, create a map with the num_row by num_col, with a cell_size.
	// Example: 500 x 500 grid with a grid size of 50mm creates a map of 10x10 cells in an 
	// array.
	
	int row, col, idx;
	setArray(sim_map_arr, num_row, num_col, 0.0);

	// Assign a simple outer wall as 1.0
	for (row=0; row<num_row; row++){
		for (col=0; col<num_col; col++){
			if (row == 0 || row == (num_row - 1) || col == 0 || col == (num_col - 1)){
				idx = row * num_col + col;
				sim_map_arr[idx] = 1.0;
			}
			
		}
	}
}

void scanDistance(double *dist_arr, double bot_x, double bot_y, double theta)
{
	return 0;
}

void setArray(double *arr, int rows, int cols, double value)
{
	int i, j;
	int index = 0;

	for (i=0; i<rows; i++){
		for (j=0; j<cols; j++){
			arr[index] = value;
			index++;
		}
	}
}