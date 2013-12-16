#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "process_array.h"
#include "simulator.h"
#include "array_parameters.h"

void createMap(double *sim_map_arr, int num_row, int num_col, int cell_size)
{
	// Given an array address, create a map with the num_row by num_col, with a cell_size.
	// Example: 500 x 500 grid with a grid size of 50mm creates a map of 10x10 cells in an 
	// array.
	// TODO: Try to tie in the setArray method with the one that is the process array module.
	
	int row, col, idx;
	setArray(sim_map_arr, num_row, num_col, 0.0);
	idx = 0;

	// Assign a simple outer wall as 1.0
	for (row=0; row<num_row; row++){
		for (col=0; col<num_col; col++){
			if (row == 0 || row == (num_row - 1) || col == 0 || col == (num_col - 1)){
				idx = row * num_col + col;
				sim_map_arr[idx] = 1.0;
			}
			idx++;
		}
	}
}

void scanDistance(double *dist_arr, double *sim_map_arr, double bot_x, double bot_y, double theta)
{
	// Given a distance array address, the location of the robot and its orientation,
	// and the simulated map array, find the distances between the robot and either
	// obstacles found, or the max distance of the sensor that is measuring.
	double local_angle_rad, current_x, current_y;
	int x, y;
	int angle, length, local_angle, grid_x, grid_y;

	x = (int)(bot_x);
	y = (int)(bot_y);

	printf("Entering\n");

	for (angle=0; angle<SENSOR_FOV; angle++){
		local_angle = (int)(theta + angle) % 360;
		// printf("local_angle = %i\n", local_angle);

		local_angle_rad = (double)(local_angle * M_PI / 180.0);
		// printf("local_angle_rad = %.5f\n", local_angle_rad);

		for (length=0; length<MAX_RANGE; length++){
			current_x = x + cos(local_angle_rad) * length;
			current_y = y + sin(local_angle_rad) * length;

			grid_x = (int)(current_x / CELL_SIZE);
			grid_y = (int)(current_y / CELL_SIZE);

			// printf("current: %i, %i, grid: %i, %i\n", current_x, current_y, grid_x, grid_y);
			if (grid_y >= WORLD_HEIGHT / CELL_SIZE || grid_y < 0 || grid_x >= WORLD_WIDTH / CELL_SIZE || grid_x < 0){
				dist_arr[local_angle] = length;
				// printf("out of bounds @ %i\n", local_angle);
				break;
			}
			else if (sim_map_arr[grid_x * NUM_COLS + grid_y] == 1.0){
				dist_arr[local_angle] = length;
				// printf("Hit @ %i, with %i\n", local_angle, length);
				break;
			}
			else if (length == MAX_RANGE - 1){
				dist_arr[local_angle] = MAX_RANGE - 1;
				// printf("On forever @ %i with %i\n", local_angle, length);
			}
		}
	}
	printf("Finished\n");
	
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