#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "process_array.h"
#include "array_parameters.h"
#include "process_GPS.h"

void initialize(double *GPS_arr)
{
	setArray(GPS_arr, GPS_HEIGHT_CELLS, GPS_WIDTH_CELLS, 0.5);
}

void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double botx, double boty, double theta)
{
	// Take an array (LPS_arr) with an array of distances (dist_arr)
	// Initialize the LPS_arr with 0.5 (unknown) occupancies
	// Detect hits on the cells within the LPS based on the distances given in the dist_arr.
	// Assign 1.0 (Occupied) or 0.0 (Unoccupied) values to the cells
	
	// setArray(LPS_arr, LPS_HEIGHT_CELLS, LPS_WIDTH_CELLS, 0.5);
	
	// detectHits(LPS_arr, dist_arr, theta, LPS_ORIGINx*CELL_SIZE, LPS_ORIGINy*CELL_SIZE);

	// updateFromLPS(LPS_arr, GPS_arr, botx, boty, theta);
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

void detectHits(double *LPS_arr, int *dist_arr, double theta)
{
	int senseObstacle;
	int i, dist, arc, offx, offy;
	double hitx, hity;

	printf("Hi, inside DetectHits\n");

	arc = 0;
	offx = 0;
	offy = 0;

	for (i=0; i<SENSOR_FOV; i++){
		dist = dist_arr[i];
		if (dist < MAX_RANGE-1)
			senseObstacle = TRUE;
		else
			senseObstacle = FALSE;
		hitx = cos(i*M_PI/180) * dist + LPS_ORIGINx;
		hity = sin(i*M_PI/180) * dist + LPS_ORIGINy;

		// printf("%.2f, %.2f\n", hitx, hity);

		assignOccupancy(LPS_arr, offx, offy, hitx, hity, arc, senseObstacle);
	}
}

void assignOccupancy(double *LPS_arr, int offx, int offy, double hitx, double hity, int arc, int senseObstacle)
{
	double rise, run, stepx, stepy, fcurrent_cell_x, fcurrent_cell_y;
    int steps, step, cell_hitx, cell_hity, current_cell_x, current_cell_y;

	rise = (hity - LPS_ORIGINy) / CELL_SIZE;
    if (fabs(rise) < 0.1){
        rise = 0.0;
    }

    run = (hitx - LPS_ORIGINx) / CELL_SIZE;
    if (fabs(run) < 0.1){
        run = 0.0;
    }

    steps = lrint(lround(fmax(fabs(rise), fabs(run))));

    stepx = run/steps;
    stepy = rise/steps;

    if (fabs(stepx) > CELL_SIZE){
        stepx = CELL_SIZE;
        if (run < 0){
            stepx *= -1;
        }
    }

    if (fabs(stepy) > CELL_SIZE){
        stepy = CELL_SIZE;
        if (rise < 0){
            stepy *= -1;
        }
    }

    fcurrent_cell_x = (LPS_ORIGINx / CELL_SIZE);
	fcurrent_cell_y = (LPS_ORIGINy / CELL_SIZE);

    current_cell_x = lrint(fcurrent_cell_x);
    current_cell_y = lrint(fcurrent_cell_y);

    LPS_arr[current_cell_y * LPS_WIDTH_CELLS + current_cell_x] = 2; //UNOCCUPIED
    
    cell_hitx = lrint(hitx/CELL_SIZE);
    cell_hity = lrint(hity/CELL_SIZE);

    for (step=0; step<steps; step++){
        fcurrent_cell_x += stepx;
        fcurrent_cell_y += stepy;

        current_cell_x = lrint(fcurrent_cell_x);
        current_cell_y = lrint(fcurrent_cell_y);

        LPS_arr[current_cell_y * LPS_WIDTH_CELLS + current_cell_x] = UNOCCUPIED;

        if (senseObstacle == TRUE){
            LPS_arr[cell_hity * LPS_WIDTH_CELLS + cell_hitx] = OCCUPIED;
        }else{
            LPS_arr[cell_hity * LPS_WIDTH_CELLS + cell_hitx] = UNOCCUPIED;
        }
    }
}