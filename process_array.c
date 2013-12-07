#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "process_array.h"
#include "array_parameters.h"
#include "process_GPS.h"

void initialize(double *GPS_arr)
{
	setArray(GPS_arr, GPS_HEIGHT, GPS_WIDTH, 0.5);
}

void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double origx, double origy, double theta)
{
	// Take an array (LPS_arr) with an array of distances (dist_arr)
	// Initialize the LPS_arr with 0.5 (unknown) occupancies
	// Detect hits on the cells within the LPS based on the distances given in the dist_arr.
	// Assign 1.0 (Occupied) or 0.0 (Unoccupied) values to the cells
	
	setArray(LPS_arr, LPS_HEIGHT, LPS_WIDTH, 0.5);
	
	detectHits(LPS_arr, dist_arr, theta, LPS_ORIGINx, LPS_ORIGINy);

	updateFromLPS(LPS_arr, GPS_arr, origx, origy, theta);
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

void detectHits(double *LPS_arr, int *dist_arr, double theta, double origx, double origy)
{
	int senseObstacle;
	int i, dist, arc, offx, offy;
	double hitx, hity;

	arc = 0;
	offx = 0;
	offy = 0;

	for (i=0; i<SENSOR_FOV; i++){
		dist = dist_arr[i];
		if (dist < MAX_RANGE)
			senseObstacle = TRUE;
		else
			senseObstacle = FALSE;
		hitx = cos(i*M_PI/180) * dist + origx;
		hity = sin(i*M_PI/180) * dist + origy;

		assignOccupancy(LPS_arr, offx, offy, hitx, hity, origx, origy, arc, senseObstacle);
	}
}

void assignOccupancy(double *LPS_arr, int offx, int offy, double hitx, double hity, double origx, double origy, int arc, int senseObstacle)
{
	double rise, run, stepx, stepy, currx, curry;
	int steps, step;
	double colScaleMM, rowScaleMM;

	colScaleMM = 1.0;
	rowScaleMM = 1.0;

	rise = hity - origy;
	if (fabs(rise) < 0.1){
		rise = 0.0;
	}

	run = hitx - origx;
	if (fabs(run) < 0.1){
		run = 0.0;
	}

	steps = lrint(lround(fmax(fabs(rise/rowScaleMM), fabs(run/colScaleMM))));

	if(steps == 0){
		LPS_arr[lrint(origy/colScaleMM) * LPS_WIDTH + lrint(origx/colScaleMM)] = OCCUPIED;
		return;
	}

	stepx = run / steps;

	if (fabs(stepx) > colScaleMM){
		stepx = colScaleMM;
		if (run < 0){
			stepx *= -1;
		}
	}

	stepy = rise / steps;

	if (fabs(stepy) > rowScaleMM){
		stepy = rowScaleMM;
		if (run < 0){
			stepy *= -1;
		}
	}

	currx = origx;
	curry = origy;

	LPS_arr[lrint(curry/rowScaleMM) * LPS_WIDTH + lrint(currx/colScaleMM)] = UNOCCUPIED;
		

	for (step=0; step<steps; step++){
		currx += stepx;
		curry += stepy;

		LPS_arr[lrint(curry/rowScaleMM) * LPS_WIDTH + lrint(currx/colScaleMM)] = UNOCCUPIED;
			
		if (senseObstacle == TRUE){
			LPS_arr[lrint(hity/rowScaleMM)*LPS_WIDTH + lrint(hitx/colScaleMM)] = OCCUPIED;
		}
		else{
			LPS_arr[lrint(hity/rowScaleMM)*LPS_WIDTH + lrint(hitx/colScaleMM)] = UNOCCUPIED;
		}
	}
}