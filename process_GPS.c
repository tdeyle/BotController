#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "process_GPS.h"
#include "array_parameters.h"

/* 
	File will hold methods to process GPS array given the LPS array, robot Position and Angle.
	- Have a shared file with common methods for both the LPS and GPS:
		-> setting grid location (0 or 1 for LPS, 0 to 100 for GPS)
*/

void updateFromLPS(double *LPS_arr, double *GPS_arr, double origx, double origy, double theta)
{
	// Declare and find LPS_lower_bounds, LPS_upper_bounds, GPS_lower_bounds, GPS_upper_bounds; for x and y
	int LPS_lower_boundsX, LPS_lower_boundsY, LPS_upper_boundsX, LPS_upper_boundsY;
	int GPS_lower_boundsX, GPS_lower_boundsY, GPS_upper_boundsX, GPS_upper_boundsY;
	int Lower_boundary_flagX, Lower_boundary_flagY, Upper_boundary_flagX, Upper_boundary_flagY;
	int botx, boty;

	int active_rows, active_cols;

	int GPS_skip, LPS_skip;

	int GPSidx, LPSidx;

	botx = (int)(origx);
	boty = (int)(origy);

	if (botx < 0 || botx > GPS_WIDTH){
		printf("Invalid X location\n");
		return;
	}

	if (boty < 0 || boty > GPS_HEIGHT){
		printf("Invalid Y Location\n");
		return;
	}

	// Setup default boundary values
	LPS_lower_boundsX = 0;
	LPS_lower_boundsY = 0;
	GPS_lower_boundsX = botx - LPS_ORIGINx;
	GPS_lower_boundsY = boty - LPS_ORIGINy;

	LPS_upper_boundsX = LPS_WIDTH - 1;
	LPS_upper_boundsY = LPS_HEIGHT - 1;
	GPS_upper_boundsX = botx + LPS_ORIGINx - 1; //Changed: Added the -1
	GPS_upper_boundsY = boty + LPS_ORIGINy - 1;

	// Check to make sure that the LPS is not going past the limits of the GPS in either the x/y direction 
	// at the upper and lower extremeties
	Lower_boundary_flagX = botx - LPS_ORIGINx;
	Lower_boundary_flagY = boty - LPS_ORIGINy;
	Upper_boundary_flagX = botx + LPS_ORIGINx;
	Upper_boundary_flagY = boty + LPS_ORIGINy;

	// DEBUG
	printf("Bot Location: %i, %i\n", botx, boty);
	printf("NumCols: %i, NumRows: %i\n", GPS_WIDTH, GPS_HEIGHT);
	printf("GPS_lower_boundsX: %i, GPS_lower_boundsY: %i, GPS_higher_boundsX: %i, GPS_higher_boundsY: %i\n", GPS_lower_boundsX, GPS_lower_boundsY, GPS_upper_boundsX, GPS_upper_boundsY);
	printf("LPS_lower_boundsX: %i, LPS_lower_boundsY: %i, LPS_higher_boundsX: %i, LPS_higher_boundsY: %i\n", LPS_lower_boundsX, LPS_lower_boundsY, LPS_upper_boundsX, LPS_upper_boundsY);
	printf("LowerBoundaryFlagX: %i, LowerBoundaryFlagY: %i\n", Lower_boundary_flagX, Lower_boundary_flagY);
	printf("UpperBoundaryFlagX: %i, UpperBoundaryFlagY: %i\n", Upper_boundary_flagX, Upper_boundary_flagY);
	
	// Perform a BoundsCheck on the upper and lower X and Y coords, adjusting the LPS and GPS
	if (Lower_boundary_flagX < 0){
		LPS_lower_boundsX = abs(Lower_boundary_flagX);
		GPS_lower_boundsX = 0;
		printf("Lower_boundary_flagX hit");
	}

	if (Lower_boundary_flagY < 0){
		LPS_lower_boundsY = abs(Lower_boundary_flagX);
		GPS_lower_boundsY = 0;
		printf("Lower_boundary_flagY hit");
	}

	if (Upper_boundary_flagX > GPS_WIDTH - 1){
		LPS_upper_boundsX = GPS_WIDTH - 1 - botx + LPS_ORIGINx;
		GPS_upper_boundsX = GPS_WIDTH - 1;
		printf("Upper_boundary_flagX hit");
	}

	if (Upper_boundary_flagY > GPS_HEIGHT - 1){
		LPS_upper_boundsY = GPS_HEIGHT - 1 - boty + LPS_ORIGINy;
		GPS_upper_boundsY = GPS_HEIGHT - 1;
		printf("Upper_boundary_flagY hit");
	}

	// DEBUG
	printf("\n--------------After---------------\n");
	printf("GPS_lower_boundsX: %i, GPS_lower_boundsY: %i, GPS_higher_boundsX: %i, GPS_higher_boundsY: %i\n", GPS_lower_boundsX, GPS_lower_boundsY, GPS_upper_boundsX, GPS_upper_boundsY);
	printf("LPS_lower_boundsX: %i, LPS_lower_boundsY: %i, LPS_higher_boundsX: %i, LPS_higher_boundsY: %i\n", LPS_lower_boundsX, LPS_lower_boundsY, LPS_upper_boundsX, LPS_upper_boundsY);
	
	active_rows = LPS_upper_boundsY - LPS_lower_boundsY + 1;
	active_cols = LPS_upper_boundsX - LPS_lower_boundsX + 1;

	GPSidx = GPS_lower_boundsX + (GPS_lower_boundsY * GPS_WIDTH);
	LPSidx = LPS_lower_boundsX + (LPS_lower_boundsY * LPS_WIDTH);

	GPS_skip = GPS_WIDTH - active_cols; // Here is the issue. Are the width and height switched out??
	LPS_skip = LPS_WIDTH - active_cols;

	// DEBUG
	printf("\n---------------------------------------------------------\n");
	// printf("LPS_row_cells: %i, LPS_col_cells: %i\n", LPS_row_cells, LPS_col_cells);
	printf("active_rows: %i, active_cols: %i\n", active_rows, active_cols);
	// printf("GPS_row_cells: %i, GPS_col_cells: %i\n", GPS_row_cells, GPS_col_cells);
	printf("GPSskip: %i, GPSidx: %i\n", GPS_skip, GPSidx);
	printf("LPSskip: %i, LPSidx: %i\n", LPS_skip, LPSidx);

	int x, y;

	for (y=0;  y<active_rows; y++){
		for (x=0; x<active_cols; x++){
			if (LPS_arr[LPSidx] == 0.5){
				GPS_arr[GPSidx] = GPS_arr[GPSidx];
			}else{
				GPS_arr[GPSidx] = getProb(GPS_arr[GPSidx], (int)(LPS_arr[LPSidx]));
			}

			// DEBUG
			// printf("With LPS_arr[%i] == %.2f, GPS_arr[%i] = %.2f\n", LPSidx, LPS_arr[LPSidx], GPSidx, GPS_arr[GPSidx]);
			
			GPSidx += 1;
			LPSidx += 1;
		}
		GPSidx += GPS_skip;
		LPSidx += LPS_skip;
	}
}

double getProb(double prior_occ, int obstacle_is_sensed)
{
	double POcc, PEmp, inv_prior, new_prob;

	inv_prior = 1.0 - prior_occ;

	POcc = max_occupied;
	PEmp = max_empty;
	new_prob = prior_occ;

	if (obstacle_is_sensed == 1){
		// DEBUG
		// printf("	%.2f Sensed\n", POcc);
		new_prob = (POcc * prior_occ) / ((POcc * prior_occ) + (PEmp * inv_prior));
	}else if (obstacle_is_sensed == 0){
		// DEBUG
		// printf("	%.2f Not Sensed\n", PEmp);
		new_prob = (PEmp * prior_occ) / ((PEmp * prior_occ) + (POcc * inv_prior));
	}

	return new_prob;
	
}