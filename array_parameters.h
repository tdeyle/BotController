// array_parameters.h
// - Holds all parameters for array processing

// TODO: Add a UI settings header file
#define WORLD_WIDTH 12000	// in millimeters
#define WORLD_HEIGHT 12000	// in millimeters
#define SCREEN_WIDTH 1200	// in px
#define SCREEN_HEIGHT 600	// in px
#define CELL_SIZE 200 		// in millimeters

#define LPS_WIDTH (MAX_RANGE * 2 + CELL_SIZE)
#define LPS_HEIGHT (MAX_RANGE * 2 + CELL_SIZE)
#define LPS_ORIGINx (int)(LPS_WIDTH/2) 
#define LPS_ORIGINy (int)(LPS_HEIGHT/2)
#define LPS_ORIGIN_CELLx (LPS_WIDTH_CELLS/2)
#define LPS_ORIGIN_CELLy (LPS_HEIGHT_CELLS/2)
#define LPS_WIDTH_CELLS (int)(LPS_WIDTH/CELL_SIZE)
#define LPS_HEIGHT_CELLS (int)(LPS_HEIGHT/CELL_SIZE) 

#define GPS_WIDTH_CELLS (WORLD_WIDTH/CELL_SIZE)
#define GPS_HEIGHT_CELLS (WORLD_HEIGHT/CELL_SIZE)
#define GPS_WIDTH WORLD_WIDTH
#define GPS_HEIGHT WORLD_HEIGHT

#define NUM_ROWS (WORLD_WIDTH/CELL_SIZE)
#define NUM_COLS (WORLD_HEIGHT/CELL_SIZE)

#define worldToScreenx (WORLD_WIDTH/SCREEN_WIDTH)	// TODO: Define these constants in a Grid Utilities header file
#define worldToScreeny (WORLD_HEIGHT/SCREEN_HEIGHT)

#define worldToGridx (WORLD_WIDTH/CELL_SIZE)
#define worldToGridy (WORLD_HEIGHT/CELL_SIZE)

#define screenToWorldx (SCREEN_WIDTH/WORLD_WIDTH)
#define screenToWorldy (SCREEN_HEIGHT/WORLD_HEIGHT)

#define gridToWorldx (1.0/worldToGridx)
#define gridToWorldy (1.0/worldToGridy)

// process_arrays parameters
#define SENSOR_FOV 360
#define MAX_RANGE 5000

#define UNKNOWN 0.5
#define OCCUPIED 1.0
#define UNOCCUPIED 0.0

#define TRUE 1
#define FALSE 0

#define max_occupied 1.0
#define max_empty 0.0