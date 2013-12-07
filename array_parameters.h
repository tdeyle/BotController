// array_parameters.h
// - Holds all parameters for array processing

#define LPS_ORIGINx (int) LPS_WIDTH/2 	// TODO: Add to a LPS settings header file
#define LPS_ORIGINy (int) LPS_HEIGHT/2
#define LPS_WIDTH 20
#define LPS_HEIGHT 20

// TODO: Add a UI settings header file
#define WORLD_WIDTH 12000	// in millimeters
#define WORLD_HEIGHT 6000	// in millimeters
#define SCREEN_WIDTH 1200	// in px
#define SCREEN_HEIGHT 600	// in px
#define CELL_SIZE 240 		// in millimeters

#define GPS_WIDTH (WORLD_WIDTH/CELL_SIZE)
#define GPS_HEIGHT (WORLD_HEIGHT/CELL_SIZE)

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
#define MAX_RANGE 10

#define UNKNOWN 0.5
#define OCCUPIED 1.0
#define UNOCCUPIED 0.0

#define TRUE 1
#define FALSE 0

#define max_occupied 0.98
#define max_empty 0.05