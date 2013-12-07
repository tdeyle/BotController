void initialize(double *GPS_arr);
void process(double *GPS_arr, double *LPS_arr, int *dist_arr, double origx, double origy, double theta);
void setArray(double *arr, int rows, int cols, double value);
void detectHits(double *LPS_arr, int *dist_arr, double theta, double origx, double origy);
void assignOccupancy(double *LPS_arr, int offx, int offy, double hitx, double hity, double origx, double origy, int arc, int senseObstacle);