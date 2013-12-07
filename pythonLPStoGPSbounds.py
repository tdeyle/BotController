import numpy as np

GPS_WIDTH, GPS_HEIGHT = (20,20)
LPS_WIDTH, LPS_HEIGHT = (7,7)

GPS = np.arange(GPS_WIDTH*GPS_HEIGHT, dtype=np.float64).reshape((GPS_WIDTH,GPS_HEIGHT))
LPS = np.arange(LPS_WIDTH*LPS_HEIGHT, dtype=np.float64).reshape((LPS_WIDTH,LPS_HEIGHT))

GPS[0:GPS_WIDTH*GPS_HEIGHT] = 0.5
LPS[0:LPS_WIDTH*LPS_HEIGHT] = 0.0

LPS_ORIGINx = LPS_WIDTH/2
LPS_ORIGINy = LPS_HEIGHT/2

botX, botY = (17,10)

LPS_lower_boundsX, LPS_lower_boundsY = (0,0)
GPS_lower_boundsX, GPS_lower_boundsY = (botX - LPS_ORIGINx
, botY - LPS_ORIGINy)
LPS_higher_boundsX, LPS_higher_boundsY = (LPS_WIDTH-1, LPS_HEIGHT-1)
GPS_higher_boundsX, GPS_higher_boundsY = (botX + LPS_ORIGINx
, botY + LPS_ORIGINy	)

Lower_boundary_flagX = botX - LPS_ORIGINx
Lower_boundary_flagY = botY - LPS_ORIGINy

Upper_boundary_flagX = botX + LPS_ORIGINx
Upper_boundary_flagY = botY + LPS_ORIGINy

print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY, "GPS_higher_boundsX: ", GPS_higher_boundsX, "GPS_higher_boundsY: ", GPS_higher_boundsY
print "LPS_lower_boundsX: ", LPS_lower_boundsX, "LPS_lower_boundsY: ", LPS_lower_boundsY, "LPS_higher_boundsX: ", LPS_higher_boundsX, "LPS_higher_boundsY: ", LPS_higher_boundsY

print "Lower_boundary_flagX: ", Lower_boundary_flagX, "Lower_boundary_flagY: ", Lower_boundary_flagY
print "Upper_boundary_flagX: ", Upper_boundary_flagX, "Upper_boundary_flagY: ", Upper_boundary_flagY

if Lower_boundary_flagX < 0: 
	LPS_lower_boundsX = abs(Lower_boundary_flagX)
	GPS_lower_boundsX = 0
	print "Lower_boundary_flagX hit"
else:
	GPS_lower_boundsX = botX - LPS_ORIGINx


if Lower_boundary_flagY < 0:
	LPS_lower_boundsY = abs(Lower_boundary_flagY)
	GPS_lower_boundsY = 0
	print "Lower_boundary_flagY hit"
else:
	GPS_lower_boundsY = botY - LPS_ORIGINy

if Upper_boundary_flagX > GPS_WIDTH - 1:
	LPS_higher_boundsX = GPS_WIDTH - 1 - botX + LPS_ORIGINx

	GPS_higher_boundsX = GPS_WIDTH - 1
	print "Upper_boundary_flagX hit"
else:
	GPS_higher_boundsX = botX + LPS_ORIGINx


if Upper_boundary_flagY > GPS_HEIGHT - 1:
	LPS_higher_boundsY = GPS_HEIGHT- 1 - botY + LPS_ORIGINy
	GPS_higher_boundsY = GPS_HEIGHT - 1
	print "Upper_boundary_flagY hit"
else:
	GPS_higher_boundsY = botY + LPS_ORIGINy

print "--------------After--------------	"
print "GPS_lower_boundsX: ", GPS_lower_boundsX, "GPS_lower_boundsY: ", GPS_lower_boundsY, "GPS_higher_boundsX: ", GPS_higher_boundsX, "GPS_higher_boundsY: ", GPS_higher_boundsY
print "LPS_lower_boundsX: ", LPS_lower_boundsX, "LPS_lower_boundsY: ", LPS_lower_boundsY, "LPS_higher_boundsX: ", LPS_higher_boundsX, "LPS_higher_boundsY: ", LPS_higher_boundsY

numRowCells = LPS_higher_boundsY - LPS_lower_boundsY + 1
numColCells = LPS_higher_boundsX - LPS_lower_boundsX + 1

GPSskip = (GPS_WIDTH - GPS_higher_boundsX) + GPS_lower_boundsX - 1
LPSskip = (LPS_WIDTH - LPS_higher_boundsX) + LPS_lower_boundsX - 1

GPSidx = (GPS_higher_boundsX * numRowCells) + GPS_lower_boundsY
LPSidx = (LPS_higher_boundsX * numRowCells) + LPS_lower_boundsY

GPSidxX = GPS_lower_boundsX
GPSidxY = GPS_lower_boundsY
LPSidxX = LPS_lower_boundsX
LPSidxY = LPS_lower_boundsY

print ""
print "---------------------------------------------------------"
print "numRowCells: ", numRowCells, "numColCells: ", numColCells
print "GPSskip: ", GPSskip, "GPSidx: ", GPSidx
print "LPSskip: ", LPSskip, "LPSidx: ", LPSidx

for x in range(numRowCells):
	for y in range(numColCells):
		GPS[GPSidxY, GPSidxX] = LPS[LPSidxY, LPSidxX]
		GPSidxX += 1
		LPSidxX += 1
	GPSidxY += 1 #GPSskip
	LPSidxY += 1 #LPSskip
	GPSidxX = GPS_lower_boundsX
	LPSidxX = LPS_lower_boundsX

# GPS[GPS_lower_boundsY,GPS_lower_boundsX] = 0.0
LPS[LPS_lower_boundsY,LPS_lower_boundsX] = 0.1
# GPS[GPS_higher_boundsY,GPS_higher_boundsX] = 1.1
LPS[LPS_higher_boundsY,LPS_higher_boundsX] = 1.1

GPS[botY, botX] = 2.0
LPS[LPS_ORIGINy, LPS_ORIGINx] = 2.0

np.set_printoptions(linewidth = 200)
print "Bot Location (x,y): ", botX, botY
print ""
print GPS
print ""
print LPS