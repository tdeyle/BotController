import time
from math import sin, cos
# from scan import *
import numpy as np
import cython_process_array as cpa
import cython_simulator as cs
# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.clock import Clock

# class MyWidget(Widget):
#     pass

# class MyApp(App):
#     def build(self):

#         return MyWidget()

if __name__ == "__main__":
    
    control = True

    my_map = cpa.Mapping()
    fBotx = 1000
    fBoty = 1000
    CELL_SIZE = 200

    np.set_printoptions(linewidth=600, threshold='nan', precision=2, suppress=True)
    
    if control is True:
        input_key = ""

        while input_key != "q":
            input_key = raw_input()

            if input_key == "w":
                fBoty -= CELL_SIZE
            elif input_key == "s":
                fBoty += CELL_SIZE
            elif input_key == "a":
                fBotx -= CELL_SIZE
            elif input_key == "d":
                fBotx += CELL_SIZE

            my_map.update_pos(fBotx, fBoty, 0)

            print "LPS: "
            print my_map.LPS
            print "GPS: "
            print my_map.GPS
    else:
        print my_map.GPS
        
        for i in range(10):
            # before = time.clock()
            # my_map.update_pos(1000, 1000, 0)
            my_map.update_pos(1000+200*i, 1000+200*i, 0)

        # my_map.measureDistance(my_map.distance, my_map.sim_map, my_map.x, my_map.y, my_map.theta)
        # LPS[0:] = 0.5
        # my_map.cy_detectHits(my_map.distance, my_map.LPS, my_map.theta)
        # my_map.cy_updateFromLPS(my_map.x, my_map.y, my_map.theta, my_map.LPS, my_map.GPS)

            # print time.clock() - before

        # print stuff.LPS
        print my_map.GPS
        print my_map.sim_map