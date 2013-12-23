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
    stuff = cpa.Mapping(False, (1000,1000,0))
    np.set_printoptions(linewidth=600, threshold='nan', precision=2, suppress=True)
    
    for i in range(10):
        before = time.clock()
        stuff.update_pos(1000+i*200, 1000+i*200, 0)
        print time.clock() - before
        print stuff.GPS

    # print stuff.LPS
    print stuff.GPS