# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:06:15 2020

@author: yeong
"""


def areasetting(pos, mask):
    import matplotlib.pyplot as plt
    import cv2

    def enterPoint():
        point = int(input("Please enter a number:"))
        return point

    class CoordClick:
        global offset
        offset = []

        def __init__(self, fig, clickpoint=0):
            self.fig = fig
            self.xs = list(fig.get_xdata())
            self.ys = list(fig.get_ydata())
            self.first = []
            self.second = []
            self.clickpoint = clickpoint
            self.coord = list()
            self.cid = fig.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            print('click', event)
            self.ys.append(event.ydata)
            self.xs.append(event.xdata)
            self.coord.append([event.xdata, event.ydata])
            self.fig.set_data(self.xs, self.ys)
            self.fig.figure.canvas.draw()
            offset.append(event.ydata)
            if len(self.xs) == self.clickpoint:
                self.fig.figure.canvas.mpl_disconnect(self.cid)
    point = enterPoint()
    fig, ax = plt.subplots()
    im = cv2.imread(mask)
    plt.imshow(im)
    if isinstance(pos, list):
        import numpy as np
        pos = np.asarray(pos)
    if len(pos) > 0:
        ax.scatter(pos[:, 0], pos[:, 1], s=1, c='gray')
    line, = ax.plot([])
    zonecoord = CoordClick(line, point)
    while len(offset) != point:
        plt.waitforbuttonpress(timeout=-1)
    print('Offset =', offset)
    plt.close()
    return zonecoord


def coordCal(self):
    if not self.coord:
        print(self.coord)
        pass
    else:
        from sympy import Symbol, solve
        y = Symbol('y')
        x = Symbol('x')
        equation1 = (y-self.coord[1][1])/(x-self.coord[1][0]) + (self.coord[0][0]-self.coord[1][0])/(self.coord[0][1]-self.coord[1][1])
        equation2 = (self.coord[1][0]-self.coord[0][0])**2 + (self.coord[1][1]-self.coord[0][1])**2-(x-self.coord[1][0])**2-(y-self.coord[1][1])**2
        self.first.append(solve((equation1, equation2), dict=True))
        equation1 = (y-self.coord[0][1])/(x-self.coord[0][0]) + (self.coord[1][0]-self.coord[0][0])/(self.coord[1][1]-self.coord[0][1])
        equation2 = (self.coord[1][0]-self.coord[0][0])**2 + (self.coord[1][1]-self.coord[0][1])**2-(x-self.coord[0][0])**2-(y-self.coord[0][1])**2
        self.second.append(solve((equation1, equation2), dict=True))
        # Add 2 calculated coordinations to self.coord
        y = Symbol('y')
        x = Symbol('x')
        self.coord.append([self.first[0][0][x], self.first[0][0][y]])
        self.coord.append([self.second[0][0][x], self.second[0][0][y]])
    return self.coord


class Linearize:

    def __init__(self, polygon, axis):
        self.point = []
        self.projectpoint = []
        self.linearDistance = []
        self.angle = []
        self.time = []
        self.diffdist = []
        self.axis = axis
        self.polygon = polygon

    def findLinearizepoint(self, position, time, armlength):
        from shapely.geometry import Point, LineString
        from etc.etcfunc import calculateDistance
        import numpy as np
        import sys
        epsilon = sys.float_info.epsilon
        for ix in range(len(position)):
            if self.polygon.contains(Point(position[ix])):
                self.time.append(time[ix])
                self.point.append([Point(position[ix]).coords[0][0],
                                   Point(position[ix]).coords[0][1]])
                point = Point(position[ix])
                line = LineString(self.axis.coord)
                avg = [float(sum(col))/len(col) for col in zip(*self.axis.coord)]

                # Project all the points to linearize axis
                x = np.array(point.coords[0])
                u = np.array(line.coords[0])
                v = np.array(line.coords[len(line.coords)-1])
                scale = armlength/calculateDistance(u[0], u[1], v[0], v[1])
                n = v-u
                n /= np.linalg.norm(n, 2)
                zip_object = zip([u+n*np.dot(x-u, n)], [avg])
                x_project = []
                x_project = [[u+n*np.dot(x-u, n)][0][0], [u+n*np.dot(x-u, n)][0][1]]
                dist_1 = calculateDistance(x_project[0], x_project[1], u[0], u[1])
                dist_2 = calculateDistance(x_project[0], x_project[1], v[0], v[1])
                dist_3 = calculateDistance(u[0], u[1], v[0], v[1])
                if abs(dist_3-(dist_1 + dist_2)) > epsilon*1000:
                    self.diffdist.append((abs(dist_3-(dist_1 + dist_2))))
                    if dist_1 > dist_2:
                        x_project = v
                    else:
                        x_project = u
                else:
                    x_project = x
                # Calculate the relative coordinates
                for list1_i, list2_i in zip_object:
                    self.projectpoint.append(list1_i-list2_i)     
                # Calculate the distance from center
                self.linearDistance.append(calculateDistance([u+n*np.dot(x_project-u, n)][0][0],
                                                             [u+n*np.dot(x_project-u, n)][0][1],
                                                             self.axis.coord[0][0],
                                                             self.axis.coord[0][1])*scale)

    def findLinearizepoint_EZM(self, position, time, radius):
        from shapely.geometry import Point, LineString
        from etc.etcfunc import getAngle
        import numpy as np
        import math

        for ix in range(len(position)):
            if self.polygon.contains(Point(position[ix])):
                self.time.append(time[ix])
                self.point.append([Point(position[ix]).coords[0][0],
                                   Point(position[ix]).coords[0][1]])
                tmppoint = Point(position[ix])
                line = LineString(self.axis.coord)
                # Project all the points to linearize axis
                x = np.array(tmppoint.coords[0])
                centerpoint = np.array(line.coords[0])
                referencepoint = np.array(line.coords[1])

                angle = getAngle(referencepoint, centerpoint, x)
                if (angle > 90) & (angle < 180):
                    angle = 90
                elif angle > 270:
                    angle = 0
                rad_angle = angle*math.pi/180

                # Calculate the distance from center
                self.linearDistance.append(radius*rad_angle)
                self.angle.append(angle)
