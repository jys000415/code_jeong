# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:52:42 2022

@author: yeong
"""

from Linearize.areaSetting import areasetting, Linearize
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
import numpy as np
import pdb

def linearize_maze(pos, time, mask, armlength, contextname):
    print("Type a number of area you want")
    area_num = int(input())
    area_name = ['p_closed', 'np_closed', 'center', 'p_open', 'np_open']
    cardinal_temp = ['North', 'East', 'South', 'West', 'Center']
    LinearAnimal = []
    PosAnimal = []
    TimeAnimal = []
    card_direct = []
    arm_axis = []
    duration = 15
    array_time = np.array(time)
    array_pos = np.array(pos)
    time = array_time[np.where(array_time <= time[0]+duration*60)[0].tolist()]
    pos = array_pos[np.where(array_time <= time[0]+duration*60)[0].tolist()]
    new_pos = pos.copy()
    if 'ezm' in mask:
        area_name = ['p_closed', 'np_closed', 'p_open', 'np_open']
        cardinal_temp = ['North', 'East', 'South', 'West']
    if 'openfield' in mask:
        area_name = ['corner_1', 'corner_2', 'center', 'corner_3', 'corner_4',
                     'All']
        cardinal_temp = ['NW', 'NE', 'C', 'SW', 'SE', 'All']
    for iarea in range(area_num):
        print(area_name[iarea])
        tmparea = areasetting(new_pos, mask)
        tmpPolygon = Polygon(tmparea.coord)
        print("First point = Start, Second click = End")
        tmpaxis = areasetting([], mask)
        print("Type a cardianl direction %s" % (str(cardinal_temp)))
        direction_num = int(input())
        card_direct.append(cardinal_temp[direction_num])
        tmpLinear = Linearize(tmpPolygon, tmpaxis)
        if contextname == 'EZM':
            tmpLinear.findLinearizepoint_EZM(new_pos, time, armlength)
        else:
            tmpLinear.findLinearizepoint(new_pos, time, armlength)
        LinearAnimal.append(tmpLinear.linearDistance)
        PosAnimal.append(tmpLinear.point)
        
        new_pos = [[i,j] for [i,j] in new_pos if [i,j] not in tmpLinear.point]
        TimeAnimal.append(tmpLinear.time)
        arm_axis.append(tmpaxis.coord)
    return LinearAnimal, PosAnimal, TimeAnimal, card_direct, arm_axis


def project_outpoint(pos, mask):
    print('Setting the boundary-Click n points')
    tmparea = areasetting(pos, mask)
    tmpPolygon = Polygon(tmparea.coord)
    centerpoint = tmpPolygon.centroid
    newpos = [0]*len(pos)
    for ix in range(len(pos)):
        if not tmpPolygon.contains(Point(pos[ix])):
            # Project
            p1, p2 = nearest_points(tmpPolygon, Point(pos[ix]))
            newpos[ix] = [p1.coords[0][0], p1.coords[0][1]]
        else:
            newpos[ix] = pos[ix]

    return newpos, centerpoint, tmparea.coord