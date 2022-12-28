# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:00:32 2022

@author: yeong
"""


def rotate(point, angle):
    import math
    x = math.cos(angle)*point[0] - math.sin(angle)*point[1]
    newy = math.sin(angle)*point[0] + math.cos(angle)*point[1]
    newx = x
    rot_cord = [newx, newy]
    return rot_cord


def trajectory_2d(pos, time, tmpaxis, card_direct, card_direct_2, remap):
    import math
    import numpy as np
    # Calculate cosine value btw two axes
    axis_order = [0,1,3,4]
    if remap:
       axis_order = [[i for i in range(len(card_direct))
                      if item1 == card_direct_2[i]] for item1 in card_direct]
       axis_order = [j for i in axis_order for j in i]
       del axis_order[2]
    armaxis = tmpaxis[axis_order[0]]
    armaxis_other = tmpaxis[axis_order[1]]
    if len(time[axis_order[0]]) < len(time[axis_order[1]]):
        armaxis.reverse()
        armaxis_other.reverse()
    tmppreferclosedaxis = [[armaxis[0][0]-armaxis[1][0],
                            armaxis[0][1]-armaxis[1][1]], [0, 0]]
    fixed_point = tmppreferclosedaxis[1]
    tmppos = pos[axis_order[0]]
    tmpanimalpos = [[tmppos[ipoint][0]-armaxis[1][0],
                     tmppos[ipoint][1]-armaxis[1][1]] for ipoint in range(len(tmppos))]
    angle1 = math.atan2(tmppreferclosedaxis[0][1]-fixed_point[1],
                        tmppreferclosedaxis[0][0]-fixed_point[0])
    angle2 = math.atan2(1, 0)
    rot_angle = (angle2-angle1)
    rot_prefer_closed = [rotate(tmpanimalpos[ipos], rot_angle)
                         for ipos in range(len(tmpanimalpos))]
    
    # Rotate position - other closed arm

    tmpotherclosedaxis = [[0, 0],
                          [armaxis_other[1][0]-armaxis_other[0][0],
                           armaxis_other[1][1]-armaxis_other[0][1]]]
    fixed_point = tmpotherclosedaxis[1]
    tmppos = pos[axis_order[1]]
    tmpanimalpos = [[tmppos[ipoint][0]-armaxis_other[0][0],
                     tmppos[ipoint][1]-armaxis_other[0][1]] for ipoint in range(len(tmppos))]
    angle1 = math.atan2(tmpotherclosedaxis[0][1]-fixed_point[1],
                        tmpotherclosedaxis[0][0]-fixed_point[0])
    angle2 = math.atan2(1, 0)
    rot_angle = (angle2-angle1)
    rot_other_closed = [rotate(tmpanimalpos[ipos], rot_angle)
                        for ipos in range(len(tmpanimalpos))]
    
    # Rotate position value - Prefered open arm
    armaxis = tmpaxis[axis_order[2]]
    armaxis_other = tmpaxis[axis_order[3]]
    if len(time[axis_order[2]]) < len(time[axis_order[3]]):
        armaxis.reverse()
        armaxis_other.reverse()
    tmpotheropenaxis = [[armaxis[0][0]-armaxis[1][0],
                            armaxis[0][1]-armaxis[1][1]], [0, 0]]
    fixed_point = tmpotheropenaxis[1]
    tmppos = pos[axis_order[2]]
    tmpanimalpos = [[tmppos[ipoint][0]-armaxis[1][0], tmppos[ipoint]
                     [1]-armaxis[1][1]] for ipoint in range(len(tmppos))]
    angle1 = math.atan2(tmpotheropenaxis[0][1]-fixed_point[1],
                        tmpotheropenaxis[0][0]-fixed_point[0])
    angle2 = math.atan2(0, -1)
    rot_angle = (angle2-angle1)
    rot_prefer_open = [rotate(tmpanimalpos[ipos], rot_angle)
                       for ipos in range(len(tmpanimalpos))]

    # Rotate position value - other open arm
    tmpotheropenaxis = [[0, 0], [armaxis_other[1][0]-armaxis_other[0][0],
                                 armaxis_other[1][1]-armaxis_other[0][1]]]
    fixed_point = tmpotheropenaxis[1]
    tmppos = pos[axis_order[3]]
    tmpanimalpos = [[tmppos[ipoint][0]-armaxis_other[0][0], tmppos[ipoint]
                     [1]-armaxis_other[0][1]] for ipoint in range(len(tmppos))]
    angle1 = math.atan2(tmpotheropenaxis[0][1]-fixed_point[1],
                        tmpotheropenaxis[0][0]-fixed_point[0])
    angle2 = math.atan2(0, -1)
    rot_angle = (angle2-angle1)
    rot_other_open = [rotate(tmpanimalpos[ipos], rot_angle)
                      for ipos in range(len(tmpanimalpos))]

    # Trajectory (Linear plot)
    b = np.vstack(rot_prefer_closed)
    c = np.vstack(rot_other_closed)
    d = np.vstack(rot_prefer_open)
    e = np.vstack(rot_other_open)
    rot_pos_vstack = np.concatenate([b, c, d, e])
    rot_time_vstack = np.concatenate([time[axis_order[0]], time[axis_order[1]],
                                      time[axis_order[2]], time[axis_order[3]]])
    z = np.insert(rot_pos_vstack, 0, rot_time_vstack, axis=1)
    sortedZ = z[z[:, 0].argsort()]
    sortedZ = np.delete(sortedZ, 0, 1)
    rot_time_vstack = sorted(rot_time_vstack)
    rot_time_vstack = np.array(rot_time_vstack)

    return sortedZ, rot_time_vstack


def trajectory_2d_open(pos, time, tmpaxis):
    import math
    import numpy as np
    # Calculate cosine value btw two axes
    armaxis = [tmpaxis[2], tmpaxis[1]]
    tmparmdaxis = [[armaxis[0][0]-armaxis[1][0],
                            armaxis[0][1]-armaxis[1][1]], [0, 0]]
    fixed_point = tmparmdaxis[1]
    tmppos = pos[-1]
    tmpanimalpos = [[tmppos[ipoint][0]-armaxis[1][0],
                     tmppos[ipoint][1]-armaxis[1][1]] for ipoint in range(len(tmppos))]
    angle1 = math.atan2(tmparmdaxis[0][1]-fixed_point[1],
                        tmparmdaxis[0][0]-fixed_point[0])
    angle2 = math.atan2(1, 0)
    rot_angle = (angle2-angle1)
    rot_of = [rotate(tmpanimalpos[ipos], rot_angle)
                         for ipos in range(len(tmpanimalpos))]

    # Trajectory (Linear plot)
    rot_pos_vstack = np.vstack(rot_of)
    rot_time_vstack = time[-1]
    z = np.insert(rot_pos_vstack, 0, rot_time_vstack, axis=1)
    sortedZ = z[z[:, 0].argsort()]
    sortedZ = np.delete(sortedZ, 0, 1)
    rot_time_vstack = sorted(rot_time_vstack)
    rot_time_vstack = np.array(rot_time_vstack)

    return sortedZ, rot_time_vstack