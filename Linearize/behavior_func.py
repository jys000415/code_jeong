# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:56:17 2021

@author: yeong
"""


def linearize_EPM(pathinfo, mask, area_name, maze, context_name, axis_name, duration, armlength, radius, remap):

    from Data_process.dataimport import dataorganize
    from Linearize.areaSetting import areasetting, Linearize
    from shapely.geometry import Polygon
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    LinearAnimal = []
    PosAnimal = []
    TimeAnimal = []
    AxisAnimal = []
# Rearragne of linear position

    for ianimal in range(len(pathinfo)):
        animalPosition, animalTime, animalSpk, animalCsc, animallight, animalWv = dataorganize(
            pathinfo[ianimal], maze, remap)
        # Get area data
        print(area_name)
        tmparea = areasetting(animalPosition[context_name], mask)
        tmpPolygon = Polygon(tmparea.coord)
        print(axis_name)
        tmpaxis = areasetting([], mask)

        tmptime = animalTime[context_name]
        tmppos = animalPosition[context_name]

        epmtime = tmptime[np.where(
            tmptime <= tmptime[0]+duration*60)[0].tolist()]
        epmpos = tmppos[np.where(
            tmptime <= tmptime[0]+duration*60)[0].tolist()]

        smoothopenx = gaussian_filter1d(epmpos[:, 0], 10)
        smoothopeny = gaussian_filter1d(epmpos[:, 1], 10)
        epmpos = np.concatenate((smoothopenx.reshape(
            len(smoothopenx), 1), smoothopeny.reshape(len(smoothopeny), 1)), axis=1)

        tmpLinear = Linearize(tmpPolygon, tmpaxis)
        tmpLinear.findLinearizepoint(epmpos, epmtime, armlength)
        LinearAnimal.append(tmpLinear)
        PosAnimal.append(tmpLinear.point)
        TimeAnimal.append(tmpLinear.time)
        AxisAnimal.append(tmpaxis.coord)
    return LinearAnimal, PosAnimal, TimeAnimal, AxisAnimal


def linearize_EZM(pathinfo, mask, area_name, maze, context_name, axis_name, duration, num_spatial_bin, radius, remap):
    from Data_process.dataimport import dataorganize
    from Linearize.areaSetting import areasetting, Linearize
    from shapely.geometry import Polygon
    from scipy.ndimage import gaussian_filter1d
    import numpy as np
    LinearAnimal = []
    PosAnimal = []
    TimeAnimal = []
    AxisAnimal = []
# Rearragne of linear position
    for ianimal in range(len(pathinfo)):
        animalPosition, animalTime, animalSpk, animalCsc, animallight, animalWv = dataorganize(
            pathinfo[ianimal], maze, remap)
        # Get area data
        print(area_name)
        tmparea = areasetting(animalPosition[context_name], mask)
        tmpPolygon = Polygon(tmparea.coord)
        print(axis_name)
        tmpaxis = areasetting([], mask)

        tmptime = animalTime[context_name]
        tmppos = animalPosition[context_name]

        epmtime = tmptime[np.where(
            tmptime <= tmptime[0]+duration*60)[0].tolist()]
        epmpos = tmppos[np.where(
            tmptime <= tmptime[0]+duration*60)[0].tolist()]

        smoothopenx = gaussian_filter1d(epmpos[:, 0], 10)
        smoothopeny = gaussian_filter1d(epmpos[:, 1], 10)
        epmpos = np.concatenate((smoothopenx.reshape(
            len(smoothopenx), 1), smoothopeny.reshape(len(smoothopeny), 1)), axis=1)

        tmpLinear = Linearize(tmpPolygon, tmpaxis)
        tmpLinear.findLinearizepoint_EZM(epmpos, epmtime, radius)
        LinearAnimal.append(tmpLinear)
        PosAnimal.append(tmpLinear.point)
        TimeAnimal.append(tmpLinear.time)
        AxisAnimal.append(tmpaxis.coord)
    return LinearAnimal, PosAnimal, TimeAnimal, AxisAnimal
