# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:17:00 2020

@author: yeong
"""
import numpy as np
tilt = 7
verticaldv = 4.1
ml = verticaldv-verticaldv*np.tan(tilt*np.pi/180)
adddv = ml*np.tan(tilt*np.pi/180)
tiltdv = verticaldv/np.cos(tilt*np.pi/180)
dv = adddv+tiltdv
print(ml, adddv,tiltdv,dv)