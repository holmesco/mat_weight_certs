#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys
import spatialmath.base as sm
from mwcerts.stereo_problems import Localization, Camera, stereo_meas_model
from poly_matrix.poly_matrix import PolyMatrix
from run_test import run_loc_test
# Create results Folder
loc_res = os.path.dirname(__file__) + '/LocStereoResults/loc_ster_tight/'
if not os.path.isdir(loc_res):
    os.makedirs(loc_res)

# Ground Truth Translation
r_p = []
r_p += [np.array([[0,0,0]]).T]
r_p += [np.array([[0.5,0,0]]).T]
# Ground Truth Rotations
C_p = []
C_p += [sm.roty(0)]
C_p += [sm.roty(-0.1)]

# Ground Truth Map Points
Nm = 7
offs = np.array([[0,0,2]]).T
r_l = 0.5*np.eye(3) @ np.random.randn(3,Nm) + offs
r_l = np.expand_dims(r_l.T,axis=2)
# Generate problem instance
loc = Localization(r_p,C_p,r_l)
fig = plt.figure()
ax = loc.plot_ground_truth()
# Test Ground truth values
x = loc.get_gt_soln()
print(loc.var_list)
print(f"Length of GT vars: {len(x)}")
# Define edges
edges = []
for i in range(len(r_p)):
    for j in range(len(r_l)):
        edges += [(f'x{i}',f"m{j}")]
# Define camera model
camera = Camera()
camera.f_u = 200
camera.f_v = 200
camera.c_u = 0
camera.c_v = 0
camera.b = 0.3
camera.sigma_u = 0.5
camera.sigma_v = 0.5
# Generate measurements
u,v,d = stereo_meas_model(loc,edges,camera)
loc.plot_measurements(ax)
plt.savefig(loc_res + 'situational.png')
# Run tests
run_loc_test(loc, loc_res)