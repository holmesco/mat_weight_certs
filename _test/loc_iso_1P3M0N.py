#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys
from run_test import run_test

# Append working dir
sys.path.append(os.path.dirname(__file__) + "/../")
print("appended:", sys.path[-1])
# Import module files
from stereo_problems import Localization
from poly_matrix.poly_matrix import PolyMatrix

# Create results Folder
loc_res = os.path.dirname(__file__) + '/LocalizeResults/test_1P3M0N/'
if not os.path.isdir(loc_res):
    os.makedirs(loc_res)

# Define Ground truth values
Np = 1
Nl = 3
r_p = []
C_p = []
for i in range(Np):
    r_p += [np.array([[0,0,i]]).T]
    C_p += [np.eye(3)]
r_l = []
r_l += [np.array([[1,0,0]]).T]
r_l += [np.array([[0,1,0]]).T]
r_l += [np.array([[0,0,1]]).T]


# Generate problem instance
loc = Localization(r_p,C_p,r_l)
# Test Ground truth values
x = loc.get_gt_soln()
print(loc.var_list)
print(f"Length of GT vars: {len(x)}")

# Define edges
edges = [('x0','m0'),
         ('x0','m1'),
         ('x0','m2')]

# Generate measurements
sigma = 0.01
loc.gauss_isotrp_meas_model(edges,sigma)

# Call test
run_test(loc, loc_res)