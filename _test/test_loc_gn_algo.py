#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys

# dev imports
from mwcerts.stereo_problems import Localization, GaussNewtonOpts

def test_gn_no_noise():
    """No Noise Gauss Newton Test
    """
    # Define Ground truth values
    r_p = []
    C_p = []
    r_p += [np.array([[0,0,0]]).T]
    C_p += [np.eye(3)]
    r_l = []
    r_l += [np.array([[1,0,0]]).T]
    r_l += [np.array([[0,1,0]]).T]
    r_l += [np.array([[0,0,1]]).T]
    r_l += [np.array([[1,1,0]]).T]
    r_l = np.array(r_l)
    # Generate problem instance
    loc = Localization(r_p,C_p,r_l)
    # Test Ground truth values
    x_gt = loc.get_gt_soln()
    # Generate measurements
    edges = [('x0','m0'),
            ('x0','m1'),
            ('x0','m2'),
            ('x0','m3')]
    sigma = 0.1
    loc.gauss_isotrp_meas_model(edges,sigma)
    # Run GN - start at random point.
    opt = GaussNewtonOpts()
    vert_list = list( loc.G.Vp.values())
    np.random.seed(0)
    x_init = 0.001*np.random.randn(6*len(vert_list))
    x_gn = loc.gauss_newton(vert_list ,opt, x_init=x_init)
    np.testing.assert_allclose(x_gn,x_gt,err_msg="Gauss Newton - No Noise")
    
if __name__ == "__main__":
    test_gn_no_noise()

