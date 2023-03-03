#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import spatialmath as sm
# dev imports
from mwcerts.stereo_problems import Localization, Camera, stereo_meas_model
import pytest

def init_problem():
    # Define Ground truth values
    r_p = []
    C_p = []
    r_p += [np.array([[0,0,1]]).T]
    r_p += [np.array([[0,3,1]]).T]
    C_p += [np.eye(3)]
    C_p += [np.array(sm.SO3.Rz(np.pi/2))]
    r_l = []
    r_l += [np.array([[1,0,2]]).T]
    r_l += [np.array([[0,1,2]]).T]
    r_l += [np.array([[0,0,2]]).T]
    r_l += [np.array([[1,1,2]]).T]
    # Generate problem instance
    loc = Localization(r_p,C_p,r_l)
    # Test Ground truth values
    x_gt = loc.get_gt_soln()
    # Generate measurements
    edges = [('x0','m0'),
            ('x0','m1'),
            ('x0','m2'),
            ('x0','m3'),
            ('x1','m0'),
            ('x1','m1'),
            ('x1','m2'),
            ('x1','m3')]
    # Seed RNG
    np.random.seed(0)
    
    return loc, x_gt, edges

def test_gn_no_noise():
    """No Noise Gauss Newton Test
    """
    # Init
    loc, x_gt, edges = init_problem()
    # Define edges with no noise
    sigma = 0
    loc.gauss_isotrp_meas_model(edges,sigma)
    # Run GN - start at random point.
    x_init = 0.1*np.random.randn(3*(len(loc.var_list)-1))
    x_sol, info = loc.gauss_newton(x_init=None)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    assert info['n_iter'] < info['options'].max_iter, "Gauss Newton did not converge"
    np.testing.assert_allclose(x_gn,x_gt,atol=1e-12,err_msg="Gauss Newton - No Noise")
    
    del loc
    
def test_gn_iso_noise():
    """Isotropic noise Gauss Newton Test
    """
    # Init
    loc, x_gt, edges = init_problem()
    # Generate edges with noise.
    sigma = 0.0005
    loc.gauss_isotrp_meas_model(edges,sigma)
    # Run GN - start at ground truth
    x_init = loc.init_gauss_newton()
    x_sol, info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    assert info['n_iter'] < info['options'].max_iter, "Gauss Newton did not converge"
    np.testing.assert_allclose(x_gn, x_gt, atol=0.005,rtol=np.Inf,err_msg="Gauss Newton - Iso Noise")
    
def test_gn_stereo():
    """Noisy Gauss Newton Test
    """
    # Init
    loc, x_gt, edges = init_problem()
    # Define Camera Model
    camera = Camera()
    camera.f_u = 200
    camera.f_v = 200
    camera.c_u = 0
    camera.c_v = 0
    camera.b = 0.3
    camera.sigma_u = 0.5
    camera.sigma_v = 0.5
    # Generate Stereo measurements
    u_meas,v_meas,d_meas = stereo_meas_model(loc,edges,camera)
    # Run GN - start at ground truth
    np.random.seed(0)
    sigma_init = 0.1
    x_init = loc.init_gauss_newton(sigma_init)
    x_sol, info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    np.testing.assert_allclose(x_gn,x_gt,atol=1e-1,err_msg="Stereo GT compare")
    
    # Regenerate Camera measurements based on solution:
    for v in loc.G.Vp.values():
        v.C_p0 = np.reshape(x_sol[v.label+"_C"], (3,3), order="f")
        v.r_in0 = v.C_p0.T @ x_sol[v.label+"_t"]
    camera.sigma_u = 0.00001
    camera.sigma_v = 0.00001
    u_est,v_est,d_est = stereo_meas_model(loc,edges,camera)
    plt.figure()
    ax = plt.axes()
    ax.scatter(u_est,v_est,marker='o',color='r',label='estimated')
    ax.scatter(u_meas,v_meas,marker='.',color='b',label='measured')
    ax.legend()
    ax.set_title("Pixel Coordinates")
    plt.show()
    # Check measurements 
    assert info['n_iter'] < info['options'].max_iter, "Gauss Newton did not converge"
    np.testing.assert_allclose(u_est,u_meas,atol=3*0.5,err_msg="Stereo meas horizontal")
    np.testing.assert_allclose(v_est,v_meas,atol=3*0.5,err_msg="Stereo meas vertical")
    np.testing.assert_allclose(d_est,d_meas,atol=3*np.sqrt(2)*0.5,err_msg="Stereo meas disparity")
    
if __name__ == "__main__":
    test_gn_no_noise()
    test_gn_iso_noise()
    test_gn_stereo()

