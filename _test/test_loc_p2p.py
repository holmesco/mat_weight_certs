#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys
import spatialmath.base as sm

from mwcerts.stereo_problems import Localization, Camera,stereo_meas_model

"""This file is for testing Gauss-Newton and SDP solutions for Localization
    with pose-to-map AND pose-to-pose translation measurements. 
"""

def init_prob():
    # Ground Truth Translation
    r_p = []
    r_p += [np.array([[0,0,0]]).T]
    r_p += [np.array([[0.5,0,0]]).T]
    
    # Ground Truth Rotations
    C_p = []
    C_p += [sm.roty(0.2)]
    C_p += [sm.roty(-0.2)]
 
    
    # Ground Truth Map Points
    np.random.seed(0)
    Nm = 20
    offs = np.array([[0,0,4]]).T
    r_l = 0.5*np.eye(3) @ np.random.randn(3,Nm) + offs
    r_l = np.expand_dims(r_l.T,axis=2)

    # Generate problem instance
    loc = Localization(r_p,C_p,r_l)
    # Test Ground truth values
    x = loc.get_gt_soln()
    print(loc.var_list)
    print(f"Length of GT vars: {len(x)}")

    # Define edges
    p2m_edges = []
    for i in range(len(r_p)):
        for j in range(len(r_l)):
            p2m_edges += [(f'x{i}',f"m{j}")]
    p2p_edges = [('x0','x1')]
    return loc, p2m_edges, p2p_edges

def test_no_noise_gn():
    #Init
    loc, p2m_edges, p2p_edges = init_prob()
    # Generate measurements
    sigma = 0.0
    loc.gauss_isotrp_meas_model(p2m_edges,sigma)
    loc.add_p2p_meas(edge_list=p2p_edges,sigma=sigma, mtype='trans')
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    # Run Gauss Newton
    x_init=loc.init_gauss_newton(sigma=1)
    x_sol,info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    # Get Ground truth solution
    x_gt = loc.get_gt_soln()
    # Test cost function
    cost_gt = x_gt.T @ Q @ x_gt 
    cost_gn = x_gn.T @ Q @ x_gn
    np.testing.assert_allclose(cost_gn, cost_gt, atol=1e-8, rtol=0.01)
    np.testing.assert_allclose(x_gn, x_gt,atol=1e-8, rtol=0.01)
    
def test_iso_noise_gn():
    #Init
    loc, p2m_edges, p2p_edges = init_prob()
    # Generate measurements
    sigma = 1e-3
    loc.gauss_isotrp_meas_model(edge_list=p2m_edges,sigma=sigma)
    sigma = 1e-2
    loc.add_p2p_meas(edge_list=p2p_edges,sigma=sigma,mtype='trans')
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    print(loc.var_list)
    # Run Gauss Newton
    x_init=loc.init_gauss_newton(sigma=0)
    x_sol,info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    # Run SDP
    X,cprob = loc.solve_primal_sdp(use_redun=True)
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    tol = 1e-4
    rank = np.sum(S >= tol)
    x_rnd,x_vals = loc.round_sdp_soln(X.value)
    # Get Ground truth solution
    x_gt = loc.get_gt_soln()
    # Test cost function
    cost_gt = x_gt.T @ Q @ x_gt 
    cost_gn = x_gn.T @ Q @ x_gn
    print(loc.var_list)
    np.testing.assert_allclose(cost_gn, info['cost'],atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(cost_gn, cost_gt, atol=np.Inf, rtol=0.1)
    np.testing.assert_allclose(x_gn, x_gt,atol=np.inf, rtol=0.05)
    
def test_no_noise_sdp():
    #Init
    loc, p2m_edges, p2p_edges = init_prob()
    # Generate measurements
    sigma = 0.0
    loc.gauss_isotrp_meas_model(p2m_edges,sigma)
    loc.add_p2p_meas(edge_list=p2p_edges,sigma=sigma,mtype='trans')
    
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    Q = Q / sla.norm(Q)

    # Test constraints
    loc.validate_constraints()

    # Run Gauss Newton
    x_init=loc.init_gauss_newton(sigma=0.0)
    x_sol,info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])

    # Run SDP
    X,cprob = loc.solve_primal_sdp(use_redun=True)
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    tol = 1e-4
    rank = np.sum(S >= tol)
    assert rank == 1, "Rank of SDP solution is not 1"

    x,_ = loc.round_sdp_soln(X.value)
    x_gt = loc.get_gt_soln()
    x_err = x - x_gt
    
    # Test cost function
    cost_gt = x_gt.T @ Q @ x_gt 
    cost_rnd = x.T @ Q @ x
    cost_gn = x_gn.T @ Q @ x_gn
    cost_sdp = np.trace(Q @ X.value)
    print(f"Ground Truth Cost:  {cost_gt}")
    print(f"Gauss-Newton Cost: {cost_gn}")
    print(f"SDP Rounded Cost:  {cost_rnd}")
    print(f"SDP cost: {cost_sdp}")
    np.testing.assert_allclose(cost_gn,info['cost'], atol=1e-8, rtol=0.01)
    np.testing.assert_allclose(cost_gn,cost_sdp, atol=1e-8, rtol=0.01)
   
def test_stereo_sdp():
    #Init
    loc, p2m_edges, p2p_edges = init_prob()
    # Define camera model
    camera = Camera()
    camera.f_u = 200
    camera.f_v = 200
    camera.c_u = 0
    camera.c_v = 0
    camera.b = 1
    camera.sigma_u = 0.5
    camera.sigma_v = 0.5
    # Generate measurements
    stereo_meas_model(loc,p2m_edges,camera)
    sigma = 0.1
    loc.add_p2p_meas(edge_list=p2p_edges,sigma=sigma,mtype='trans')

    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    
    # Test constraints
    loc.validate_constraints(use_redun=True)

    # Run Gauss Newton
    x_init=loc.init_gauss_newton(sigma=0.0)
    x_sol,info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])

    # Run SDP
    X,cprob = loc.solve_primal_sdp(use_redun=True)
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    tol = 1e-4
    rank = np.sum(S >= tol)
    assert rank == 1, "Rank of SDP solution is not 1"
    x_rnd,x_vals = loc.round_sdp_soln(X.value)
    x_gt = loc.get_gt_soln()
    
    # Test cost function
    cost = x_gt.T @ Q @ x_gt 
    cost_rnd = x_rnd.T @ Q @ x_rnd
    cost_gn = x_gn.T @ Q @ x_gn
    cost_sdp = np.trace(Q @ X.value)
    print(f"Ground Truth Cost:  {cost}")
    print(f"Gauss-Newton Cost: {cost_gn}")
    print(f"SDP Rounded Cost:  {cost_rnd}")
    print(f"SDP cost: {cost_sdp}")
    np.testing.assert_allclose(cost_gn, info['cost'],err_msg='GN cost not consistent with cost matrix')
    np.testing.assert_allclose(cost_gn,cost_sdp, atol=1e-8, rtol=0.001)


if __name__ == "__main__":
    test_no_noise_gn()
    test_iso_noise_gn()
    test_no_noise_sdp()
    test_stereo_sdp()
    
