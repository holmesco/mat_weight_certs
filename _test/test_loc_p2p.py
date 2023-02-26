#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys
import spatialmath.base as sm

from mwcerts.stereo_problems import Localization, Camera,stereo_meas_model
def init_prob():
    # Ground Truth Translation
    r_p = []
    r_p += [np.array([[0,0,0]]).T]
    r_p += [np.array([[0.5,0,0]]).T]
    # r_p += [np.array([[0.5,0.5,0]]).T]
    # Ground Truth Rotations
    C_p = []
    C_p += [sm.roty(0)]
    C_p += [sm.roty(0.2)]
    # C_p += [sm.roty(-0.2)]
    
    # Ground Truth Map Points
    r_l = []
    r_l += [np.array([[0,0,1]]).T]
    r_l += [np.array([[0,0,2]]).T]
    r_l += [np.array([[0,0,3]]).T]
    r_l += [np.array([[1,0,2]]).T]
    r_l += [np.array([[1,1,3]]).T]

    # Generate problem instance
    loc = Localization(r_p,C_p,r_l)
    # Test Ground truth values
    x = loc.get_gt_soln()
    print(loc.var_list)
    print(f"Length of GT vars: {len(x)}")

    # Define edges
    p2m_edges = [('x0','m0'),
            ('x0','m1'),
            ('x0','m2'),
            ('x0','m3'),
            ('x0','m4'),
            ('x1','m0'),
            ('x1','m1'),
            ('x1','m2'),
            ('x1','m3'),
            ('x1','m4')]
    p2p_edges = [('x0','x1')]
    return loc, p2m_edges, p2p_edges

def test_iso_sdp():
    #Init
    loc, p2m_edges, p2p_edges = init_prob()
    # Generate measurements
    sigma = 0.01
    loc.gauss_isotrp_meas_model(p2m_edges,sigma)
    loc.add_p2p_trans_meas(edge_list=p2p_edges,sigma=sigma)
    
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    Q = Q / sla.norm(Q)

    # Test constraints
    loc.validate_constraints(useRedun=True)

    # Run Gauss Newton
    # x_init=loc.init_gauss_newton(sigma=0.0)
    # x_sol,info = loc.gauss_newton(x_init=x_init)
    # x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])

    # Run SDP
    X,cprob = loc.solve_primal_sdp(useRedun=True)
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    tol = 1e-4
    rank = np.sum(S >= tol)
    assert rank == 1, "Rank of SDP solution is not 1"

    x = loc.round_sdp_soln(X.value)
    x_gt = loc.get_gt_soln()
    x_err = x - x_gt
    
    # Test cost function
    cost = x_gt.T @ Q @ x_gt 
    cost_rnd = x.T @ Q @ x
    # cost_gn = x_gn.T @ Q @ x_gn
    print(f"Ground Truth Cost:  {cost}")
    # print(f"Gauss-Newton Cost: {cost_gn}")
    print(f"SDP Rounded Cost:  {cost_rnd}")
    print(f"SDP cost: {cprob.value}")
    np.testing.assert_allclose(cprob.value, cost_gn, atol=1e-8, rtol=0.01)

# def test_stereo_sdp():
#     #Init
#     loc, edges = init_prob()
#     # Define camera model
#     camera = Camera()
#     camera.f_u = 200
#     camera.f_v = 200
#     camera.c_u = 0
#     camera.c_v = 0
#     camera.b = 1
#     camera.sigma_u = 0.5
#     camera.sigma_v = 0.5
#     # Generate measurements
#     stereo_meas_model(loc,edges,camera)

#     # Generate cost matrix
#     loc.generate_cost()
#     Q = loc.Q.get_matrix(variables=loc.var_list)
#     Q = Q / sla.norm(Q)

#     # Test constraints
#     loc.validate_constraints()

#     # Run Gauss Newton
#     x_init=loc.init_gauss_newton(sigma=0.0)
#     x_sol,info = loc.gauss_newton(x_init=x_init)
#     x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])

#     # Run SDP
#     X,cprob = loc.solve_primal_sdp()
#     # Check Rank of solution
#     U,S,V = la.svd(X.value)
#     tol = 1e-4
#     rank = np.sum(S >= tol)
#     assert rank == 1, "Rank of SDP solution is not 1"

#     x = loc.round_sdp_soln(X.value)
#     x_gt = loc.get_gt_soln()
#     x_err = x - x_gt
    
#     # Test cost function
#     cost = x_gt.T @ Q @ x_gt 
#     cost_rnd = x.T @ Q @ x
#     cost_gn = x_gn.T @ Q @ x_gn
#     print(f"Ground Truth Cost:  {cost}")
#     print(f"Gauss-Newton Cost: {cost_gn}")
#     print(f"SDP Rounded Cost:  {cost_rnd}")
#     print(f"SDP cost: {cprob.value}")
#     np.testing.assert_allclose(cprob.value, cost_gn, atol=1e-8, rtol=0.01)


if __name__ == "__main__":
    test_iso_sdp()
    # test_stereo_sdp()
