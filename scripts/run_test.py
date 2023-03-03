
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la

# Import module files
from mwcerts.stereo_problems import Localization
from poly_matrix.poly_matrix import PolyMatrix

def run_loc_test(loc : Localization, results : str, use_redun : bool=False):
    """Runs tests and generates plots for a localization problem instance

    Args:
        loc (Localization): Localization problem instance
        results (str): Path to results directory
    """
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    Q = Q / sla.norm(Q)
    # Show Cost function
    loc.Q.matshow(loc.var_list)
    if not results is None:
        plt.savefig(results + 'costMat.png')
    
    # Show constraints
    A = PolyMatrix()
    for constraint in loc.constraints:
        A += constraint.A
    if use_redun:
        for constraint in loc.constraints_r:
            A += constraint.A
    A.matshow(loc.var_list)
    if not results is None:
        plt.savefig(results + 'cumCostr.png')
    # Test constraints
    loc.validate_constraints(use_redun=use_redun)

    # Run Gauss Newton
    x_init=loc.init_gauss_newton(sigma=0.0)
    x_sol,info = loc.gauss_newton(x_init=x_init)
    x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])
    
    # Run SDP
    X,cprob = loc.solve_primal_sdp(use_redun=use_redun)
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    plt.figure()
    plt.semilogy(S,'.')
    plt.title("Solution Singular Values")
    if not results is None:
        plt.savefig(results + 'RankPlot.png')
    tol = 1e-5
    rank = np.sum(S >= tol)
    print(f"Solution rank for tolerance {tol} is {rank}")

    x = loc.round_sdp_soln(X.value)
    x_gt = loc.get_gt_soln()
    x_err = x - x_gt
    plt.figure()
    plt.plot(x_err)
    if not results is None:
        plt.savefig(results + 'x_error.png')

    # Test cost function
    cost = x_gt.T @ Q @ x_gt 
    cost_rnd = x.T @ Q @ x
    cost_gn = x_gn.T @ Q @ x_gn
    cost_sdp = np.trace(Q @ X.value)
    print(f"Ground Truth Cost:  {cost}")
    print(f"Gauss-Newton Cost: {cost_gn}")
    print(f"SDP Rounded Cost:  {cost_rnd}")
    print(f"SDP cost: {cost_sdp}")
