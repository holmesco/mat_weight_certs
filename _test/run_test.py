
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys

# Append working dir
sys.path.append(os.path.dirname(__file__) + "/../")
print("appended:", sys.path[-1])
# Import module files
from stereo_problems import Localization
from poly_matrix.poly_matrix import PolyMatrix

def run_test(loc : Localization, loc_res : str):
    """Runs tests and generates plots for a problem instance

    Args:
        loc (Localization): Localization problem instance
        loc_res (str): Path to results directory
    """
    # Generate cost matrix
    loc.generate_cost()
    Q = loc.Q.get_matrix(variables=loc.var_list)
    Q = Q / sla.norm(Q)
    # Show Cost function
    plt.matshow(Q.todense())
    plt.savefig(loc_res + 'costMat.png')


    # Show constraints
    A = PolyMatrix()
    for constraint in loc.constraints:
        A += constraint.A
    plt.matshow(A.get_matrix(loc.var_list).todense())
    plt.savefig(loc_res + 'cumCostr.png')
    # Test constraints
    loc.validate_constraints()

    # Run SDP
    X,cprob = loc.solve_primal_sdp()
    # Check Rank of solution
    U,S,V = la.svd(X.value)
    plt.figure()
    plt.semilogy(S,'.')
    plt.title("Solution Singular Values")
    plt.savefig(loc_res + 'RankPlot.png')
    tol = 1e-5
    rank = np.sum(S >= tol)
    print(f"Solution rank for tolerance {tol} is {rank}")


    x = loc.round_sdp_soln(X.value)
    x_gt = loc.get_gt_soln()
    x_err = x - x_gt
    plt.figure()
    plt.plot(x_err)
    plt.savefig(loc_res + 'x_error.png')

    # Test cost function
    cost = x_gt.T @ Q @ x_gt 
    cost_rnd = x.T @ Q @ x
    print(f"Ground Truth Cost:  {cost}")
    print(f"SDP Rounded Cost:  {cost_rnd}")
    print(f"SDP cost: {cprob.value}")
