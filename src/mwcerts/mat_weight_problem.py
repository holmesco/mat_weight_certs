#!/usr/bin/env python3

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from mwcerts.meas_graph import MeasGraph, MapVertex
import cvxpy as cp
import scipy.linalg as la
import scipy.sparse.linalg as sla
import spatialmath as sm
# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class Constraint:
    def __init__(self,A : PolyMatrix, b : float , label : str):
        self.A = A
        self.b = b
        self.label = label

   
class MatrixWeightedProblem:
    """
    Generic Matrix Weighted Problem Class
    This class is intended to encompass all of the possible types of problems we can see, the most general
    being matrix weighted SLAM (matrix weights on pose-to-pose and pose-to-landmark measurements)
    """
    def __init__(self, r_p_in0, C_p_0, r_m_in0):
        # Define Measurement graph using ground truth values
        self.G = MeasGraph(r_p_in0, C_p_0, r_m_in0)
        # Define dictionary of variables for order and size
        self.var_list = {}
        # Init cost: J = x^t Q x
        self.Q = PolyMatrix()
        # Init Constraint and Redundant Constraint lists
        # A x = b
        self.constraints = []
        self.constraints_r = []
        # Translation reference frames
        self.pose_trans_in_world = False
        # Optimization parameters
        self.SDP_solvr = cp.MOSEK
        
    def generate_cost(self):
        """
        Generate cost matrix using defined list of edges. Should be defined by child class,
        but depend entirely on edges of measurement graph.
        """
        raise Exception("generate_cost must be defined by child class")

    def generate_constraints(self):
        """
        Generate the quadratic constraint matrices for the problem
        Should generate both original constraints and redundant constraints (if any)
        """
        raise Exception("generate_constraints must be defined by child class")
        
    def get_gt_soln(self, var_list=None):
        """Loop through var_list and pull ground truth values from corresponding vertices.

        Args:
            var_list (list of strings): list of variables for the problem. Expected format of
            list elements: w_0, x#_C, x#_t, m#
        Returns:
            numpy array: solution vector
        """
        # Default to list defined in class
        if var_list is None:
            var_list = self.var_list
        
        x_sol = []
        for var in var_list.keys():
            lbl = var.split('_')
            if 'w' in lbl[0]:
                # Homogenizing Variable
                x_sol += [np.array(1,ndmin=2)]
            elif 'x' in lbl[0]:
                # Pose Variable
                v = self.G.Vp[lbl[0]]
                if lbl[1] == 'C':
                    # Rotation - var = vec(C)
                    C = v.C_p0.copy()
                    x_sol += [C.reshape((-1,1),order='F')]
                else:
                    # Translation
                    if not self.pose_trans_in_world:
                        x_sol += [v.C_p0 @ v.r_in0]
                    else: # 
                        x_sol += [v.r_in0]
            else:
                # Map Variable
                v = self.G.Vp[lbl[0]]
                x_sol += [v.r_in0]
        # Return array
        x_sol_ = np.concatenate(x_sol,axis=0)
        return x_sol_
    
    def validate_constraints(self, useRedun=False, tol=1e-8):
        """Check that constraints are satisfied by the ground truth solution

        Args:
            useRedun ( bool ): Test redundant constraints as well
        """
        if len(self.constraints) == 0:
            raise Exception("No constraints to test")
        # Get solution
        x_sol = self.get_gt_soln()
        # Loop through constraints
        viol = []
        for c in self.constraints:
            A = c.A.get_matrix(variables=self.var_list)
            viol += [x_sol.T @ A @ x_sol - c.b]
        # Assert no violation
        check = np.abs(viol) < tol
        assert all(check), f"Constraints Violated: {[i for i,x in enumerate(check) if not x ]}"
        # Redundant constraints
        if useRedun:
            # Loop through constraints
            viol = []
            for c in self.constraints_r:
                A = c.A.get_matrix(variables=self.var_list)
                viol += x_sol.T @ A @ x_sol - c.b
            # Assert no violation
            check = np.abs(viol) < tol
            assert all(check), f"Redundant Constraints Violated: {[i for i,x in enumerate(check) if not x ]}"
        
        print("Constraints Validated!")
    
    def solve_primal_sdp(self, useRedun : bool=False, vars=None):
        """
        Solve the relaxed SDP for the original QCQP problem

        Args:
            useRedun (bool): When true, enables redundant constraints to tighten
                            problem.
        """
        # Variable list
        if vars is None:
            vars = self.var_list
        # Define SDP variable
        X = cp.Variable(self.Q.get_matrix(vars).shape,symmetric=True)
        # Define constraints
        #   Positive Semidefiniteness:
        constraints = [X >> 0]
        #   Primal Affine:
        assert len(self.constraints) > 0, "Constraints have not been generated."
        constraints += [cp.trace(c.A.get_matrix(vars) @ X) == c.b \
            for c in self.constraints]
        #   Redundant Affine:
        if useRedun:
            assert len(self.constraints_r) > 0, "Constraints have not been generated."
            constraints += [cp.trace(c.A.get_matrix(vars) @ X) == c.b \
                for c in self.constraints_r]
        # Condition the cost matrix
        Q = self.Q.get_matrix(vars)
        q_scale = sla.norm(Q)
        Q = Q / q_scale
        # Run CVX
        cprob = cp.Problem(cp.Minimize(cp.trace(Q @ X)), constraints)
        cprob.solve(solver=self.SDP_solvr,verbose=True)

        
        return X,cprob
        
    def round_sdp_soln(self,X):
        """Round the SDP solution using 

        Args:
            X (): Get rounded solution from SDP solution x
        """
        # SVD
        U,S,V = la.svd(X)
        # Compute solution from largest singular value
        x = U[:,[0]] * np.sqrt(S[0])
        # Find homogenizing var and flip if its negative.
        ind = 0
        for key in self.var_list.keys():
            if key == 'w_0':
                break
            else:
                ind += self.var_list[key]
        if x[ind] < 0:
              x = -x      
        return x
    
    def gauss_isotrp_meas_model(self, edgeList, sigma):
        """Generate isotropic Gaussian corrupted measurements based on a list of edges

        Args:
            edgeList (_type_): list of tuples of strings indicating edges
            sigma (_type_): Noise level
        """
        if np.abs(sigma) < 1e-9:
            W = np.eye(3)
        else:
            W = np.eye(3)/sigma**2
            
        for edge in edgeList:
            # Get vertices
            v1Lbl, v2Lbl = edge
            v1 = self.G.Vp[v1Lbl]
            v2 = self.G.Vm[v2Lbl]
            # Generate Measurement and add edge
            y = v1.C_p0 @ (v2.r_in0 - v1.r_in0) + sigma*np.random.randn(3,1)
            self.G.add_edge(v1,v2,y,W)
    
    def plot_ground_truth(self):
        ax = plt.axes(projection='3d')
        # Plot poses
        for v in self.G.Vp.values():
            pose = sm.SE3.Rt(v.C_p0,v.r_in0)
            pose.plot()
        # Plot map points
        for v in self.G.Vm.values():
            ax.scatter3D(v.r_in0[0],v.r_in0[1],v.r_in0[2])
        # Setup
        ax.axis('equal')
        plt.grid()
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_zlabel("Z-Axis")
        
        return ax
        
    def plot_measurements(self,ax):
        """Loop through measurements and plot points + error ellipsoid
        """
        for v1 in self.G.E.keys():
            for v2 in self.G.E[v1]:
                if isinstance(v2, MapVertex):
                    # Plot measurement in base frame
                    y_inP = self.G.E[v1][v2].meas['trans']
                    C_p0 = v1.C_p0
                    y_in0 = C_p0.T @ y_inP
                    ax.scatter3D(y_in0[0,0],y_in0[1,0],y_in0[2,0],marker='*',color='g')
                    # Rotate ellipsoid to base frame and plot
                    W = self.G.E[v1][v2].weight["trans"]
                    Cov = np.linalg.inv(W)
                    Cov_0 = C_p0.T @ Cov @ C_p0
                    plot_ellipsoid(y_in0, Cov_0, ax=ax)

                    
def plot_ellipsoid(bias, cov, ax=None, color='b',stds=1,label=None):
    if ax is None:
        ax = plt.axes(projection='3d')
    # Parameterize in terms of angles
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    L = np.linalg.cholesky(cov)
    ellipsoid = (stds*L @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(3, *x.shape)
    surf = ax.plot_surface(*ellipsoid, rstride=4, cstride=4, color=color, alpha=0.25,label=label)
    # These lines required for legend
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    return surf