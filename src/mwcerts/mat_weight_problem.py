#!/usr/bin/env python3

import numpy as np
from poly_matrix.poly_matrix import PolyMatrix
from mwcerts.meas_graph import MeasGraph, MapVertex, Vertex
import cvxpy as cp
import scipy.linalg as la
import scipy.sparse as sp
import spatialmath as sm
# Plotting
import matplotlib.pyplot as plt
from pylgmath.pylgmath.se3.transformation import Transformation as Trans


class Constraint:
    def __init__(self, A : PolyMatrix, b : float , label : str):
        self.A = A
        self.b = b
        self.label = label
        
class GaussNewtonOpts:
    def __init__(self):
        self.tol_grad_norm_sq = 1e-10
        self.max_iter = 200

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
                elif lbl[1] == "t":
                    # Translation pose frame
                    x_sol += [v.C_p0 @ v.r_in0]
                elif lbl[1] == "t0":
                    # Translation world frame
                    x_sol += [v.r_in0]
                else:
                    raise Exception(f"Variable {var} Unknown")
            elif "m" in lbl[0]:
                # Map Variable
                v = self.G.Vp[lbl[0]]
                x_sol += [v.r_in0]
            else:
                raise Exception(f"Variable {var} Unknown")
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
                viol += [x_sol.T @ A @ x_sol - c.b]
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
        q_scale = sp.linalg.norm(Q)
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
    
    def gauss_newton(self, opt : GaussNewtonOpts=GaussNewtonOpts(), x_init=None):
        """ Run Gauss-Newton algorithm for the localization problem

        Args:
            opt (GaussNewtonOpts, optional): Options to be used with GN. Defaults to GaussNewtonOpts().
            x_init (_type_, optional): Initialization vector. Defaults to None => all zeros.

        Returns:
            x_sol : dictionary of Gauss-Newton solution variables
            info : dictionary of useful optimization outputs.
        """
        # define variable index list
        var_inds={}
        n_vars = 0
        for var in self.var_list.keys():
            # Only assign variables for rotations or translations
            if not "w" in var:
                var_inds[var]=list(range(n_vars*3, (n_vars+1)*3))
                n_vars += 1
                     
        # Init GN: variable is stored in the "Lie algebra" form according to
        # the vertex list vert_list
        if x_init is None:
            x_init = np.zeros((n_vars*3,1))
        else:
            if len(x_init.shape) == 1:
                x_init = np.expand_dims(x_init,1)
            assert x_init.shape[0] == n_vars*3, "Initialixation vector has wrong number of variables"
        x = x_init
        grad_norm_sq = np.Inf
        n_iter = 0
        # Main loop
        print("| Iteration | Grd Nrm Sq |   Cost    |")
        while grad_norm_sq > opt.tol_grad_norm_sq and n_iter < opt.max_iter:
            # Init list of values in Jacobian and Weight Matrices
            J_rows = np.array([])
            J_cols = np.array([])
            J_vals = np.array([])
            W_rows = np.array([])
            W_cols = np.array([])
            W_vals = np.array([])
            err = []
            cnt = 0
            # Loop through edges in factor graph
            for v1 in self.G.E.keys():
                for v2 in self.G.E[v1].keys():
                    # variable indicies (addition of lists)
                    inds = var_inds[v1.label+"_t"] + var_inds[v1.label+"_C"]
                    # Get relevant variables
                    xi = x[inds]
                    y_ba_a = self.G.E[v1][v2].meas['trans'] 
                    T_a0 = Trans(xi_ab=xi)
                    p_b0_0 = np.vstack((v2.r_in0,[[1]]))
                    # Construct error
                    p_ba_a = T_a0 @ p_b0_0
                    err += [ y_ba_a - p_ba_a[0:3,[0]] ]
                    # Construct Jacobian and get weight
                    J_ba = -circ_dot(p_ba_a)
                    W_ba = self.G.E[v1][v2].weight['trans'] 
                    # Store sparse values
                    nz = np.nonzero(J_ba)
                    J_rows=np.append(J_rows, cnt*3+nz[0])
                    J_cols=np.append(J_cols, [inds[i] for i in nz[1]])
                    J_vals=np.append(J_vals, J_ba[nz])
                    nz = np.nonzero(W_ba)
                    W_rows=np.append(W_rows, cnt*3+nz[0])
                    W_cols=np.append(W_cols, cnt*3+nz[1])
                    W_vals=np.append(W_vals, W_ba[nz])                    
                    # error counter
                    cnt += 1
            # Assemble error
            err_vec = np.vstack(err)
            # Construct sparse matrices
            J = sp.coo_matrix((J_vals, (J_rows, J_cols)), shape=(len(err_vec),n_vars*3))
            W = sp.coo_matrix((W_vals, (W_rows, W_cols)), shape=(len(err_vec),len(err_vec)))
            # Rescale Problem to avoid numerical issues
            w_scl = sp.linalg.norm(W)
            W /= w_scl
            # Compute gradiant
            Grad = - J.T @ W @ err_vec
            grad_norm_sq = np.linalg.norm(Grad)
            Hessian = J.T @ W @ J
            # Compute and apply update
            del_x = sp.linalg.spsolve(Hessian, Grad)
            del_x=np.expand_dims(del_x,1)
            for v1 in self.G.E.keys():
                # varialble indicies (addition of lists)
                inds = var_inds[v1.label+"_t"] + var_inds[v1.label+"_C"]
                # Get relevant variables
                xi = x[inds]
                del_xi = del_x[inds]
                T_a0 = Trans(xi_ab=xi)
                T_a0_new = Trans(xi_ab=del_xi) @ T_a0
                x[inds] = T_a0_new.vec()
            # Update and status
            n_iter += 1
            cost = err_vec.T @ W @ err_vec * w_scl
            print(f"| {n_iter:9.4e} | ",f"{grad_norm_sq:9.4e} | ",f"{cost[0,0]:9.4e} |")
            
        # convert solution to expected format based on variable list
        x_sol = {}
        for var in self.var_list.keys():
            if var == "w_0":
                x_sol[var]=np.array([[1]])
            elif "t" in var:
                label = var.split("_")[0]
                inds = var_inds[label+"_t"] + var_inds[label+"_C"]
                T_ba = Trans(xi_ab=x[inds])
                x_sol[label+'_C'] = np.reshape(T_ba.C_ba(),(9,1),order='F')
                x_sol[label+'_t'] = T_ba.C_ba() @ T_ba.r_ba_ina()
                x_sol[label+'_T_io'] = T_ba.matrix()
        # Solution info
        info = {}
        info['cost'] = cost
        info['grad_norm_sq'] = grad_norm_sq
        info['x_lie'] = x
        info['n_iter'] = n_iter
        
        return x_sol, info
                          
    
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

# UTILITY FUNCTIONS
def plot_ellipsoid(bias, cov, ax=None, color='b',stds=1,label : str=None):
    """ Plot a 3D ellipsoid

    Args:
        bias (_type_): offset to center of ellipsoid
        cov (_type_): covariance matrix associated with ellipsoid
        ax (_type_, optional): axis handle. Defaults to None.
        color (str, optional): color of ellipsoid. Defaults to 'b'.
        stds (int, optional): Number of standard deviations for ellipsoid. Defaults to 1.
        label (str): Label string for plotting. Defaults to None.

    Returns:
        _type_: _description_
    """
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

def circ_dot(homog_vec):
    return np.hstack( (homog_vec[3,[0]]*np.eye(3) , -skew(homog_vec[0:3,0])) )    
def skew(vec): 
    return np.array([[0, -vec[2], vec[1]],[vec[2], 0, -vec[0]],[-vec[1], vec[0], 0]])           
