#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mwcerts.mat_weight_problem import MatrixWeightedProblem, Constraint
from mwcerts.meas_graph import MapVertex, Vertex
from poly_matrix.poly_matrix import PolyMatrix
import numpy as np
from scipy.linalg import issymmetric
import scipy.linalg as la
import spatialmath as sm
import scipy.sparse as sp
from pylgmath.pylgmath.se3.transformation import Transformation as Trans

class GaussNewtonOpts:
    def __init__(self):
        self.tol_grad_norm_sq = 1e-10
        self.max_iter = 200
        

class Camera():
    def __init__(camera):
        camera.f_u = 200
        camera.f_v = 200
        camera.c_u = 0
        camera.c_v = 0
        camera.b = 0.05
        camera.sigma_u = 0.5
        camera.sigma_v = 0.5

class Localization(MatrixWeightedProblem):
    """Localization problem with map points known. 
    It is assumed that there are no pose-to-pose measurements.
    Poses: C_io - Rotation matrix from world to pose frame
           t_io_i  - Translation from world to pose frame in pose frame
    Map Points: p_io_o - Translation from world to map point in world frame
    Measurements: y_ki_i - Translation from world  

    """
    def __init__(self, r_p_in0, C_p_0, r_m_in0):
        # Initialization Matrix weighted problem
        super().__init__(r_p_in0, C_p_0, r_m_in0)
        # Generate Constraints
        self.generate_constraints()
        # Default variable list ordering
        for v in self.G.Vp.values():
            self.var_list[v.label+'_C']=9
            self.var_list[v.label+'_t']=3
        # for v in self.G.Vm.values():
        #     self.var_list += [v.label]
        self.var_list['w_0'] = 1
    
    def generate_cost(self):
        """
        Generate cost function matrix based on the edges in the edge graph
        """
        # Loop through edges in measurement graph and define cost matrix
        for v1 in self.G.E.keys():
            for v2 in self.G.E[v1].keys():
                # Get measurement and weight
                y = self.G.E[v1][v2].meas['trans']
                W = self.G.E[v1][v2].weight['trans']
                # Define dummy poly matrix for cost element
                Q_e = PolyMatrix()
                # Homogenization variable
                w0 = 'w_0'
                if not isinstance(v2,MapVertex) :
                    # Pose to Pose measurement
                    raise Exception('StereoLocalization problem should not have pose to pose measurements')
                else:
                    # Pose to Map measurement
                    # Define labels
                    C1 = v1.label + '_C'
                    t1 = v1.label + '_t'
                    # Get Map point 
                    p = v2.r_in0
                    # Define matrix
                    Q_e[C1, C1] = np.kron(p @ p.T, W)
                    Q_e[C1, t1] = -np.kron(p , W) 
                    Q_e[C1, w0] = -np.kron(p, W @ y) 
                    Q_e[t1, t1] = W
                    Q_e[t1, w0] = W @ y
                    Q_e[w0, w0] = y.T @ W @ y   
                # Add to cost matrix
                self.Q += Q_e
                
    def generate_constraints(self):
        """
        Generate the constraint matrices for each variable
        """
        # Homogenization variable
        w0 = 'w_0'
        # O(3) constraints on pose rotations
        for v in self.G.Vp.values():
            # Get label
            C = v.label + '_C'
            # Generate six orthogonality constraints           
            for i in range(3):
                for j in range(i,3):
                    A = PolyMatrix()
                    E = np.zeros((3,3))
                    E[i,j] = 1./2.
                    A[C,C] = np.kron(E + E.T, np.eye(3))
                    if i == j:
                        A[w0,w0] = -1.
                    else:
                        A[w0,w0] = 0.
                    self.constraints += [Constraint(A,0.,"O3")]
        # Homogenization 
        A = PolyMatrix()
        A[w0,w0] = 1.
        self.constraints += [Constraint(A, 1., "Homog")]
        
    def init_gauss_newton(self, sigma=0.):
        """Generate ground truth vector with gaussian noise in lie algebra.

        Args:
            sigma (float, optional): Standard Deviation for Gaussian Noise. Defaults to 0..

        Returns:
            numpy array : initialization vector
        """
        x_init = []
        for var in self.var_list:
            if not "w" in var:
                label = var.split("_")[0]
                v = self.G.Vp[label]
                xi = Trans(C_ba=v.C_p0, r_ba_in_a=v.r_in0).vec()
                if "t" in var:
                    x_init += [ xi[:3,[0]] + sigma*np.random.randn(3,1) ]
                elif "C" in var:
                    x_init += [ xi[3:, [0]] + sigma*np.random.randn(3,1) ]
        return np.vstack(x_init)
        
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
                    # varialble indicies (addition of lists)
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
                          
def circ_dot(homog_vec):
    return np.hstack( (homog_vec[3,[0]]*np.eye(3) , -skew(homog_vec[0:3,0])) )
    
def skew(vec): 
    return np.array([[0, -vec[2], vec[1]],[vec[2], 0, -vec[0]],[-vec[1], vec[0], 0]])           

def stereo_meas_model(prb : MatrixWeightedProblem, edgeList : list, c : Camera, \
        lin_about_gt : bool=False):
    # Loop over edges in pose to measurement graph
    u_list = []
    v_list = []
    d_list = []
    for edge in edgeList:
        # Get vertices
        v1Lbl, v2Lbl = edge
        v1 = prb.G.Vp[v1Lbl]
        v2 = prb.G.Vm[v2Lbl]
        # Get Euclidean Measurement
        meas = v1.C_p0 @ (v2.r_in0 - v1.r_in0)
        #Define pixel values and disparity
        u_bar = c.f_u*meas[0,0]/meas[2,0]+c.c_u
        v_bar = c.f_v*meas[1,0]/meas[2,0]+c.c_v
        d_bar = c.f_u*c.b/meas[2,0]
        # Add noise (separate noise vector for u defined 
        # so that the noise is actually correlated)
        u_noise = c.sigma_u*np.random.randn(2,1)
        u = u_bar + u_noise[0,0]
        v = v_bar + c.sigma_v*np.random.randn(1)[0]
        d = d_bar + np.sum(u_noise,axis = 0)[0]
        # TODO there should be probably be some filtering based on
        # negative disparity
        # Pixel and disparity covariance
        Sigma_cam = np.array([[c.sigma_u**2, 0, c.sigma_u**2],
                            [0,     c.sigma_v**2, 0],
                            [c.sigma_u**2, 0 ,2*c.sigma_u**2]])
        # Define covariance. We assume that the
        # measured values are close to their means and linearize about
        # them.
        if lin_about_gt:
            G = np.array([[c.b/d_bar , 0, -c.b*(u_bar  - c.c_u)/d_bar **2],
                    [0 , c.f_u*c.b/c.f_v/d_bar, -c.f_u/c.f_v*c.b*(v_bar  - c.c_v)/d_bar **2],
                    [0, 0, -c.f_u*c.b/d_bar **2]])
        else:
            G = np.array([[c.b/d, 0, -c.b*(u - c.c_u)/d**2],
                    [0 , c.f_u*c.b/c.f_v/d, -c.f_u/c.f_v*c.b*(v - c.c_v)/d**2],
                    [0, 0, -c.f_u*c.b/d**2]])
        Cov = G @ Sigma_cam @ G.T
        # Invert to get weighting (in camera frame) - Force to be symmetric
        W = pd_inv(Cov)
        W = (W + W.T)/2
        if not issymmetric(W,atol=1e-13):
            raise ValueError("Weight Matrix not Symmetric")
        # Re-estimate euclidian position
        r = np.array([[(u - c.c_u)*c.b/d, c.f_u/c.f_v*(v-c.c_v)*c.b/d, c.f_u*c.b/d]]).T
        # Add edge to measurement graph
        prb.G.add_edge(v1,v2,r,W)
        # Store list of pixel points
        u_list += [u]
        v_list += [v]
        d_list += [d]
    u_list = np.array(u_list)
    v_list = np.array(v_list)
    d_list = np.array(d_list)
    return u_list, v_list, d_list
    
def pd_inv(a):
    n = a.shape[0]
    I = np.identity(n)
    return la.solve(a, I, sym_pos = True, overwrite_b = True)
    