#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mwcerts.mat_weight_problem import MatrixWeightedProblem, Constraint
from mwcerts.meas_graph import MapVertex
from poly_matrix.poly_matrix import PolyMatrix
import numpy as np
from scipy.linalg import issymmetric
import scipy.linalg as la
import spatialmath as sm



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
        
def stereo_meas_model(prb : MatrixWeightedProblem, edgeList : list, c : Camera,\
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
    return (u_list, v_list, d_list)
    
def pd_inv(a):
    n = a.shape[0]
    I = np.identity(n)
    return la.solve(a, I, sym_pos = True, overwrite_b = True)
    