#!/usr/bin/env python3

import matplotlib.pyplot as plot
from mat_weight_problem import MatrixWeightedProblem, Constraint
from meas_graph import MapVertex
from poly_matrix.poly_matrix import PolyMatrix
import numpy as np


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
                y, W = self.G.E[v1][v2]
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
                    Q_e[C1, t1] = np.kron(p , W)
                    Q_e[C1, w0] = np.kron(p, W @ y)
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
            # Init constraint
            A = PolyMatrix()
            for i in range(3):
                for j in range(i,3):
                    E_ = np.zeros((3,3))
                    E_[i,j] = 1./2.
                    A[C,C] = E_ + E_.T
                    A[w0,w0] = -1.
                    self.constraints += [Constraint(A,0,"O3")]

        # Homogenization 
        A = PolyMatrix()
        A[w0,w0] = 1.
        self.constraints += [Constraint(A, 1., "Homog")]
    
    
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
            y = v1.C_p0 @ (v2.r_in0 - v1.r_in0) + np.random.randn(3,1)
            self.G.addEdge(v1,v2,y,W)