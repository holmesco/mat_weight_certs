#!/usr/bin/env python3

import numpy as np
import mwcerts.mat_weight_problem as mw
import pytest


# Define Ground truth 
r_p = [np.array([[1,1,1]]).T]
r_p += [np.array([[2,2,2]]).T]
r_p += [np.array([[3,3,3]]).T]
C = [np.eye(3)*1.,np.eye(3)*2.,np.eye(3)*3.]
r_m = [np.array([[0,1,1]]).T]
r_m += [np.array([[0,2,2]]).T]
r_m += [np.array([[0,3,3]]).T]

def test_init():
    # Test init fail
    with pytest.raises(Exception):
        G = mw.MeasGraph(r_p[:2], C, r_m)
    # Define Graph
    G = mw.MeasGraph(r_p, C, r_m)
    # Check vertices
    assert [ k for k in G.Vp.keys()] == ['x0','x1','x2']
    assert [ k for k in G.Vm.keys()] == ['m0','m1','m2']
    # Check vertex value
    np.testing.assert_allclose(G.Vp['x0'].r_in0, np.array([[1,1,1]]).T)


def test_edge_creation():
    # Generate graph
    G = mw.MeasGraph(r_p, C, r_m)
    # Valid Edge Addtions
    G.add_edge('x0','m1',np.ones((3,1)), np.eye(3))
    G.add_edge('x1','m1',np.ones((3,1)), np.eye(3))
    G.add_edge('x1','x2',np.ones((3,1)), np.eye(3))
    G.add_edge('x1','x2',np.eye(3), np.eye(3),'rot')
    # Add rotations
    with pytest.raises(ValueError):
        G.add_edge('x0','x1',np.ones((3,1)), np.eye(3),'rot')
    # Should generate error
    with pytest.raises(Exception):
        G.add_edge('x23','m1',np.ones((3,1)), np.eye(3)) 
    with pytest.raises(Exception):
        G.add_edge('x1','m12',np.ones((3,1)), np.eye(3))
        
def test_edge_retrieval():
    # Generate graph
    G = mw.MeasGraph(r_p, C, r_m)
    # Valid Edge Addtions
    G.add_edge('x1','x2',np.ones((3,1)), np.eye(3))
    G.add_edge('x1','x2',np.eye(3), np.eye(3),'rot')
    # Rotations
    np.testing.assert_allclose(G.get_edge('x1','x2').meas['rot'],np.eye(3), \
        err_msg="Edge Rotatation Meas")
    np.testing.assert_allclose(G.get_edge('x1','x2').weight['rot'],np.eye(3), \
        err_msg="Edge Rotatation Weight")
    # Translations
    np.testing.assert_allclose(G.get_edge('x1','x2').meas['trans'],np.ones((3,1)), \
        err_msg="Edge Translation Meas")
    np.testing.assert_allclose(G.get_edge('x1','x2').weight['trans'],np.eye(3), \
        err_msg="Edge Translation Weight")
    # Update
    G.add_edge('x1','x2',3*np.ones((3,1)), np.eye(3),'trans')
    np.testing.assert_allclose(G.get_edge('x1','x2').meas['trans'],3*np.ones((3,1)), \
        err_msg="Edge Translation Meas")

    

# if __name__ == "__main__":
#     test_init()
#     test_edge_creation()
#     test_edge_retrieval()
    
