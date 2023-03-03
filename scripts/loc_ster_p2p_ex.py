#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import os
import sys
import spatialmath.base as sm
from mwcerts.stereo_problems import Localization, Camera, stereo_meas_model
from poly_matrix.poly_matrix import PolyMatrix
from run_test import run_loc_test
import argparse


   
def loc_stereo_p2p_exp(Nm:int=10,use_redun:bool=False, sigma_p2p:float=0.,results:str=None):
    
    # Seed RNG
    np.random.seed(0)
    # Ground Truth Translation
    r_p = []
    r_p += [np.array([[0,0,0]]).T]
    r_p += [np.array([[0.5,0,0]]).T]
    # Ground Truth Rotations
    C_p = []
    C_p += [sm.roty(0)]
    C_p += [sm.roty(-0.1)]

    # Ground Truth Map Points
    offs = np.array([[0,0,2]]).T
    r_l = 0.5*np.eye(3) @ np.random.randn(3,Nm) + offs
    r_l = np.expand_dims(r_l.T,axis=2)
    # Generate problem instance
    loc = Localization(r_p,C_p,r_l)
    fig = plt.figure()
    ax = loc.plot_ground_truth()
    # Test Ground truth values
    x = loc.get_gt_soln()
    print(loc.var_list)
    print(f"Length of GT vars: {len(x)}")
    # Define edges
    edges = []
    for i in range(len(r_p)):
        for j in range(len(r_l)):
            edges += [(f'x{i}',f"m{j}")]
    # Define camera model
    camera = Camera()
    camera.f_u = 200
    camera.f_v = 200
    camera.c_u = 0
    camera.c_v = 0
    camera.b = 0.3
    camera.sigma_u = 0.5
    camera.sigma_v = 0.5
    # Generate measurements
    u,v,d = stereo_meas_model(loc,edges,camera)
    # Add translation measurements
    loc.add_p2p_meas([('x0','x1')],mtype='trans',sigma=sigma_p2p)
    loc.plot_measurements(ax)
    if not results is None:
        plt.savefig(results + 'situational.png')

    # Run tests
    run_loc_test(loc, results, use_redun=use_redun)
    
    if results is None:
        print('showing results')
        plt.show()
        
if __name__ == "__main__":
    
    # **** Parameters ****
    use_redun = False
    Nm = 20
    sigma_p2p = 0.01
    # ************************
    
    par = argparse.ArgumentParser()
    par.add_argument('-num_lm', type=int, default=Nm,required=False)
    par.add_argument('-sigma_p2p', type=float, default=sigma_p2p,required=False)
    par.add_argument('-use_redun',type=bool,default=use_redun,required=False)
    par.add_argument('-save_results',type=bool,default=False,required=False)
    args = par.parse_args()
    # Create results Folder
    if args.save_results:
        results = os.path.dirname(__file__) + '/LocStereoResults/loc_ster_p2p'
        if not args.use_redun:
            results += '/'
        else:
            results += '_r/'
    else:
        results = None
    Nm=args.num_lm
    sigma_p2p=args.sigma_p2p
    use_redun = args.use_redun
    loc_stereo_p2p_exp(Nm=Nm, sigma_p2p=sigma_p2p,results=results,use_redun=use_redun)