#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import numpy.linalg as la
import pandas as pd
import spatialmath.base as sm
import os
from datetime import datetime
from mwcerts.stereo_problems import Localization, Camera, stereo_meas_model
from progress.bar import ChargingBar


#  ***** PARAMETERS ******
# Number of map points
num_map_pts_list = [4,5,7,9,11,15,20,25]
# p2p Noise level
sigma_vals = np.logspace(-3,-1,5)
# Camera Distance 
z_vals = np.linspace(1,5,5)
# rank tolerance
tol_rank = 1e-5
# Redundant constraint enable
use_redun = True
#**************************
   
# Seed RNG
np.random.seed(0)
# Ground Truth Translation
r_p = []
r_p += [np.array([[0,0,0]]).T]
r_p += [np.array([[0.1,0,0]]).T]
# Ground Truth Rotations
C_p = []
C_p += [sm.roty(0)]
C_p += [sm.roty(-0.1)]

# Define camera model
camera = Camera()
camera.f_u = 200
camera.f_v = 200
camera.c_u = 0
camera.c_v = 0
camera.b = 0.3
camera.sigma_u = 0.5
camera.sigma_v = 0.5

#  ***** PARAMETERS ******
# Number of map points
num_map_pts_list = [4,5,7,9,11,15,20,25,35]
# p2p Noise level
sigma_vals = np.logspace(-3,-1,5)
# Camera Distance 
z_vals = np.linspace(1,10,5)
# rank tolerance
tol_rank = 1e-5
# Trials
num_trials = 10
# Redundant constraint enable
use_redun = True
#**************************

# Wait bar
num_runs = num_trials * len(num_map_pts_list) * len(sigma_vals) * len(z_vals)
bar = ChargingBar('Running Analysis:', max=num_runs,suffix = '%(percent).1f%% - %(eta)ds')
# Loop
df_rows = []
for Nm in num_map_pts_list:
    for sigma_p2p in sigma_vals:
        for z_val in z_vals:
            for trial_num in range(num_trials):
                # Ground Truth Map Points 
                offs = np.array([[0,0,z_val]]).T
                r_l = 0.5*np.eye(3) @ np.random.randn(3,Nm) + offs
                r_l = np.expand_dims(r_l.T,axis=2)
                # Generate problem instance
                loc = Localization(r_p,C_p,r_l)

                # Define edges (fully connected)
                edges = []
                for i in range(len(r_p)):
                    for j in range(len(r_l)):
                        edges += [(f'x{i}',f"m{j}")]

                # MEASUREMENTS
                u,v,d = stereo_meas_model(loc,edges,camera)
                # Add translation measurements
                loc.add_p2p_meas([('x0','x1')],mtype='trans',sigma=sigma_p2p)
                # Generate cost function
                loc.generate_cost()
                Q = loc.Q.get_matrix()
                
                # Run Gauss Newton - initialize with ground truth
                x_init=loc.init_gauss_newton(sigma=0.0)
                x_sol,info = loc.gauss_newton(x_init=x_init)
                x_gn = np.vstack([x_sol[var] for var in loc.var_list.keys()])

                # Run SDP
                X,cprob = loc.solve_primal_sdp(use_redun=use_redun)
                # Check Rank of solution
                U,S,V = la.svd(X.value)
                
                rank = np.sum(S >= tol_rank)
                # Optimality Gap
                cost_lcl = x_gn.T @ Q @ x_gn
                cost_sdp = np.trace(Q @ X.value)
                gap = (cost_lcl-cost_sdp)/cost_sdp
                
                # Append to dataframe
                data = {'trial_num' : trial_num, 
                        'num_map_pts': Nm,
                        'sigma_p2p' : sigma_p2p,
                        'z_dist': z_val,
                        'sing_vals' : S,
                        'rank' : rank,
                        'cost_sdp' : cost_sdp,
                        'cost_lcl' : cost_lcl,
                        'gap' : gap,
                        'info' : info}
                df_rows.append(data)
                bar.next()
bar.finish()

# Create Dataframe
df = pd.DataFrame(df_rows)
# Save to CSV
csv_file = os.path.dirname(__file__) + '/LocStereoResults/loc_ster_p2p'
now = datetime.now()
datestr = f"-{now.year}-{now.month}-{now.day}-T{now.hour}-{now.minute}"
csv_file += datestr
csv_file += ".csv"
df.to_csv(csv_file)