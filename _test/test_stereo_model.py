#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mwcerts.stereo_problems import stereo_meas_model, Camera, Localization
from mwcerts.mat_weight_problem import plot_ellipsoid
import pytest


def test_covariance():
    # Number of test samples
    N = 1000
    # Set up a reasonable camera model
    c = Camera()
    c.b = 0.2
    c.f_u = 200
    c.f_v = 200
    c.sigma_u = 1
    c.sigma_v = c.sigma_u
    # Define points
    r_p = [np.zeros((3,1))]
    C = [np.eye(3)]
    # Map Point
    r_m = [np.array([[0,1,3]]).T]
    # Make graph
    loc = Localization(r_p,C,r_m)
    # Get Covariances by running model N times
    Sigma, Sigma_hat, Sigma_cam, Sigma_cam_hat, mu, X = \
        get_covariances(loc , N, c,lin_about_gt=True)
    # Plot Ellipsoids
    plt.figure()
    ax = plt.axes(projection='3d')
    plot_ellipsoid(mu, Sigma,ax=ax,color='r',stds=3,label="Computed $3\sigma$")
    plot_ellipsoid(mu, Sigma_hat,ax=ax,color='b',stds=3,label="Sample $3\sigma$")
    plt.axis('equal')
    ax.scatter(X[0,:],X[1,:],X[2,:],marker='.',color='k',s=2,alpha=0.2,label="Sample Points")
    ax.legend()
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

    # Check differences
    np.testing.assert_allclose(Sigma, Sigma_hat,rtol=0.05)
    
# This function runs N models and returns analytic and sample covariance
def get_covariances(loc : Localization, N, c : Camera, lin_about_gt : bool=False):
    # Seed RNG
    np.random.seed(0)
    # Run Trials
    X = np.zeros((3,N))
    X_cam = np.zeros((3,N))
    for i in range(N):
        u,v,d = stereo_meas_model(loc, [('x0','m0')], c, lin_about_gt=lin_about_gt)
        e = loc.G.get_edge('x0','m0')
        X[:,[i]] = e.meas['trans']
        X_cam[:,[i]] = np.array([[u,v,d]]).T
    # Get sample means:
    mu = np.mean(X,axis=1,keepdims=True)
    mu_cam = np.mean(X_cam,axis=1,keepdims=True)
    # Get sample covariances
    Sigma_hat = (X-mu) @ (X-mu).T / (N-1)
    Sigma_cam_hat = (X_cam-mu_cam) @ (X_cam-mu_cam).T / (N-1)
    # Get analytic covariances
    Sigma = np.linalg.inv(e.weight['trans'])
    Sigma_cam = np.array([[c.sigma_u**2, 0, c.sigma_u**2],
                          [0,     c.sigma_v**2, 0],
                          [c.sigma_u**2, 0 ,2*c.sigma_u**2]])

    return Sigma, Sigma_hat, Sigma_cam, Sigma_cam_hat, mu, X


if __name__ == "__main__":
    test_covariance() 



