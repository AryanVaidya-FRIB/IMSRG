#!/usr/bin/env python

#------------------------------------------------------------------------------
# imsrg_pod.py
#
# author:   A. Vaidya 
# version:  1.0
# date:     April 15, 2025
# 
# tested with Python v3.10
# 
# Performs the offline stage for constructing the ROM
#
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pysindy as ps
import opinf
from numpy import array, dot, diag, reshape, transpose
from scipy.linalg import eigvalsh, svd
from scipy.integrate import odeint, ode
from sklearn.decomposition import IncrementalPCA
from math import pi
import pickle

from basis import *
from classification import *
from generators import *
from commutators import *

import os
from sys import argv
import time
import tracemalloc

#-----------------------------------------------------------------------------------
# pairing Hamiltonian
#-----------------------------------------------------------------------------------
def pairing_hamiltonian(delta, g, b, user_data):
  bas1B = user_data["bas1B"]
  bas2B = user_data["bas2B"]
  idx2B = user_data["idx2B"]

  dim = len(bas1B)
  H1B = np.zeros((dim,dim))

  for i in bas1B:
    H1B[i,i] = delta*np.floor_divide(i, 2)

  dim = len(bas2B)
  H2B = np.zeros((dim, dim))

  # spin up states have even indices, spin down the next odd index
  for (i, j) in bas2B:
    if (i % 2 == 0 and j == i+1):
      for (k, l) in bas2B:
        if (k % 2 == 0 and l == k+1):
          H2B[idx2B[(i,j)],idx2B[(k,l)]] = -0.5*g
          H2B[idx2B[(j,i)],idx2B[(k,l)]] = 0.5*g
          H2B[idx2B[(i,j)],idx2B[(l,k)]] = 0.5*g
          H2B[idx2B[(j,i)],idx2B[(l,k)]] = -0.5*g

  # pair-breaking contributions
  for (i, j) in bas2B:
    if (i % 2 == 0 and j % 2 == 1 and j != i+1):
      for (k, l) in bas2B:
        if (k % 2 == 0 and l == k+1):
          H2B[idx2B[(i,j)],idx2B[(k,l)]] = -0.5*b
          H2B[idx2B[(j,i)],idx2B[(k,l)]] = 0.5*b
          H2B[idx2B[(i,j)],idx2B[(l,k)]] = 0.5*b
          H2B[idx2B[(j,i)],idx2B[(l,k)]] = -0.5*b
          H2B[idx2B[(k,l)],idx2B[(i,j)]] = -0.5*b
          H2B[idx2B[(k,l)],idx2B[(j,i)]] = 0.5*b
          H2B[idx2B[(l,k)],idx2B[(i,j)]] = 0.5*b
          H2B[idx2B[(l,k)],idx2B[(j,i)]] = -0.5*b
  
  return H1B, H2B

#-----------------------------------------------------------------------------------
# normal-ordered pairing Hamiltonian
#-----------------------------------------------------------------------------------
def normal_order(H1B, H2B, user_data):
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # 0B part
  E = 0.0
  for i in holes:
    E += H1B[i,i]

  for i in holes:
    for j in holes:
      E += 0.5*H2B[idx2B[(i,j)],idx2B[(i,j)]]  

  # 1B part
  f = H1B
  for i in bas1B:
    for j in bas1B:
      for h in holes:
        f[i,j] += H2B[idx2B[(i,h)],idx2B[(j,h)]]  

  # 2B part
  Gamma = H2B

  return E, f, Gamma

#-----------------------------------------------------------------------------------
# derivative wrapper
#-----------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, dim2B):  
  # reshape the solution vector into 0B, 1B, 2B pieces
  #print(y)
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  #print(y[ptr:ptr+dim1B*dim1B])
  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

  ptr += dim1B*dim1B
  two_body = reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

  return zero_body,one_body,two_body


def derivative_wrapper(t, y, user_data):

  dim1B = user_data["dim1B"]
  dim2B = dim1B*dim1B


  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  basph2B   = user_data["basph2B"]
  idx2B     = user_data["idx2B"]
  idxph2B   = user_data["idxph2B"]
  occA_2B   = user_data["occA_2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]
  calc_eta  = user_data["calc_eta"]
  calc_rhs  = user_data["calc_rhs"]

  # extract operator pieces from solution vector
  E, f, Gamma = get_operator_from_y(y, dim1B, dim2B)


  # calculate the generator
  eta1B, eta2B = calc_eta(f, Gamma, user_data)

  # calculate the right-hand side
  dE, df, dGamma = calc_rhs(eta1B, eta2B, f, Gamma, user_data)

  # convert derivatives into linear array
  dy   = np.append([dE], np.append(reshape(df, -1), reshape(dGamma, -1)))

  # share data
  user_data["dE"] = dE
  user_data["eta_norm"] = np.linalg.norm(eta1B,ord='fro')+np.linalg.norm(eta2B,ord='fro')
  
  return dy

#------------------------------------------------------------------------------
# Streaming operations
#------------------------------------------------------------------------------
def incremental_svd(ys, rank_data, tol=1e-10, tol_reorth=1e-8):
  U      = rank_data["U"]
  S      = rank_data["S"]
  Vh     = rank_data["Vh"]
  r      = rank_data["r"]
  ys     = np.array(ys).reshape(-1,1)
  W      = np.identity(ys.shape[0])

  d = U.T @ W @ ys
  if not np.shape(d):
    d *= np.eye(1, 1)
  e = ys - U @ d
  p = np.sqrt(e.T @ W @ e) * np.eye(1, 1)

  if p < tol:
    p = np.zeros((1, 1))
  else:
    e = e / p[0, 0].item()

  k = np.shape(S)[0] if np.shape(S) else 1
  Y = np.vstack((np.hstack((S, d)), np.hstack((np.zeros((1, k)), p))))
  Uy, Sy, Vhy = np.linalg.svd(Y, full_matrices=False, compute_uv=True)
  Sy = np.diag(Sy)

  l = np.shape(Vh)[0]
  if p < tol:
    U = U @ Uy[:k, :k]
    S = Sy[:k, :k]
    Vh = np.vstack((np.hstack((Vh, np.zeros((l, 1)))), np.hstack(
        (np.zeros((1, k)), np.eye(1))))) @ Vhy[:, :k]
  else:
    U = np.hstack((U, e)) @ Uy
    S = Sy
    Vh = np.vstack((np.hstack((Vh, np.zeros((l, 1)))),
                np.hstack((np.zeros((1, k)), np.eye(1))))) @ Vhy
      
  if np.abs(U[:, -1].T @ W @ U[:, 0]) > tol_reorth:
    k = U.shape[1]
    for i in range(k):
      a = U[:, i]
      for j in range(i):
          U[:, i] = U[:, i] - ((a.T @ W @ U[:, j]) / (U[:, j].T @ W @ U[:, j])) * U[:, j]
      norm = np.sqrt(U[:, i].T @ W @ U[:, i])
      U[:, i] = U[:, i] / norm

  rank_data["U"] = U
  rank_data["S"] = S
  rank_data["Vh"] = Vh
#  print(Vh.shape)

  return rank_data

def form_phi(q):
  # Constructs Phi matrix for RLS - constant, linear, Kronecker product
  return np.concatenate(([1.0],q,np.kron(q,q)))

def initialize_RLS(rank_data, tol=6e-1):
#  print("Initializing RLS")
  S     = rank_data["S"]
  U     = rank_data["U"]
#  print(np.diag(S))
  r     = 6
  Ur    = U[:,:r]
#  print(f"Will construct a rank {r} ROM.")

  rank_data["r"]  = r
  alpha = rank_data["alpha"]
  rank_data["m"] = 1+r+r**2
  # Initializes rank_data for RLS fitting
  rank_data["Ur"] = Ur
  rank_data["w"]  = np.zeros((rank_data["r"],rank_data["m"]))
  rank_data["P"]  = np.array([(1/alpha)*np.identity(rank_data["m"]) for _ in range(rank_data["r"])])
#  print("Starting RLS Fitting")
  return rank_data, True

def fit_RLS(ys, dys, rank_data):
  Ur = rank_data["Ur"]
  P = rank_data["P"]
  w = rank_data["w"]
  ff = rank_data["ff"]
  r = rank_data["r"]
  # Current projected form
#  print(U.shape)
  qs  = Ur.T @ ys
  dqs = Ur.T @ dys

  # Form RLS prediction
  phi_s = form_phi(qs).reshape(-1,1)

  for i in range(r):
    dq_i    = dqs[i]
    P_i     = P[i]
    w_i     = w[i]
#    print(phi_s.shape)
#    print(P_i.shape)

    denom = ff+ phi_s.T @ P_i @ phi_s
    K_i = ((P_i @ phi_s) / denom).reshape(-1,1)
#   print(K_i.shape)
    error = dq_i - w_i.T @ phi_s
#   print(error.shape)
#   print(w[i].shape)
    w[i] += (K_i * error)[:,0]
    P[i] = (P_i - K_i @ phi_s.T @ P_i) / ff
  
#  dq_pred = (w @ phi_s).flatten()
#  print("True dq:", dqs)
#  print("Predicted dq:", dq_pred)
#  print("Error:", np.linalg.norm(dqs-dq_pred))
  
  rank_data["w"] = w
  rank_data["P"] = P

  return rank_data


#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------
def main():
  # grab delta and g from the command line - edited for batch compatibility
  delta      = float(argv[1])
  g          = float(argv[2])
  b          = float(argv[3])
  model      = str(argv[4])

  # Number of particles
  particles   = 4
  # Length of each POD flow
  sPod        = 1.0
  flow_length = 100

  # IO commands
  outpath    = "/mnt/c/Users/aryan/Documents/MSU_FRIB/IMSRG/proper_orth_decomp/ROMs/"

  # Construct output arrays
  glist = []
  final_E = []
  final_step = []
  total_time = []
  total_RAM = []
  POD_RAM = []

  # setup shared data
  dim1B     = 8

  # this defines the reference state
  # 1st state
  holes     = [0,1,2,3]
  particles = [4,5,6,7]

  # 2nd state
  # holes     = [0,1,4,5]
  # particles = [2,3,6,7]

  # 3rd state
  # holes     = [0,1,6,7]
  # particles = [2,3,4,5]

  # basis definitions
  bas1B     = range(dim1B)
  bas2B     = construct_basis_2B(holes, particles)
  basph2B   = construct_basis_ph2B(holes, particles)

  idx2B     = construct_index_2B(bas2B)
  idxph2B   = construct_index_2B(basph2B)

  # occupation number matrices
  occ1B     = construct_occupation_1B(bas1B, holes, particles)
  occA_2B   = construct_occupationA_2B(bas2B, occ1B)
  occB_2B   = construct_occupationB_2B(bas2B, occ1B)
  occC_2B   = construct_occupationC_2B(bas2B, occ1B)

  occphA_2B = construct_occupationA_2B(basph2B, occ1B)

  # store shared data in a dictionary, so we can avoid passing the basis
  # lookups etc. as separate parameters all the time
  user_data  = {
    "dim1B":      dim1B, 
    "holes":      holes,
    "particles":  particles,
    "bas1B":      bas1B,
    "bas2B":      bas2B,
    "basph2B":    basph2B,
    "idx2B":      idx2B,
    "idxph2B":    idxph2B,
    "occ1B":      occ1B,
    "occA_2B":    occA_2B,
    "occB_2B":    occB_2B,
    "occC_2B":    occC_2B,
    "occphA_2B":  occphA_2B,

    "eta_norm":   0.0,                # variables for sharing data between ODE solver
    "dE":         0.0,                # and main routine

    "calc_eta":   eta_imtime,         # specify the generator (function object)
    "calc_rhs":   commutator_2b,      # specify the right-hand side and truncation
    "model":      model               # projection model to construct ROM
  }

  # Define starting RAM use
  tracemalloc.start()
  time_start = time.perf_counter()

  # Get number of time steps to iterate over
  ds_pod = sPod/flow_length

  # Construct POD matrix - integration happens in make_design()
  # set up initial Hamiltonian
  H1B, H2B = pairing_hamiltonian(delta, g, b, user_data)
  E, f, Gamma = normal_order(H1B, H2B, user_data) 

  # reshape Hamiltonian into a linear array (initial ODE vector)
  y0   = np.append([E], np.append(reshape(f, -1), reshape(Gamma, -1)))
  solver = ode(derivative_wrapper,jac=None)
  solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
  solver.set_f_params(user_data)
  solver.set_initial_value(y0, 0.)

  print("Building POD Basis")
  print("%-8s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s"%(
    "s", "E" , "DE(2)", "DE(3)", "E+DE", "dE/ds", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print("-" * 143)
  
  eta_norm0 = 1.0e10
  failed = False
  orig  = y0.T @ np.identity(y0.shape[0]) @ y0

  # Dictionary of stored rank information
  rank_data = {
    "full rank":  flow_length,                        # Total number of snapshots to calculate
    "r":          0,                                  # Maximum projected rank

    "U":          (y0/orig).reshape(-1,1),             # U in the SVD
    "S":          orig.reshape(-1,1),                                # S in the SVD 
    "Vh":         np.eye(1, 1),                        # V in the SVD
    "Ur":         0,                                   # Galerkin projection basis

    "n":          len(y0),                             # Full space dimension
    "m":          10,                                  # Proposed RLS model dimension (modified later)
    "ff":         0.995,                               # Forgetting factor in RLS
    "alpha":      1e-5,                                # P scaling factor

    "w":          0,                                   # Model Parameter vector
    "P":          0                                    # Inverse Correlation Matrix
  }

  have_init = False

#  yList = [y0]
#  dyList = []

  qList  = []
  dqList = []
  tol_reorth = 1e-8

  while solver.successful() and solver.t < sPod:
      ys  = solver.integrate(solver.t+ds_pod)
      dys = solver.f(solver.t+ds_pod, ys, user_data)   
      if solver.t < sPod/2:
        rank_data = incremental_svd(ys, rank_data, tol_reorth=tol_reorth)
#        yList.append(ys)
#        dyList.append(dys)
      
#      if solver.t > sPod/2 and not have_init:
#        rank_data, have_init = initialize_RLS(rank_data)
      if solver.t > sPod/2 and not have_init:
        r = np.sum(np.diag(rank_data["S"]) > tol_reorth)
        print("Will constructed reduced order model with rank", r)
        rank_data["r"] = r
        rank_data["Ur"] = rank_data["U"][:,:r]
        have_init = True
      
      if solver.t > sPod/2:
        qList.append(rank_data["Ur"].T @ ys)
        dqList.append(rank_data["Ur"].T @ dys)
#        rank_data = fit_RLS(ys, dys, rank_data)
      
      if user_data["eta_norm"] > 1.25*eta_norm0: 
          failed=True
          break
  
      dim2B = dim1B*dim1B
      E, f, Gamma = get_operator_from_y(ys, dim1B, dim2B)

      DE2 = calc_mbpt2(f, Gamma, user_data)
      DE3 = calc_mbpt3(f, Gamma, user_data)

      norm_fod     = calc_fod_norm(f, user_data)
      norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

      print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
      solver.t, E , DE2, DE3, E+DE2+DE3, user_data["dE"], user_data["eta_norm"], norm_fod, norm_Gammaod ))
      if abs(DE2/E) < 1e-6: break # 1e-9 before

      eta_norm0 = user_data["eta_norm"]

  # Note that c = Theta[:,0], A = Theta[:,1:1+r], H = Theta[:,1+r:]
  r = rank_data["r"]
  Ur = rank_data["Ur"]
  w = rank_data["w"]
  
  basis = opinf.basis.LinearBasis(Ur, check_orthogonality=True)
  """
  model = opinf.models.ContinuousModel(
    operators = [
      opinf.operators.ConstantOperator(w[:,0]),
      opinf.operators.LinearOperator(w[:,1:1+r]),
      opinf.operators.QuadraticOperator(w[:,1+r:])
    ]
  )
  """
  # Make ROM matrix
  print(f"Constructing ROM using {model} model type.")
  X_ = np.vstack(qList).T
  Xdot_ = np.vstack(dqList).T
  print(rank_data["U"].shape)
  print(rank_data["S"].shape)
  print(rank_data["Vh"].shape)
  """
  X_approx = rank_data["U"] @ rank_data["S"] @ rank_data["Vh"].T
  error = np.linalg.norm(X_-X_approx)/np.linalg.norm(X_)
  print("Reconstruction Error: ", error)
  U, _, _ = np.linalg.svd(X_)
  r = np.sum(np.diag(rank_data["S"]) > 1e-8)
  Ur = U[:,:6]
  Ur_approx = rank_data["U"][:,:r]
  error = np.linalg.norm(Ur @ Ur.T-Ur_approx @ Ur_approx.T)/np.linalg.norm(Ur @ Ur.T)
  print("Low order incremental error: ", error)
  print("Low order rank: ",r)
  """

  model = opinf.models.ContinuousModel(
    operators = "cAH",
    solver=opinf.lstsq.L2Solver(regularizer=1e-8)
  ).fit(states = X_, ddts = Xdot_)

  # Get memory use from making POD
  total_time = time.perf_counter()-time_start
  pod_memkb_current, pod_memkb_peak = tracemalloc.get_traced_memory()
  pod_memkb_peak = pod_memkb_peak/1024.
  tracemalloc.stop()
  print(f"RAM Use: {pod_memkb_peak} kb\nTime Spent: {total_time} s")
  
  oiPath = outpath+f"OpInf_Streaming_d{delta}_g{g}_b{b}_s{sPod}_rank{r}_N4"
  os.mkdir(oiPath)
  basis.save(oiPath+"/basis.h5")
  model.save(oiPath+"/model.h5")
  
#    solver.integrate(solver.t + ds)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()

