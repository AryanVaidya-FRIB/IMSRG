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

#-----------------------------------------------------------------------------------
# Design Matrix Constructor
#-----------------------------------------------------------------------------------
def make_design(y0, sfinal, ds, user_data):
  # Generates list of Hamiltonians for early times to construct POD
  dim1B = user_data["dim1B"]

  ys_list = [y0]
  dys_list = []
    # integrate flow equations 
  solver = ode(derivative_wrapper,jac=None)
  solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
  solver.set_f_params(user_data)
  solver.set_initial_value(y0, 0.)
  if user_data["model"] != "Galerkin":
    dys_list.append(solver.f(solver.t, solver.y, user_data))

  print("Constructing list of snapshots")
  print("%-8s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s"%(
      "s", "E" , "DE(2)", "DE(3)", "E+DE", "dE/ds", 
      "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print("-" * 148)
  
  eta_norm0 = 1.0e10
  failed = False

  while solver.successful() and solver.t < sfinal:
      ys = solver.integrate(solver.t+ds)
  
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
      solver.t, E , DE2, DE3, E+DE2+DE3, user_data["dE"], user_data["eta_norm"], norm_fod, norm_Gammaod))
      if abs(DE2/E) < 10e-8: break

      eta_norm0 = user_data["eta_norm"]

      ys_list.append(ys)
      if user_data["model"] != "Galerkin":
          dys_list.append(solver.f(solver.t+ds, ys, user_data))

  return ys_list, dys_list

#-----------------------------------------------------------------------------------
# Operator Inference Calculations
#-----------------------------------------------------------------------------------
def OpInf_model(ys_list, dys_list, params):
  Xs  = []
  dXs = []
  for flow, dflow in zip(ys_list, dys_list):
    Xs.append(np.vstack(flow).transpose())
    dXs.append(np.vstack(dflow).transpose())

  X = np.hstack(Xs)
  print(X.shape)

  # Use OpInf for calculations - construct basis (similar to Galerkin projection)
  basis = opinf.basis.PODBasis(svdval_threshold=1e-10)
  basis.fit(X)
  print(basis)
  r = basis.shape[1]

  # Fit X, Xdot to quadratic order
  Xs_  = []
  dXs_ = []
  for X, Xdot in zip(Xs, dXs):
    Xs_.append(basis.compress(X))
    dXs_.append(basis.compress(Xdot))
  model = opinf.models.InterpContinuousModel(
      operators = "cAH",
      solver=opinf.lstsq.L2Solver(regularizer=1e-8)
      ).fit(parameters = params, states = Xs_, ddts = dXs_)
  
  return basis, model, r

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------
def main():
  # grab delta and g from the command line - edited for batch compatibility
  delta      = 1
  g          = [-0.5, 0.1, 0.5]
  b          = [-0.5, 0.1, 0.5]
  model      = "OpInf"

  # Number of particles
  particles   = 4
  # Length of each POD flow
  sPod        = 0.5
  flow_length = 50
  full_rank   = flow_length*(len(g)+len(b))
  print(f"ROM will use {full_rank} snapshots maximum.")

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

    "calc_eta":   eta_imtime,          # specify the generator (function object)
    "calc_rhs":   commutator_2b,      # specify the right-hand side and truncation
    "model":      model               # projection model to construct ROM
  }

  # Define starting RAM use
  tracemalloc.start()
  time_start = time.perf_counter()

  # Get number of time steps to iterate over
  ds_pod = sPod/flow_length
  ys_list = []
  dys_list = []
  params_list = []

  # Construct POD matrix - integration happens in make_design()
  for g_val in g:
    for b_val in b:
      # set up initial Hamiltonian
      if g_val == 0 and b_val == 0:
        continue
      print(f"Now creating snapshots for g={g_val}, b={b_val}")
      H1B, H2B = pairing_hamiltonian(delta, g_val, b_val, user_data)
      E, f, Gamma = normal_order(H1B, H2B, user_data) 

      # reshape Hamiltonian into a linear array (initial ODE vector)
      y0   = np.append([E], np.append(reshape(f, -1), reshape(Gamma, -1)))
      ys_temp, dys_temp = make_design(y0, sPod, ds_pod, user_data)
      print(len(ys_temp))
      ys_list.append([ys_temp])
      dys_list.append([dys_temp])
      params_list.append([g_val,b_val])
  Ur = 0
  reduced = 0

  # Make ROM matrix
  print(f"Constructing ROM using {model} model type.")
  basis, mod, r = OpInf_model(ys_list, dys_list, params_list)

  # Get memory use from making POD
  total_time = time.perf_counter()-time_start
  pod_memkb_current, pod_memkb_peak = tracemalloc.get_traced_memory()
  pod_memkb_peak = pod_memkb_peak/1024.
  tracemalloc.stop()
  
  print(f"RAM Use: {pod_memkb_peak} kb\nTime Spent: {total_time} s")
  oiPath = outpath+f"OpInf_Parametric_s{sPod}_rank{r}_N4"
  os.mkdir(oiPath)
  basis.save(oiPath+"/basis.h5")
  mod.save(oiPath+"/model.h5")
  
#    solver.integrate(solver.t + ds)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()

