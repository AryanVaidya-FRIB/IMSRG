#!/usr/bin/env python

#------------------------------------------------------------------------------
# imsrg_pairing.py
#
# author:   H. Hergert 
# version:  1.5.1
# date:     Jul 6, 2020
# 
# tested with Python v2.7 and v3.7
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of an In-Medium Similarity Renormalization 
# Group (IMSRG) flow.
#
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
from numpy import array, dot, diag, reshape, transpose
from scipy.linalg import eigvalsh
from scipy.integrate import odeint, ode
from math import pi
from basis import *
from classification import *
from generators import *
from commutators import *

from sys import argv
import time
import tracemalloc

#-----------------------------------------------------------------------------------
# derivative wrapper
#-----------------------------------------------------------------------------------
def get_operator_from_y(y, dim1B, dim2B):
  
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
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

#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------

def main():
  # grab delta and g from the command line - edited for batch compatibility
  delta      = float(argv[1])
  g          = float(argv[2])
  b          = float(argv[3])

  particles  = 4

  # Construct output arrays
  glist = []
  final_E = []
  final_step = []
  total_time = []
  total_RAM = []

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
    "calc_rhs":   commutator_2b         # specify the right-hand side and truncation
  }

  # Define starting RAM use
  tracemalloc.start()

  # Start the clock
  time_start = time.perf_counter()

  # set up initial Hamiltonian
  H1B, H2B = pairing_hamiltonian(delta, g, b, user_data)

  E, f, Gamma = normal_order(H1B, H2B, user_data) 

  # reshape Hamiltonian into a linear array (initial ODE vector)
  y0   = np.append([E], np.append(reshape(f, -1), reshape(Gamma, -1)))

  # integrate flow equations 
  solver = ode(derivative_wrapper,jac=None)
  solver.set_integrator('vode', method='bdf', order=5, nsteps=1000)
  solver.set_f_params(user_data)
  solver.set_initial_value(y0, 0.)

  sfinal = 50
  ds = 0.01

  sList = []
  EList = []
  GammaList = []
  fullSet = []

  print("%-8s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s"%(
    "s", "E" , "DE(2)", "DE(3)", "E+DE", "dE/ds", 
    "||eta||", "||fod||", "||Gammaod||"))
  # print "-----------------------------------------------------------------------------------------------------------------"
  print("-" * 148)
  
  eta_norm0 = 1.0e10
  failed = False

  while solver.successful() and solver.t < sfinal:
    ys = solver.integrate(sfinal, step=True)
#    fullSet.append(ys)
  
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
    if abs(DE2/E) < 1e-6: break # 1e-9 before
    sList.append(solver.t)
    EList.append(E)
    GammaList.append(norm_Gammaod)

    eta_norm0 = user_data["eta_norm"]
    
  # Get g-value diagnostics
  current_time = time.perf_counter()-time_start
  memkb_current, memkb_peak = tracemalloc.get_traced_memory()
  memkb_peak = memkb_peak/1024.

  glist.append(g)
  final_E.append(E)
  final_step.append(solver.t)
  total_time.append(current_time)
  total_RAM.append(memkb_peak)
  print(f"Loop Time: {current_time} sec. RAM used: {memkb_peak} kb.")
  tracemalloc.stop()

  output = pd.DataFrame({
    'g':           glist,
    'Ref Energy':  final_E,
    'Total Steps': final_step,
    'Total Time':  total_time,
    'RAM Usage':   total_RAM
  })

  step_output = pd.DataFrame({
    "s":           sList,
    "E":           EList,
    "Gammaod":     GammaList
  })
  
  output.to_csv(f'imsrg-white_d{delta}_g{g}_b{b}_N4_ev1.csv')
  step_output.to_csv(f'imsrg-white_d{delta}_g{g}_b{b}_N4_ev1_fullflow.csv')

#  with open('allHs.pkl', 'wb') as fp:
#      pickle.dump(fullSet, fp)

#    solver.integrate(solver.t + ds)

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()