#!/usr/bin/env python

#------------------------------------------------------------------------------
# matrix_perturbation.py
#
# author:   A. Vaidya
# version:  1.0
# date:     Nov 26, 2024
# 
# tested with Python v3.10
# 
# Solves the pairing model for four particles by perturbatively expanding the 
# Magnus operator first order. Found using H(0)/Delta(0), through element-wise 
# division. Needs to be fixed.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, pi
from scipy.linalg import eigvalsh, expm
from scipy.special import bernoulli
from commutators import commutator_2b, similarity_transform
from generators import eta_white
from basis import *
from classification import *

from sys import argv

#-----------------------------------------------------------------------------------
# normal-ordered pairing Hamiltonian
#-----------------------------------------------------------------------------------
def pairing_hamiltonian(delta, g, user_data):
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
  
  return H1B, H2B
  
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
# Main Program
#-----------------------------------------------------------------------------------
def main():
  # grab delta and g from the command line
  delta      = float(argv[1])
  g          = float(argv[2])

  particles  = 4

  # setup shared data
  dim1B     = 8

  # this defines the reference state
  # 1st state
  holes     = [0,1,2,3]
  particles = [4,5,6,7]

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

    "order":      20,                 # variables for magnus series expansions
    "bernoulli":   0,                 # and lists of Bernoulli numbers
    "hamiltonian": 0,                 # stored starting Hamiltonian
  }

  # initialize Bernoulli numbers for magnus expansion
  user_data["bernoulli"] = bernoulli(user_data["order"])

  # set up initial Hamiltonian
  H1B, H2B = pairing_hamiltonian(delta, g, user_data)

  E, f, Gamma  = normal_order(H1B, H2B, user_data) 

  # Calculate starting metrics
  DE2          = calc_mbpt2(f, Gamma, user_data)
  DE3          = calc_mbpt3(f, Gamma, user_data)
  norm_fod     = calc_fod_norm(f, user_data)
  norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)
  
  user_data["hamiltonian"]  = np.append([E], np.append(reshape(f, -1), reshape(Gamma, -1)))

  print("%-8s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s   %-14s"%(
    "step No", "E" , "DE(2)", "DE(3)", "E+DE", "||Omega||", "||fod||", "||Gammaod||"))
  print("-" * 148)

  print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
      0, E , DE2, DE3, E+DE2+DE3, 0, norm_fod, norm_Gammaod))

  max_steps = 10
  for s in range(1, max_steps):
    # Construct Delta and Omega for each step using Omega = Hod(0)/Delta(0) = eta_W(0)
    Omega1B, Omega2B = eta_white(f, Gamma, user_data)

    # Use Magnus evolution to obtain new E, f, Gamma
    E, f, Gamma = similarity_transform(Omega1B, Omega2B, E, f, Gamma, user_data)

    # Calculate new metrics
    OmegaNorm    = np.linalg.norm(Omega1B,ord='fro')+np.linalg.norm(Omega2B,ord='fro')
    DE2          = calc_mbpt2(f, Gamma, user_data)
    DE3          = calc_mbpt3(f, Gamma, user_data)
    norm_fod     = calc_fod_norm(f, user_data)
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

    # Print new metrics
    print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
      s, E , DE2, DE3, E+DE2+DE3, OmegaNorm, norm_fod, norm_Gammaod))


#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()

