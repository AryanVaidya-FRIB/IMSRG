#!/usr/bin/env python

#------------------------------------------------------------------------------
# IMSRG_perturbation.py
#
# author:   A. Vaidya
# version:  1.3
# date:     Jan 16, 2025
# 
# tested with Python v3.10
# 
# Solves the P3H model for four particles by perturbatively expanding the 
# Magnus operator first and second orders. Found using H(0)/Delta(0), 
# through element-wise division.
# Currently configured for benchmarking, loops through values of g and saves
# computational values.
#
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from numpy import array, dot, diag, reshape, pi
from scipy.linalg import eigvalsh, expm
from scipy.special import bernoulli
from commutators import commutator_2b, similarity_transform, BCH
from generators import eta_white
from basis import *
from classification import *

from sys import argv
import time
import tracemalloc

#-----------------------------------------------------------------------------------
# normal-ordered pairing Hamiltonian
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

def separate_diag(A_1b, A_2b, user_data):
  # Separate the 2B operator A into diagonal and off-diagonal elements
  # Note that inputs could be Hermitian (Gamma) or anti-Hermitian (commutator pieces). Explicitly assign all values.
  dim1B     = user_data["dim1B"]
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # Initialize output arrays
  Ad_1b = np.zeros_like(A_1b)
  Aod_1b = np.zeros_like(A_1b)
  Ad_2b = np.zeros_like(A_2b)
  Aod_2b = np.zeros_like(A_2b)

  # Separate 1-body operator
  for a in particles:
    for i in holes:
      Ad_1b[a,a] = A_1b[a,a]
      Ad_1b[i,i] = A_1b[i,i]

      Aod_1b[a,i] = A_1b[a,i]
      Aod_1b[i,a] = A_1b[i,a]

  # Separate 2-body operator
  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          # First the diagonal pieces - don't need to reverse the pp and hh states since the loop will go over them
          Ad_2b[idx2B[(a,i)], idx2B[(a,i)]] = A_2b[idx2B[(a,i)], idx2B[(a,i)]]
          Ad_2b[idx2B[(i,a)], idx2B[(i,a)]] = A_2b[idx2B[(i,a)], idx2B[(i,a)]]
          Ad_2b[idx2B[(i,j)], idx2B[(i,j)]] = A_2b[idx2B[(i,j)], idx2B[(i,j)]]
          Ad_2b[idx2B[(a,b)], idx2B[(a,b)]] = A_2b[idx2B[(a,b)], idx2B[(a,b)]]

          # Now the off-diagonal - just taking the abij and ijab
          Aod_2b[idx2B[(a,b)], idx2B[(i,j)]] = A_2b[idx2B[(a,b)], idx2B[(i,j)]]
          Aod_2b[idx2B[(i,j)], idx2B[(a,b)]] = A_2b[idx2B[(i,j)], idx2B[(a,b)]]
  
  return Ad_1b, Aod_1b, Ad_2b, Aod_2b

def get_second_order_Omega(f, Gamma, user_data):
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # Since the commutator is between purely 2-body pieces, we need a zero one-body array
  zero_1b = np.zeros_like(f)

  # Separate Gamma into diagonal and off-diagonal elements and construct a ratio array
  _, _, Gamma_d, Gamma_od = separate_diag(zero_1b, Gamma, user_data)
  ratio = np.zeros_like(Gamma_od)

  # Calculate the denominator operator
  denom_1b = np.zeros_like(f)
  for a in particles:
    for i in holes:
      denom_1b[a,i] = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]

  denom_2b = np.zeros_like(Gamma)
  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom_2b[idx2B[(a,b)], idx2B[(i,j)]] = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )
          # catch small or zero denominator - matrix element inspired by arctan version
          if abs(denom_2b[idx2B[(a,b)], idx2B[(i,j)]])<1.0e-10:
            val = 0.25 * pi * np.sign(Gamma[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(denom_2b[idx2B[(a,b)], idx2B[(i,j)]])
          else:
            val = Gamma_od[idx2B[(a,b)], idx2B[(i,j)]] / denom_2b[idx2B[(a,b)], idx2B[(i,j)]]
          # Calculate ratio right here - will be used immediately
          ratio[idx2B[(a,b)], idx2B[(i,j)]] = val
          ratio[idx2B[(i,j)], idx2B[(a,b)]] = -val

  # Calculate necessary commutators and extract off-diagonals
  _, J_1b, J_2b = commutator_2b(zero_1b, ratio, zero_1b, Gamma_od, user_data)
  _, K_1b, K_2b = commutator_2b(zero_1b, ratio, zero_1b, Gamma_d, user_data)

  _, Jod_1b, _, Jod_2b = separate_diag(J_1b, J_2b, user_data)
  _, Kod_1b, _, Kod_2b = separate_diag(K_1b, K_2b, user_data)

  # Construct output Omegas
  Omega1b_2 = np.zeros_like(f)
  Omega2b_2 = np.zeros_like(Gamma)
  for a in particles:
    for i in holes:
      if abs(denom_1b[a,i])<1.0e-10:
        val = (0.125 * pi * np.sign(Jod_1b[a,i]) * np.sign(denom_1b[a,i])
        -0.25 * pi * np.sign(Kod_1b[a,i])*np.sign(denom_1b[a,i])
        )
      else:
        val = 0.5*(Jod_1b[a,i]/denom_1b[a,i])-(Kod_1b[a,i]/denom_1b[a,i])
      
      Omega1b_2[a,i] = val
      Omega1b_2[i,a] = -val

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          if abs(denom_2b[idx2B[(a,b)], idx2B[(i,j)]])<1.0e-10:
            val = (
              0.125 * pi * np.sign(Jod_2b[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(denom_2b[idx2B[(a,b)], idx2B[(i,j)]])
              - 0.25 * pi * np.sign(Kod_2b[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(denom_2b[idx2B[(a,b)], idx2B[(i,j)]])
            )
          else:
            val = (
              0.5*(Jod_2b[idx2B[(a,b)], idx2B[(i,j)]] / denom_2b[idx2B[(a,b)], idx2B[(i,j)]])
              -(Kod_2b[idx2B[(a,b)], idx2B[(i,j)]] / denom_2b[idx2B[(a,b)], idx2B[(i,j)]])
            )
          
          Omega2b_2[idx2B[(a,b)], idx2B[(i,j)]] = val
          Omega2b_2[idx2B[(i,j)], idx2B[(a,b)]] = -val

  return Omega1b_2, Omega2b_2

def get_operator_from_y(y, dim1B, dim2B):
  
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  one_body = reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

  ptr += dim1B*dim1B
  two_body = reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

  return zero_body,one_body,two_body

#-----------------------------------------------------------------------------------
# Main Program
#-----------------------------------------------------------------------------------
def main():

  # grab delta and g from the command line
  delta      = 1.0 #float(argv[1])
#  g          = float(argv[2])
  b          = 0.4828 #float(argv[3])

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
    "omegaNorm":  1e10,
    "ref_energy": 0,

    "order":      20,                 # variables for magnus series expansions
    "bernoulli":   0,                 # and lists of Bernoulli numbers
    "hamiltonian": 0,                 # stored starting Hamiltonian
  }

  # initialize Bernoulli numbers for magnus expansion
  user_data["bernoulli"] = bernoulli(user_data["order"])

  for i in range(-13,13):
    # Initialize value of g
    g = i/10
    print(f"g = {g}")

    # Define starting RAM use
    tracemalloc.start()

    # begin timer
    time_start = time.perf_counter()

    # set up initial Hamiltonian
    H1B, H2B = pairing_hamiltonian(delta, g, b, user_data)

    E, f, Gamma  = normal_order(H1B, H2B, user_data) 
    E_i = E
    f_i = f
    Gamma_i = Gamma

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

    max_steps = 20
    FinalOmega1B = np.zeros_like(f)
    FinalOmega2B = np.zeros_like(Gamma)
    #Omegas1B = []
    #Omegas2B = []
    for s in range(1, max_steps):
      # Construct Delta and Omega for each step using Omega = Hod(0)/Delta(0) = eta_W(0)
      Omega1B, Omega2B = eta_white(f, Gamma, user_data)
      # Construct the second order Omega - check the formula
      Omega1B_2, Omega2B_2 = get_second_order_Omega(f, Gamma, user_data)
      fullOmega1B = Omega1B + Omega1B_2
      fullOmega2B = Omega2B + Omega2B_2
      OmegaNorm    = np.linalg.norm(fullOmega1B,ord='fro')+np.linalg.norm(fullOmega2B,ord='fro')
      FinalOmega1B, FinalOmega2B = BCH(fullOmega1B, fullOmega2B, FinalOmega1B, FinalOmega2B, user_data)

      # Use Magnus evolution to obtain new E, f, Gamma
      E, f, Gamma = similarity_transform(fullOmega1B, fullOmega2B, E, f, Gamma, user_data)
      if abs(OmegaNorm - user_data["omegaNorm"]) < 1e-5 or abs(E-user_data["ref_energy"]) < 1e-4:
        break

    #  Omegas1B.append(fullOmega1B)
    #  Omegas2B.append(fullOmega2B)

      # Update user_data
      user_data["omegaNorm"] = OmegaNorm
      user_data["ref_energy"] = E

      # Calculate new metrics
      DE2          = calc_mbpt2(f, Gamma, user_data)
      DE3          = calc_mbpt3(f, Gamma, user_data)
      norm_fod     = calc_fod_norm(f, user_data)
      norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

      # Print new metrics
      print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f"%(
        s, E , DE2, DE3, E+DE2+DE3, OmegaNorm, norm_fod, norm_Gammaod))


    # Check final value using all stored Magnus operators
  #  E_s = E_i
  #  f_s = f_i
  #  Gamma_s = Gamma_i
  #  for i in range(len(Omegas2B)):
  #    E_s, f_s, Gamma_s = similarity_transform(Omegas1B[i], Omegas2B[i], E_s, f_s, Gamma_s, user_data)

    # Check final value using BCH stored operator (only one operator to consider)
    E_s, f_s, Gamma_s = similarity_transform(FinalOmega1B, FinalOmega2B, E_i, f_i, Gamma_i, user_data)
    print(f"Final GSE: {E_s}")

    # Get g-value diagnostics
    current_time = time.perf_counter()-time_start
    memkb_current, memkb_peak = tracemalloc.get_traced_memory()
    memkb_peak = memkb_peak/1024.

    glist.append(g)
    final_E.append(E_s)
    final_step.append(s)
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
  
  output.to_csv('imsrg-white_d1.0_b+0.4828_N4_perturbative2BCH.csv')


#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()

