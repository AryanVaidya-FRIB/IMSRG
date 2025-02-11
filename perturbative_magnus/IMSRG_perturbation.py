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

#-----------------------------------------------------------------------------------
# Helper function to split diagonal and off-diagonal elements
#-----------------------------------------------------------------------------------

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
          # strict diagonal terms - pqpq
          #Ad_2b[idx2B[(a,b)], idx2B[(a,b)]] = A_2b[idx2B[(a,b)], idx2B[(a,b)]]
          #Ad_2b[idx2B[(a,i)], idx2B[(a,i)]] = A_2b[idx2B[(a,i)], idx2B[(a,i)]]
          #Ad_2b[idx2B[(i,a)], idx2B[(i,a)]] = A_2b[idx2B[(i,a)], idx2B[(i,a)]]
          #Ad_2b[idx2B[(i,j)], idx2B[(i,j)]] = A_2b[idx2B[(i,j)], idx2B[(i,j)]]
          # The off-diagonal - pphh and hhpp states
          Aod_2b[idx2B[(a,b)], idx2B[(i,j)]] = A_2b[idx2B[(a,b)], idx2B[(i,j)]]
          Aod_2b[idx2B[(i,j)], idx2B[(a,b)]] = A_2b[idx2B[(i,j)], idx2B[(a,b)]]

  # Diagonal 2-body operator (I think this works)
  Ad_2b = A_2b-Aod_2b # (Includes terms not in the strict definition, so like aibj, abcd, ijkl terms)

  return Ad_1b, Aod_1b, Ad_2b, Aod_2b

#-----------------------------------------------------------------------------------
# Perturbative Magnus Operators
#-----------------------------------------------------------------------------------

def Delta(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # Calculate the denominator operator
  denom_1b = np.zeros_like(f)
  for p in range(dim1B):
    for q in range(p+1, dim1B):
      denom_1b[p,q] = f[p,p] - f[q,q] + Gamma[idx2B[(p,q)],idx2B[(p,q)]] 

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

  return denom_1b, denom_2b

def zeta(f, Gamma, delta1B, delta2B, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the operator - loops over all pq, not just ai
  zeta1B  = np.zeros_like(f)

  for p in range(dim1B):
    for q in range(p+1,dim1B):
      # catch small or zero denominator - matrix element inspired by arctan version
      if abs(delta1B[p,q])<1.0e-10:
        val = 0.25 * pi * np.sign(f[p,q]) * np.sign(delta1B[p,q])
      else:
        val = f[p,q]/delta1B[p,q]

      zeta1B[p, q] =  val
      zeta1B[q, p] = -val 

  # two-body part of the generator - same as White's generator
  zeta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          # catch small or zero denominator - matrix element inspired by arctan version
          if abs(delta2B[idx2B[(a,b)], idx2B[(i,j)]])<1.0e-10:
            val = 0.25 * pi * np.sign(Gamma[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(delta2B[idx2B[(a,b)], idx2B[(i,j)]])
          else:
            val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / delta2B[idx2B[(a,b)], idx2B[(i,j)]]

          zeta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          zeta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return zeta1B, zeta2B


def get_second_order_Omega(f, Gamma, delta1B, delta2B, user_data):
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # Since the commutator will have 2 1-body inputs, we need a zero one-body array
  zero_1b = np.zeros_like(f)

  # Separate Gamma into diagonal and off-diagonal elements and construct a ratio array
  #print("Diagonal for 2B Hamiltonian")
  f_d, f_od, Gamma_d, Gamma_od = separate_diag(f, Gamma, user_data)

  eta1B, eta2B = eta_white(f, Gamma, user_data)

  # Calculate necessary commutators and extract off-diagonals
  _, J_1b, J_2b = commutator_2b(eta1B, eta2B, f_od, Gamma_od, user_data)
  _, K_1b, K_2b = commutator_2b(eta1B, eta2B, zero_1b, Gamma_d, user_data)

  #print("Separated Diagonals for od-od Magnus operator")
  _, Jod_1b, _, Jod_2b = separate_diag(J_1b, J_2b, user_data)
  #print("Separated Diagonals for od-d Magnus operator")
  _, Kod_1b, _, Kod_2b = separate_diag(K_1b, K_2b, user_data)

  # Construct output Omegas
  Omega1b_2 = np.zeros_like(f)
  Omega2b_2 = np.zeros_like(Gamma)
  for a in particles:
    for i in holes:
      if abs(delta1B[a,i])<1.0e-10:
        val = (0.125 * pi * np.sign(Jod_1b[a,i]) * np.sign(delta1B[a,i])
        +0.25 * pi * np.sign(Kod_1b[a,i])*np.sign(delta1B[a,i])
        )
      else:
        val = 0.5*(Jod_1b[a,i]/delta1B[a,i])+(Kod_1b[a,i]/delta1B[a,i])
      
      Omega1b_2[a,i] = val
      Omega1b_2[i,a] = -val

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          if abs(delta2B[idx2B[(a,b)], idx2B[(i,j)]])<1.0e-10:
            val = (
              0.125 * pi * np.sign(Jod_2b[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(delta2B[idx2B[(a,b)], idx2B[(i,j)]])
              + 0.25 * pi * np.sign(Kod_2b[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(delta2B[idx2B[(a,b)], idx2B[(i,j)]])
            )
          else:
            val = (
              0.5*(Jod_2b[idx2B[(a,b)], idx2B[(i,j)]] / delta2B[idx2B[(a,b)], idx2B[(i,j)]])
              +(Kod_2b[idx2B[(a,b)], idx2B[(i,j)]] / delta2B[idx2B[(a,b)], idx2B[(i,j)]])
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
  delta      = float(argv[1])
  g          = float(argv[2])
  b          = float(argv[3])

  # Initialize starting setup
  use_second_order = True
  store_operators  = True

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

  # Initialize value of g
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

  print("%-8s   %-14s   %-14s   %-14s   %-14s   %-14s  %-14s  %-14s  %-14s"%(
    "step No", "E" , "DE(2)", "DE(3)", "E+DE", "||Omega1||", "||Omega2||", "||fod||", "||Gammaod||"))
  print("-" * 148)

  print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f   %14.8f   %14.8f  %14.8f  %14.8f"%(
      0, E , DE2, DE3, E+DE2+DE3, 0, 0, norm_fod, norm_Gammaod))

  max_steps = 20
  # Initialize output parameters before assignment
  if store_operators:
    Omegas1B = []
    Omegas2B = []
  else:
    FinalOmega1B = np.zeros_like(f)
    FinalOmega2B = np.zeros_like(Gamma)
  for s in range(1, max_steps):
    # Construct Delta and Omega for each step using Omega = Hod(0)/Delta(0) = eta(0)
    Omega1B, Omega2B = eta_white(f, Gamma, user_data)
    fullOmega1B = Omega1B
    fullOmega2B = Omega2B
    OmegaNorm1  = np.linalg.norm(Omega1B,ord='fro')+np.linalg.norm(Omega2B,ord='fro')
    OmegaNorm   = OmegaNorm1
    if use_second_order:
      # Construct the second order Omega - check the formula
      delta1B, delta2B = Delta(f, Gamma, user_data)
      Omega1B_2, Omega2B_2 = get_second_order_Omega(f, Gamma, delta1B, delta2B, user_data)
      fullOmega1B += Omega1B_2
      fullOmega2B += Omega2B_2
      OmegaNorm2  = np.linalg.norm(Omega1B_2,ord='fro')+np.linalg.norm(Omega2B_2,ord='fro')
      OmegaNorm   += OmegaNorm2
    
    if store_operators:
      Omegas1B.append(fullOmega1B)
      Omegas2B.append(fullOmega2B)
    else:
      FinalOmega1B, FinalOmega2B = BCH(fullOmega1B, fullOmega2B, FinalOmega1B, FinalOmega2B, user_data)

    # Use Magnus evolution to obtain new E, f, Gamma
    E, f, Gamma = similarity_transform(fullOmega1B, fullOmega2B, E, f, Gamma, user_data)
    if abs(OmegaNorm - user_data["omegaNorm"]) < 1e-5 or abs(E-user_data["ref_energy"]) < 1e-4:
      break

    # Update user_data
    user_data["omegaNorm"] = OmegaNorm
    user_data["ref_energy"] = E

    # Calculate new metrics
    DE2          = calc_mbpt2(f, Gamma, user_data)
    DE3          = calc_mbpt3(f, Gamma, user_data)
    norm_fod     = calc_fod_norm(f, user_data)
    norm_Gammaod = calc_Gammaod_norm(Gamma, user_data)

    # Print new metrics
    if use_second_order:
      print("%8.5f %14.8f   %14.8f   %14.8f   %14.8f %14.8f  %14.8f   %14.8f   %14.8f"%(
        s, E , DE2, DE3, E+DE2+DE3, OmegaNorm1, OmegaNorm2, norm_fod, norm_Gammaod))
    else:
      print("%8.5f %14.8f   %14.8f   %14.8f %14.8f  %14.8f %14.8f  %14.8f   %14.8f"%(
        s, E , DE2, DE3, E+DE2+DE3, OmegaNorm, 0, norm_fod, norm_Gammaod))
    # Loop ends here

  E_s = E_i
  f_s = f_i
  Gamma_s = Gamma_i

  if store_operators:
    # Check final value using all stored Magnus operators
    for i in range(len(Omegas2B)):
      E_s, f_s, Gamma_s = similarity_transform(Omegas1B[i], Omegas2B[i], E_s, f_s, Gamma_s, user_data)
  else:
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

  return

  output = pd.DataFrame({
    'g':           glist,
    'Ref Energy':  final_E,
    'Total Steps': final_step,
    'Total Time':  total_time,
    'RAM Usage':   total_RAM
  })

  out_type = ''
  if use_second_order:
    out_type += '2'
  if store_operators:
    out_type += '_Stored'
  if store_operators == False:
    out_type += '_BCH'
  
  output.to_csv(f'imsrg-white_d{delta}_b{b}_N4_perturbative{out_type}.csv')


#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()

