#!/usr/bin/env python

#------------------------------------------------------------------------------
# commutators.py
#
# author:   A. Vaidya
# version:  1.0
# date:     Nov 26, 2024
# 
# tested with Python v3.10
# 
# Contains general IMSRG(2) commutator. 
# Needs user_data to be defined when called, including bases, occupation bases
# and reference state.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, transpose
from math import factorial
from basis import ph_transform_2B, inverse_ph_transform_2B

#-----------------------------------------------------------------------------------
# commutator of matrices
#-----------------------------------------------------------------------------------
def commutator(a,b):
  return dot(a,b) - dot(b,a)

# -----------------------------------------------------------------------------
# Commutator of two operators truncated to two-body order, using C = [A,B]
# Included elements are labeled [Operator]_[n-body] 
# Ex. A_1 is the 1-body piece of A
# -----------------------------------------------------------------------------

def commutator_2b(A_1, A_2, B_1, B_2, user_data):

  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  basph2B   = user_data["basph2B"]
  idxph2B   = user_data["idxph2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]

  #############################        
  # zero-body flow equation
  C_0 = 0.0

  for i in holes:
    for a in particles:
      C_0 += A_1[i,a] * B_1[a,i] - A_1[a,i] * B_1[i,a]

  for i in holes:
    for j in holes:
      for a in particles:
        for b in particles:
          C_0 += 0.5 * A_2[idx2B[(i,j)], idx2B[(a,b)]] * B_2[idx2B[(a,b)], idx2B[(i,j)]]


  #############################        
  # one-body flow equation  
  C_1  = np.zeros_like(B_1)

  # 1B - 1B
  C_1 += commutator(A_1, B_1)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for a in particles:
          C_1[p,q] += (
            A_1[i,a] * B_2[idx2B[(a, p)], idx2B[(i, q)]] 
            - A_1[a,i] * B_2[idx2B[(i, p)], idx2B[(a, q)]] 
            - B_1[i,a] * A_2[idx2B[(a, p)], idx2B[(i, q)]] 
            + B_1[a,i] * A_2[idx2B[(i, p)], idx2B[(a, q)]]
          )

  # 2B - 2B
  # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
  crossterm_2b = dot(A_2, dot(occB_2B, B_2))
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        C_1[p,q] += 0.5*(
          crossterm_2b[idx2B[(i,p)], idx2B[(i,q)]] 
          + transpose(crossterm_2b)[idx2B[(i,p)], idx2B[(i,q)]]
        )

  crossterm_2b = dot(A_2, dot(occC_2B, B_2))
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        C_1[p,q] += 0.5*(
          crossterm_2b[idx2B[(r,p)], idx2B[(r,q)]] 
          + transpose(crossterm_2b)[idx2B[(r,p)], idx2B[(r,q)]] 
        )


  #############################        
  # two-body flow equation  
  C_2 = np.zeros_like(B_2)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for t in range(dim1B):
            C_2[idx2B[(p,q)],idx2B[(r,s)]] += (
              A_1[p,t] * B_2[idx2B[(t,q)],idx2B[(r,s)]] 
              + A_1[q,t] * B_2[idx2B[(p,t)],idx2B[(r,s)]] 
              - A_1[t,r] * B_2[idx2B[(p,q)],idx2B[(t,s)]] 
              - A_1[t,s] * B_2[idx2B[(p,q)],idx2B[(r,t)]]
              - B_1[p,t] * A_2[idx2B[(t,q)],idx2B[(r,s)]] 
              - B_1[q,t] * A_2[idx2B[(p,t)],idx2B[(r,s)]] 
              + B_1[t,r] * A_2[idx2B[(p,q)],idx2B[(t,s)]] 
              + B_1[t,s] * A_2[idx2B[(p,q)],idx2B[(r,t)]]
            )

  
  # 2B - 2B - particle and hole ladders
  # A_2.occB.Gamma
  crossterm_2b = dot(A_2, dot(occB_2B, B_2))

  C_2 += 0.5 * (crossterm_2b + transpose(crossterm_2b))

  # 2B - 2B - particle-hole chain
  
  # transform matrices to particle-hole representation and calculate 
  # eta2B_ph.occA_ph.Gamma_ph
  eta2B_ph = ph_transform_2B(A_2, bas2B, idx2B, basph2B, idxph2B)
  Gamma_ph = ph_transform_2B(B_2, bas2B, idx2B, basph2B, idxph2B)

  etaGamma_ph = dot(eta2B_ph, dot(occphA_2B, Gamma_ph))

  # transform back to standard representation
  crossterm_2b    = inverse_ph_transform_2B(etaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

  # commutator / antisymmetrization
  work = np.zeros_like(crossterm_2b)
  for i1, (i,j) in enumerate(bas2B):
    for i2, (k,l) in enumerate(bas2B):
      work[i1, i2] -= (
        crossterm_2b[i1, i2] 
        - crossterm_2b[idx2B[(j,i)], i2] 
        - crossterm_2b[i1, idx2B[(l,k)]] 
        + crossterm_2b[idx2B[(j,i)], idx2B[(l,k)]]
      )
  crossterm_2b = work

  C_2 += crossterm_2b


  return C_0, C_1, C_2

# -----------------------------------------------------------------------------
# BCH for Matrix Transformations of form 
# exp(Operator)Hamiltonian exp(-Operator)
# -----------------------------------------------------------------------------
def matrix_similarity_transform(Operator, Hamiltonian, order):
  dim = Hamiltonian.shape[0]
  Hout = np.zeros((dim,dim))
  for k in range(order):
    temp_H = Hamiltonian
    for _ in range(k):
      temp_H = commutator(Operator, Hamiltonian)
    
    if abs(temp_H[0][0]/factorial(k))< 1e-8 and k>6:
      break
    Hout += temp_H

  return Hout

# -----------------------------------------------------------------------------
# BCH for IMSRG Transformations truncated at 2-body contributions
# with form exp(Operator)Hamiltonian exp(-Operator)
# -----------------------------------------------------------------------------
def similarity_transform(Operator1B, Operator2B, E, f, Gamma, user_data):
    E_s = 0
    f_s = np.zeros(f.shape)
    Gamma_s = np.zeros(Gamma.shape)

    BCH_Order = user_data["order"]
    for k in range(BCH_Order):
      temp_E = E
      temp_f = f
      temp_Gamma = Gamma
      for _ in range(k):
        temp_E, temp_f, temp_Gamma = flow_imsrg2(Operator1B, Operator2B, temp_f, temp_Gamma, user_data)
        
      if abs(temp_E/factorial(k)) < 1E-8 and k > 6: 
        break
      E_s += temp_E/factorial(k)
      f_s += temp_f/factorial(k)
      Gamma_s += temp_Gamma/factorial(k)

    return E_s, f_s, Gamma_s