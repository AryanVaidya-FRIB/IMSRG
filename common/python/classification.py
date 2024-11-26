#!/usr/bin/env python

#------------------------------------------------------------------------------
# basis.py
#
# author:   A. Vaidya and H. Hergert
# version:  1.0
# date:     Nov 26, 2024
# 
# tested with Python v3.10
# 
# Constructs different heuristics to monitor IMSRG flow progression.
#
#------------------------------------------------------------------------------

import numpy as np

#-----------------------------------------------------------------------------------
# norms of off-diagonal Hamiltonian pieces
#-----------------------------------------------------------------------------------
def calc_fod_norm(f, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  
  norm = 0.0
  for a in particles:
    for i in holes:
      norm += f[a,i]**2 + f[i,a]**2

  return np.sqrt(norm)

def calc_Gammaod_norm(Gamma, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  norm = 0.0
  for a in particles:    
    for b in particles:
      for i in holes:
        for j in holes:
          norm += Gamma[idx2B[(a,b)],idx2B[(i,j)]]**2 + Gamma[idx2B[(i,j)],idx2B[(a,b)]]**2

  return np.sqrt(norm)

#-----------------------------------------------------------------------------------
# Perturbation theory
#-----------------------------------------------------------------------------------
def calc_mbpt2(f, Gamma, user_data):
  DE2 = 0.0

  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  for i in holes:
    for j in holes:
      for a in particles:
        for b in particles:
          denom = f[i,i] + f[j,j] - f[a,a] - f[b,b]
          me    = Gamma[idx2B[(a,b)],idx2B[(i,j)]]
          DE2  += 0.25*me*me/denom

  return DE2

def calc_mbpt3(f, Gamma, user_data):
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # DE3 = 0.0

  DE3pp = 0.0
  DE3hh = 0.0
  DE3ph = 0.0

  for a in particles:
    for b in particles:
      for c in particles:
        for d in particles:
          for i in holes:
            for j in holes:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[i,i] + f[j,j] - f[c,c] - f[d,d])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(a,b)],idx2B[(c,d)]]*Gamma[idx2B[(c,d)],idx2B[(i,j)]]
              DE3pp += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for l in holes:
          for a in particles:
            for b in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[l,l] - f[a,a] - f[b,b])
              me    = Gamma[idx2B[(a,b)],idx2B[(k,l)]]*Gamma[idx2B[(k,l)],idx2B[(i,j)]]*Gamma[idx2B[(i,j)],idx2B[(a,b)]]
              DE3hh += 0.125*me/denom

  for i in holes:
    for j in holes:
      for k in holes:
        for a in particles:
          for b in particles:
            for c in particles:
              denom = (f[i,i] + f[j,j] - f[a,a] - f[b,b])*(f[k,k] + f[j,j] - f[a,a] - f[c,c])
              me    = Gamma[idx2B[(i,j)],idx2B[(a,b)]]*Gamma[idx2B[(k,b)],idx2B[(i,c)]]*Gamma[idx2B[(a,c)],idx2B[(k,j)]]
              DE3ph -= me/denom
  return DE3pp+DE3hh+DE3ph