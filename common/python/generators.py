#!/usr/bin/env python

#------------------------------------------------------------------------------
# generators.py
#
# author:   A. Vaidya and H. Hergert
# version:  1.0
# date:     Nov 26, 2024
# 
# tested with Python v3.10
# 
# Constructs Brillouin's, White's, Wegner's and Imaginary Time Generators
# for single-referenced IMSRG calculations. Truncates all calculations to
# 2-body order. 
#
#------------------------------------------------------------------------------

import numpy as np
from math import pi
from basis import ph_transform_2B, inverse_ph_transform_2B

def eta_brillouin(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      # (1-n_a)n_i - n_a(1-n_i) = n_i - n_a
      eta1B[a, i] =  f[a,i]
      eta1B[i, a] = -f[a,i]

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_imtime(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      dE = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      val = np.sign(dE)*f[a,i]
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          dE = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          val = np.sign(dE)*Gamma[idx2B[(a,b)], idx2B[(i,j)]]

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      # catch small or zero denominator - matrix element inspired by arctan version
      if abs(denom)<1.0e-10:
        val = 0.25 * pi * np.sign(f[a,i]) * np.sign(denom)
      else:
        val = f[a,i]/denom

      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]]
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )
          # catch small or zero denominator - matrix element inspired by arctan version
          if abs(denom)<1.0e-10:
            val = 0.25 * pi * np.sign(Gamma[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(denom)
          else:
            val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_white_mp(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i]
      val = f[a,i]/denom
      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]  
          )

          val = Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B

def eta_white_atan(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]
  idx2B     = user_data["idx2B"]

  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  for a in particles:
    for i in holes:
      denom = f[a,a] - f[i,i] + Gamma[idx2B[(a,i)], idx2B[(a,i)]]
      # catch zero or small denominator
      if abs(denom)<1.0e-10:
        val = 0.25 * pi * np.sign(f[a,i]) * np.sign(denom)
      else:
        val = 0.5 * np.arctan(2 * f[a,i]/denom)

      eta1B[a, i] =  val
      eta1B[i, a] = -val 

  # two-body part of the generator
  eta2B = np.zeros_like(Gamma)

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j] 
            + Gamma[idx2B[(a,b)],idx2B[(a,b)]] 
            + Gamma[idx2B[(i,j)],idx2B[(i,j)]] 
            - Gamma[idx2B[(a,i)],idx2B[(a,i)]] 
            - Gamma[idx2B[(a,j)],idx2B[(a,j)]] 
            - Gamma[idx2B[(b,i)],idx2B[(b,i)]] 
            - Gamma[idx2B[(b,j)],idx2B[(b,j)]] 
          )

          # catch zero or small denominator
          if abs(denom)<1.0e-10:
            val = 0.25 * pi * np.sign(Gamma[idx2B[(a,b)], idx2B[(i,j)]]) * np.sign(denom)
          else:
            val = 0.5 * np.arctan(2 * Gamma[idx2B[(a,b)], idx2B[(i,j)]] / denom)

          eta2B[idx2B[(a,b)],idx2B[(i,j)]] = val
          eta2B[idx2B[(i,j)],idx2B[(a,b)]] = -val

  return eta1B, eta2B


def eta_wegner(f, Gamma, user_data):

  dim1B     = user_data["dim1B"]
  holes     = user_data["holes"]
  particles = user_data["particles"]
  bas2B     = user_data["bas2B"]
  basph2B   = user_data["basph2B"]
  idx2B     = user_data["idx2B"]
  idxph2B   = user_data["idxph2B"]
  occB_2B   = user_data["occB_2B"]
  occC_2B   = user_data["occC_2B"]
  occphA_2B = user_data["occphA_2B"]


  # split Hamiltonian in diagonal and off-diagonal parts
  fd      = np.zeros_like(f)
  fod     = np.zeros_like(f)
  Gammad  = np.zeros_like(Gamma)
  Gammaod = np.zeros_like(Gamma)

  for a in particles:
    for i in holes:
      fod[a, i] = f[a,i]
      fod[i, a] = f[i,a]
  fd = f - fod

  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          Gammaod[idx2B[(a,b)], idx2B[(i,j)]] = Gamma[idx2B[(a,b)], idx2B[(i,j)]]
          Gammaod[idx2B[(i,j)], idx2B[(a,b)]] = Gamma[idx2B[(i,j)], idx2B[(a,b)]]
  Gammad = Gamma - Gammaod


  #############################        
  # one-body part of the generator
  eta1B  = np.zeros_like(f)

  # 1B - 1B
  eta1B += commutator(fd, fod)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        for a in particles:
          eta1B[p,q] += (
            fd[i,a]  * Gammaod[idx2B[(a, p)], idx2B[(i, q)]] 
            - fd[a,i]  * Gammaod[idx2B[(i, p)], idx2B[(a, q)]] 
            - fod[i,a] * Gammad[idx2B[(a, p)], idx2B[(i, q)]] 
            + fod[a,i] * Gammad[idx2B[(i, p)], idx2B[(a, q)]]
          )

  # 2B - 2B
  # n_a n_b nn_c + nn_a nn_b n_c = n_a n_b + (1 - n_a - n_b) * n_c
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for i in holes:
        eta1B[p,q] += 0.5*(
          GammaGamma[idx2B[(i,p)], idx2B[(i,q)]] 
          - transpose(GammaGamma)[idx2B[(i,p)], idx2B[(i,q)]]
        )

  GammaGamma = dot(Gammad, dot(occC_2B, Gammaod))
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        eta1B[p,q] += 0.5*(
          GammaGamma[idx2B[(r,p)], idx2B[(r,q)]] 
          - transpose(GammaGamma)[idx2B[(r,p)], idx2B[(r,q)]] 
        )


  #############################        
  # two-body flow equation  
  eta2B = np.zeros_like(Gamma)

  # 1B - 2B
  for p in range(dim1B):
    for q in range(dim1B):
      for r in range(dim1B):
        for s in range(dim1B):
          for t in range(dim1B):
            eta2B[idx2B[(p,q)],idx2B[(r,s)]] += (
              fd[p,t] * Gammaod[idx2B[(t,q)],idx2B[(r,s)]] 
              + fd[q,t] * Gammaod[idx2B[(p,t)],idx2B[(r,s)]] 
              - fd[t,r] * Gammaod[idx2B[(p,q)],idx2B[(t,s)]] 
              - fd[t,s] * Gammaod[idx2B[(p,q)],idx2B[(r,t)]]
              - fod[p,t] * Gammad[idx2B[(t,q)],idx2B[(r,s)]] 
              - fod[q,t] * Gammad[idx2B[(p,t)],idx2B[(r,s)]] 
              + fod[t,r] * Gammad[idx2B[(p,q)],idx2B[(t,s)]] 
              + fod[t,s] * Gammad[idx2B[(p,q)],idx2B[(r,t)]]
            )

  
  # 2B - 2B - particle and hole ladders
  # Gammad.occB.Gammaod
  GammaGamma = dot(Gammad, dot(occB_2B, Gammaod))

  eta2B += 0.5 * (GammaGamma - transpose(GammaGamma))

  # 2B - 2B - particle-hole chain
  
  # transform matrices to particle-hole representation and calculate 
  # Gammad_ph.occA_ph.Gammaod_ph
  Gammad_ph = ph_transform_2B(Gammad, bas2B, idx2B, basph2B, idxph2B)
  Gammaod_ph = ph_transform_2B(Gammaod, bas2B, idx2B, basph2B, idxph2B)

  GammaGamma_ph = dot(Gammad_ph, dot(occphA_2B, Gammaod_ph))

  # transform back to standard representation
  GammaGamma    = inverse_ph_transform_2B(GammaGamma_ph, bas2B, idx2B, basph2B, idxph2B)

  # commutator / antisymmetrization
  work = np.zeros_like(GammaGamma)
  for i1, (i,j) in enumerate(bas2B):
    for i2, (k,l) in enumerate(bas2B):
      work[i1, i2] -= (
        GammaGamma[i1, i2] 
        - GammaGamma[idx2B[(j,i)], i2] 
        - GammaGamma[i1, idx2B[(l,k)]] 
        + GammaGamma[idx2B[(j,i)], idx2B[(l,k)]]
      )
  GammaGamma = work

  eta2B += GammaGamma


  return eta1B, eta2B