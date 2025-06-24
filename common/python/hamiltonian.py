import numpy as np
from numpy import array, dot, diag, reshape, transpose
from math import factorial
from basis import ph_transform_2B, inverse_ph_transform_2B

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
      Aod_1b[a,i] = A_1b[a,i]
      Aod_1b[i,a] = A_1b[i,a]

  Ad_1b = A_1b-Aod_1b

  # Separate 2-body operator
  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          # The off-diagonal - pphh and hhpp states
          Aod_2b[idx2B[(a,b)], idx2B[(i,j)]] = A_2b[idx2B[(a,b)], idx2B[(i,j)]]
          Aod_2b[idx2B[(i,j)], idx2B[(a,b)]] = A_2b[idx2B[(i,j)], idx2B[(a,b)]]

  # Diagonal 2-body operator
  Ad_2b = A_2b-Aod_2b # (Includes terms not in the strict definition, so like aibj, abcd, ijkl terms)

  return Ad_1b, Aod_1b, Ad_2b, Aod_2b

def Delta(f, Gamma, user_data):
  dim1B     = user_data["dim1B"]
  bas1B     = user_data["bas1B"]
  bas2B     = user_data["bas2B"]
  idx2B     = user_data["idx2B"]
  particles = user_data["particles"]
  holes     = user_data["holes"]

  # Calculate the denominator operator
  denom_1b = np.zeros_like(f)
  for a in particles:
    for i in holes:
      denom_1b[a,i] = f[a,a] - f[i,i]

  denom_2b = np.zeros_like(Gamma)
  for a in particles:
    for b in particles:
      for i in holes:
        for j in holes:
          denom_2b[idx2B[(a,b)], idx2B[(i,j)]] = ( 
            f[a,a] + f[b,b] - f[i,i] - f[j,j]
          )

  return denom_1b, denom_2b