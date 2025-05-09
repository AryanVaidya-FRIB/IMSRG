#!/usr/bin/env python

#------------------------------------------------------------------------------
# matrix_perturbation.py
#
# author:   A. Vaidya
# version:  1.1
# date:     Dec 18, 2024
# 
# tested with Python v3.10
# 
# Solves the pairing model for four particles by perturbatively expanding the 
# Magnus operator first order. Found using H(0)/Delta(0), through element-wise 
# division. Fixed.
#
#------------------------------------------------------------------------------

import numpy as np
from numpy import array, dot, diag, reshape, pi
from scipy.linalg import eigvalsh, expm
from commutators import commutator, matrix_similarity_transform

from sys import argv

#------------------------------------------------------------------------------
# Construct Hamiltonian
#------------------------------------------------------------------------------

# Hamiltonian for the pairing model
def Hamiltonian(delta,g):

    H = array(
        [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
         [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ],
         [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ],
         [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ],
         [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ],
         [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]]
      )

    return H

#------------------------------------------------------------------------------
# Loop functions for Delta and Omega
#------------------------------------------------------------------------------

def compute_Delta(H):
    # Dimension of the Hamiltonian
    dim = H.shape[0]
    
    # Initialize the Delta matrix
    Delta = np.zeros((dim, dim))
    
    # Calculate Delta by combining H1B and H2B pieces
    for i in range(dim):
        for j in range(dim):
            if i != j:
                Delta[i,j] = H[i,i]-H[j,j]

    Delta[Delta == 0] = 1e-12     
    
    return Delta

def compute_Omega(H, Delta):
    # Compute first order Omega = Hod/Delta. If Delta is too small, replace value with given value from arctan version
    dim = H.shape[0]
    Hod = H-diag(diag(H))  
    Omega = Hod/Delta
                
    return Omega

#------------------------------------------------------------------------------
# Compute test conditions for killing process early
#------------------------------------------------------------------------------

def compute_odNorm(H):
    Hod = H-diag(H)
    return np.linalg.norm(Hod)

def compute_E(H):
    return H[0,0]# Assumes the first slot of the Hamiltonian contains ground state energy


#------------------------------------------------------------------------------
# Main Program
#------------------------------------------------------------------------------

def main():
    # grab delta and g from the command line
    delta      = float(argv[1])
    g          = float(argv[2])

    # Get starting parameters
    H0 = Hamiltonian(delta, g)

    # calculate exact eigenvalues
    eigenvalues = eigvalsh(H0)
    print(eigenvalues)

    # initialize number of perturbative steps
    max_step = 10

    # initalize solution storage vectors
    Hs = [H0]
    steps = [0]

    # Print for table initialization
    print("%-8s   %-14s   %-14s   %-14s"%(
    "step", "E" ,"||Omega||", "||Hod||"))
    print("-" * 148)
    print("%8.5i %14.8f   %14.8f   %14.8f"%(
      0, compute_E(H0) , 0, compute_odNorm(H0)))

    for step in range(1,max_step):
        Delta = compute_Delta(H0)
        # print(Delta)
        Omega = compute_Omega(H0, Delta)
        Omega_norm = np.linalg.norm(Omega)

        H0 = expm(Omega)*H0*expm(-1*Omega)
        # print(H0)

        Hod_norm = compute_odNorm(H0)
        E = compute_E(H0)

        # Make sure od norm and ground state energy have decreased
        lastH = Hs[-1]
        last_od = compute_odNorm(lastH)
        lastE = compute_E(lastH)

        if abs(Hod_norm - last_od) < 1e-6 or abs(E-lastE) < 1e-6:
            break
        
        Hs.append(H0)
        steps.append(step)

        print("%8.5i %14.8f   %14.8f   %14.8f"%(
      step, E , Omega_norm, Hod_norm))

#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()
