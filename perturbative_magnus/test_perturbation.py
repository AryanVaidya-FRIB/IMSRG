from commutators import commutator_2b, similarity_transform, BCH
from generators import eta_white_mp
from basis import *
from classification import *
from hamiltonian import *

def main():
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
	"dE"       :  1e10,
	"omegaNorm":  1e10,
	"ref_energy": 0,

	"order":      20,                 # variables for magnus series expansions
	"bernoulli":   0,                 # and lists of Bernoulli numbers
	"hamiltonian": 0,                 # stored starting Hamiltonian
	}

	H1B, H2B = pairing_hamiltonian(1, 0.5, 0, user_data)
	E, f, Gamma = normal_order(H1B, H2B, user_data)
	fd, fod, Gammad, Gammaod = separate_diag(f, Gamma, user_data)
	eta1B, eta2B = eta_white_mp(f, Gamma, user_data)
	delta1, delta2 = Delta(f, Gamma, user_data)

	comm0, comm1, comm2 = commutator_2b(eta1B, eta2B, fod, Gammad, user_data)
	comm1d, comm1od, comm2d, comm2od = separate_diag(comm1, comm2, user_data)

	born1 = np.zeros_like(f)
	born2 = np.zeros_like(Gamma)

	for a in particles:
		for i in holes:
			val = comm1od[a,i]/delta1[a,i]
			born1[a, i] =  val
			born1[i, a] = -val 

	for a in particles:
		for b in particles:
			for i in holes:
				for j in holes:

					val = comm2od[idx2B[(a,b)], idx2B[(i,j)]] / delta2[idx2B[(a,b)], idx2B[(i,j)]]

					born2[idx2B[(a,b)],idx2B[(i,j)]] = val
					born2[idx2B[(i,j)],idx2B[(a,b)]] = -val

	Cterm0, Cterm1, Cterm2 = commutator_2b(born1, born2, eta1B, eta2B, user_data)
	C1d, C1od, C2d, C2od = separate_diag(Cterm1, Cterm2, user_data)
	print("Second C contribution: ", np.linalg.norm(C1od, ord='fro')+np.linalg.norm(C2od, ord='fro'))

    

if __name__ == "__main__": 
    main()