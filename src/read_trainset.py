import numpy as np
from data_classes import *
from read_forces_bin import *


def read_train_forces_together(tin):
	"""
	Read training set files when force training is requested
	Returns
		list_struct_forces :: List of data_classes.Structure objects included in force training
		list_struct_energy :: List of data_classes.Structure objects with only energy information
		list_removed       :: List of names of the structures above the maximum energy cutoff
		max_nnb            :: Maximum number of neighbors in the data set
		tin                :: Updated input parameters
	"""

	trainfile   = tin.train_file
	forcesfile  = tin.train_forces_file
	sys_species = tin.sys_species
	max_energy  = tin.max_energy
	max_forces  = tin.max_forces

	list_removed = []
	E_max_min_avg = [-10000000.0, 10000000.0, 0.0]
	with open (trainfile, "r") as tf, open(forcesfile, "rb") as tff:

		# Header
		N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift = tf_read_header(tf, sys_species)
		_ = tff_read_header(tff)

		# Footer (Fingerprint Setup information)
		natomstot, E_avg, E_min, E_max, setup_params, input_size = tf_read_footer(tf, N_species, species_index)

		trainset_params = TrainSetParameter(trainfile, normalized, E_scaling, E_shift,
						  N_species, sys_species, E_atomic, natomstot, N_struc, E_min, E_max, E_avg)

		# Structures in dataset
		list_struct_forces = []
		list_struct_energy = []
		max_nnb = np.array([0 for i in range(len(species_index))])
		for istruc in range(N_struc):
			name, E, E_atomic_structure, species, coords, forces, descriptors = tf_read_struc_info(tf, species_index, E_atomic)
			train_forces_struc, max_nnb, list_nblist, list_sfderiv_i, list_sfderiv_j = tff_read_struc_info_grads(tff, species_index, max_nnb)

			E_per_atom = E/len(coords)
			if max_energy and E_per_atom > max_energy:
				list_removed.append(name)
			else:
				E_max_min_avg[0] = max(E_max_min_avg[0], E_per_atom)
				E_max_min_avg[1] = min(E_max_min_avg[1], E_per_atom)
				E_max_min_avg[2] += E_per_atom

				if train_forces_struc:

					# Check if F < Fmax
					F_max_struc = np.max(np.abs(np.array(forces)))
					if not max_forces:
						list_struct_forces.append( Structure(name, species, descriptors, E, E_atomic_structure, sys_species, coords, forces, input_size, 
		         											 train_forces_struc, list_nblist, list_sfderiv_i, list_sfderiv_j) )
					elif max_forces and F_max_struc < max_forces:
						list_struct_forces.append( Structure(name, species, descriptors, E, E_atomic_structure, sys_species, coords, forces, input_size, 
		         											 train_forces_struc, list_nblist, list_sfderiv_i, list_sfderiv_j) )
					else:
						train_forces_struc = False
						list_struct_energy.append( Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
					                                    	 coords, forces, input_size, train_forces_struc) )

				else:
					list_struct_energy.append( Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
					                                     coords, forces, input_size, train_forces_struc) )

		# Recompute E_scaling and E_shift if some structures have been excluded
		trainset_params.E_max = E_max_min_avg[0]
		trainset_params.E_min = E_max_min_avg[1]
		trainset_params.E_avg = E_max_min_avg[2]/(len(list_struct_forces) + len(list_struct_energy))

		trainset_params.get_E_normalization()

		max_nnb             = np.max(max_nnb)
		tin.trainset_params = trainset_params
		tin.setup_params    = setup_params
		tin.networks_param["input_size"] = input_size

		return list_struct_forces, list_struct_energy, list_removed, max_nnb, tin



def read_train(tin):
	"""
	Read training set files with only energy training
	Returns
		list_struct_energy :: List of data_classes.Structure objects with only energy information
		list_removed       :: List of names of the structures above the maximum energy cutoff
		max_nnb            :: Maximum number of neighbors in the data set
		tin                :: Updated input parameters
	"""

	trainfile   = tin.train_file
	sys_species = tin.sys_species
	max_energy  = tin.max_energy

	list_removed = []
	E_max_min_avg = [-10000000.0, 10000000.0, 0.0]
	with open (trainfile, "r") as tf:

		# Header
		N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift = tf_read_header(tf, sys_species)


		# Footer (Fingerprint Setup information)
		natomstot, E_avg, E_min, E_max, setup_params, input_size = tf_read_footer(tf, N_species, species_index)

		trainset_params = TrainSetParameter(trainfile, normalized, E_scaling, E_shift,
						  N_species, sys_species, E_atomic, natomstot, N_struc, E_min, E_max, E_avg)


		# Structures in dataset
		list_struct_forces = []
		list_struct_energy = []
		max_nnb = np.array([0 for i in range(len(species_index))])
		for istruc in range(N_struc):
			name, E, E_atomic_structure, species, coords, forces, descriptors = tf_read_struc_info(tf, species_index, E_atomic)

			E_per_atom = E/len(coords)
			if max_energy and E_per_atom > max_energy:
				list_removed.append(name)
			else:
				E_max_min_avg[0] = max(E_max_min_avg[0], E_per_atom)
				E_max_min_avg[1] = min(E_max_min_avg[1], E_per_atom)
				E_max_min_avg[2] += E_per_atom
				
				list_struct_energy.append( Structure(name, species, descriptors, E, E_atomic_structure, sys_species,
					                                 coords, forces, input_size) )

		# Recompute E_scaling and E_shift if some structures have been excluded
		trainset_params.E_max = E_max_min_avg[0]
		trainset_params.E_min = E_max_min_avg[1]
		trainset_params.E_avg = E_max_min_avg[2]/(len(list_struct_forces) + len(list_struct_energy))
		trainset_params.get_E_normalization()

		max_nnb             = np.max(max_nnb)
		tin.trainset_params = trainset_params
		tin.setup_params    = setup_params
		tin.networks_param["input_size"] = input_size

		return list_struct_energy, list_removed, max_nnb, tin