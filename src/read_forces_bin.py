"""
Subroutines to read '*.train.forces' fortran binary files and the '*.train.ascii' ascii files
"""

import numpy as np
from data_classes import *
import torch
import array


def tf_read_header(tf, sys_species):
	"""
	 Read information about the Training set (see data_classes.TrainSetParameter)
	"""

	N_species = int(tf.readline())
	N_struc   = int(tf.readline())

	species_names = tf.readline().split()
	aux_E_atomic = [ float(x) for x in tf.readline().split() ]
	species_index = []
	E_atomic = []
	for i in range(N_species):
		index = sys_species.index( species_names[i] )
		species_index.append( index )
		E_atomic.append( aux_E_atomic[index] )

	normalized = False
	if tf.readline().strip() == "T":
		normalized = True
	else:
		pass

	E_scaling = float(tf.readline())
	E_shift   = float(tf.readline())

	return N_species, N_struc, species_index, E_atomic, normalized, E_scaling, E_shift



def tf_read_struc_info(tf, species_index, E_atomic):
	"""
	 Read next structures (see data_classes.Structure)
	"""

	coords = []
	forces = []
	descriptors = []
	species = []

	length     = int(tf.readline())
	name       = tf.readline().strip()
	aux        = tf.readline().split()
	N_at, N_sp = int(aux[0]), int(aux[1])
	E          = float(tf.readline())
	
	E_atomic_structure = 0.0
	for iatom in range(N_at):
		sp = int(tf.readline())-1
		species.append(species_index[sp])
		E_atomic_structure += E_atomic[species_index[sp]]

		coords.append( [float(x) for x in tf.readline().split()] )
		forces.append( [float(x) for x in tf.readline().split()] )
		tf.readline()
		descriptors.append( [float(x) for x in tf.readline().split()] )

	return name, E, E_atomic_structure, species, coords, forces, descriptors


def tf_read_footer(tf, N_species, species_index):
	"""
	 Read Information for the Fingerprint Setup (see data_classes.FPSetupParameter)
	"""

	setup_params = FPSetupParameter(N_species)
	input_size = [ 0 for iesp in range(N_species)]

	natomstot           = int(tf.readline())

	# Without normalizing, and without removing E > max_energy
	E_avg, E_min, E_max = [ float(x) for x in tf.readline().split() ]
	has_setups          = tf.readline().strip()

	for iesp in range(N_species):
		sp           = int(tf.readline())-1
		specie_index = species_index[sp]

		description  = tf.readline().strip()
		atomtype     = tf.readline().strip()
		nenv         = int( tf.readline() )

		envtypes  = []
		for i in range(nenv):
			envtypes.append( tf.readline().strip() )

		rcmin = float( tf.readline() )
		rcmax = float( tf.readline() )
		sftype = tf.readline().strip()
		nsf = int( tf.readline() )
		nsfparam = int( tf.readline() )

		sf      = np.array( [ int(x) for x in tf.readline().split() ] )
		sfparam = np.array( [ float(x) for x in tf.readline().split() ] )
		sfenv   = np.array( [ int(x) for x in tf.readline().split() ] )

		neval = int( tf.readline() )

		sfval_min = np.array( [ float(x) for x in tf.readline().split() ] )
		sfval_max = np.array( [ float(x) for x in tf.readline().split() ] )
		sfval_avg = np.array( [ float(x) for x in tf.readline().split() ] )
		sfval_cov = np.array( [ float(x) for x in tf.readline().split() ] )

		if len(description) > 1024: description = description[:1024]

		setup_params.add_specie(specie_index, description, atomtype, nenv, envtypes, rcmin, rcmax,
                            sftype, nsf, nsfparam, sf, sfparam, sfenv, neval,
                            sfval_min, sfval_max, sfval_avg, sfval_cov)

		input_size[specie_index]  = nsf

	return natomstot, E_avg, E_min, E_max, setup_params, input_size



def tff_read_integer(tff, N):
	"""
	Read binary N integer
	"""
	pad = array.array('i')
	result = array.array('i')
	pad.fromfile(tff, 1)       # read the length of the record
	result.fromfile(tff, N)    # read the integer data
	pad.fromfile(tff, 1)       # read the length of the record

	result = result.tolist()

	if N == 1:
		return result[0]
	else:
		return result


def tff_read_real8(tff, N):
	"""
	Read binary N real dp 
	"""
	pad = array.array('i')
	result = array.array('d')
	pad.fromfile(tff, 1)       # read the length of the record
	result.fromfile(tff, N)    # read the integer data
	pad.fromfile(tff, 1)       # read the length of the record

	result = result.tolist()

	if N == 1:
		return result[0]
	else:
		return result


def tff_read_character(tff, N):
	"""
	Read binary character
	"""
	pad = array.array('i')
	result = array.array('B')
	pad.fromfile(tff, 1)       # read the length of the record
	result.fromfile(tff, N)    # read the integer data
	pad.fromfile(tff, 1)       # read the length of the record

	result = result.tobytes().decode(encoding='utf_8')

	return result


def tff_read_header(tff):
	"""
	Read header with number of structures
	"""
	N_struc = tff_read_integer(tff, 1)
	return N_struc


def tff_read_struc_info_grads(tff, species_index, max_nnb):
	"""
	Read information about derivatives of the descriptors of the next structure if it is included in the force training
	"""
	list_nblist = []
	list_sfderiv_i = []
	list_sfderiv_j = []

	lenght           = tff_read_integer(tff, 1)
	name             = tff_read_character(tff, lenght)
	N_at, N_sp       = tff_read_integer(tff,2)
	train_forces_struc  = tff_read_integer(tff,1)

	if train_forces_struc == 1:
		for iatom in range(N_at):
			specie    = tff_read_integer(tff, 1)-1
			nsf, nnb  = tff_read_integer(tff,2)
			nblist    = tff_read_integer(tff,nnb)
			sfderiv_i = tff_read_real8(tff, 3*nsf)
			sfderiv_j = tff_read_real8(tff, 3*nsf*nnb)

			sfderiv_i = np.array(sfderiv_i).reshape(nsf, 3, order="F")
			sfderiv_j = np.array(sfderiv_j).reshape(nnb,nsf,3, order="F")

			if nnb == 1:
				list_nblist.append([nblist])
			else:
				list_nblist.append(nblist)
			list_sfderiv_i.append(torch.tensor(sfderiv_i))
			list_sfderiv_j.append(torch.tensor(sfderiv_j))

			index   = species_index[specie]
			max_nnb[index] = max(max_nnb[index], nnb)

		return train_forces_struc, max_nnb, list_nblist, list_sfderiv_i, list_sfderiv_j

	else:
		return train_forces_struc, max_nnb, None, None, None
