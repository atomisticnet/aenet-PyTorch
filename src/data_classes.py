import torch
import numpy as np


class InputParameters(object):
	"""
	Information from the input file (train.in)
		train_file     :: [TRAININGSET] Training set file, output of 'generate.x'
		test_split     :: [TESTPERCENT] Train/test split
		epoch_size     :: [ITERATIONS]  Number of training iterations
		epoch_write    :: [ITERWRITE]   Write loss function value every N train steps
		batch_size     :: [BATCHSIZE]   Batch size

		numpy_seed     :: [NPSEED] Random seed for Numpy
		pytorch_seed   :: [PHSEED] Random seed for PyTorch
		max_energy     :: [MAXENERGY] Maximum energy allowed in the training set
		save_energies  :: [SAVE_ENERGIES] Save energy of the train/test sets in the end
		save_forces    :: [SAVE_FORCES] Save forces of the train/test sets in the end
		memory_mode    :: [MEMORY_MODE] Mode to handle memory: CPU or GPU
		verbose        :: [VERBOSE] Display long output

		train_forces   :: [FORCES] Logical. Train or not forces
		alpha          :: [ALPHA] Alpha parameter to weigh energy and force errors
		max_forces     :: [MAXFORCES] Maximum force value allowed in the training set

		method         :: [METHOD] Optimization algorithm
		lr             :: [LR] Learning rate
		regularization :: [REGULARIZATION] L2 regularization value

		mode           :: Train/Batches/Transfer
		save_batches   :: 
		load_batches   :: 
	"""
	def __init__(self):

		# Default values:
		self.train_file    = ""
		self.numpy_seed    = 11
		self.pytorch_seed  = 22
		self.epoch_write   = 1
		self.batch_size    = 256
		self.mode          = "train"
		self.max_energy    = False
		self.max_forces    = False
		self.save_energies = False
		self.save_forces   = False
		self.train_forces  = False
		self.memory_mode   = "cpu"
		self.alpha         = 0.0
		self.verbose       = False
		self.save_batches  = False
		self.load_batches  = False

	def initialize(self):
		self.test_split = float(self.test_split)/100
		self.epoch_size = int(self.epoch_size)
		self.method     = self.method_param["method"].lower()
		self.lr         = float(self.method_param["lr"]) 
		self.N_species  = len(self.sys_species)

		self.original_batch_size = self.batch_size


		if self.train_forces:
			self.train_forces_file = self.train_file[:-6]+".forces"
			self.alpha             = float(self.forces_param["alpha"])



class FPSetupParameter(object):
	"""
	Information for the Fingerprint Setup
		description :: character(1024)     Description of the training set
		atomtype    :: character(2)        Element to which the setup belong
		nenv        :: integer             Number of elements in the environment
		envtypes    :: Character(nenv)     Elements in the environment
		rcmin       :: real                Minimum radius in the neighbor list
		rcmax       :: real                Maximum radius in the neighbor list
		sftype      :: Character(100)      Descriptor type (Chevyshev etc)
		nsfparam    :: integer             Number of parameters of that descriptor
		nsf         :: integer             Size of the descriptor vector
		sf          :: float(nsf)          Values of the descriptor
		sfparam     :: float(nsfparam,nsf) Parameters of the descriptor
		sfenv       :: float(2nsf)         
		neval       :: integer             Number of times that the descriptor has been computed
		sfval_min   :: float(nsf)          Minimum of the descriptors in the set
		sfval_max   :: float(nsf)          Maximum of the descriptors in the set
		sfval_avg   :: float(nsf)          Average of the descriptors in the set
		sfval_cov   :: float(nsf)          Covariance of the descriptors in the set
	"""
	def __init__(self, N_species):
		self.N_species = N_species

		self.description = [ 0 for iesp in range(N_species) ]
		self.atomtype    = [ 0 for iesp in range(N_species) ]
		self.nenv        = [ 0 for iesp in range(N_species) ]
		self.envtypes    = [ 0 for iesp in range(N_species) ]
		self.rcmin       = [ 0 for iesp in range(N_species) ]
		self.rcmax       = [ 0 for iesp in range(N_species) ]
		self.sftype      = [ 0 for iesp in range(N_species) ]
		self.nsfparam    = [ 0 for iesp in range(N_species) ]
		self.nsf         = [ 0 for iesp in range(N_species) ]
		self.sf          = [ 0 for iesp in range(N_species) ]
		self.sfparam     = [ 0 for iesp in range(N_species) ]
		self.sfenv       = [ 0 for iesp in range(N_species) ]

		self.neval       = [ 0 for iesp in range(N_species) ]
		self.sfval_min   = [ 0 for iesp in range(N_species) ]
		self.sfval_max   = [ 0 for iesp in range(N_species) ]
		self.sfval_avg   = [ 0 for iesp in range(N_species) ]
		self.sfval_cov   = [ 0 for iesp in range(N_species) ]

	def add_specie(self, iesp, description, atomtype, nenv, envtypes, rcmin, rcmax, 
					sftype, nsf, nsfparam, sf, sfparam, sfenv, neval,
					sfval_min, sfval_max, sfval_avg, sfval_cov):
		self.description[iesp] = description
		self.atomtype[iesp]    = atomtype
		self.nenv[iesp]        = nenv
		self.envtypes[iesp]    = envtypes
		self.rcmin[iesp]       = rcmin
		self.rcmax[iesp]       = rcmax
		self.sftype[iesp]      = sftype
		self.nsfparam[iesp]    = nsfparam
		self.nsf[iesp]         = nsf
		self.sf[iesp]          = sf
		self.sfparam[iesp]     = sfparam
		self.sfenv[iesp]       = sfenv

		self.neval[iesp]       = neval
		self.sfval_min[iesp]   = sfval_min
		self.sfval_max[iesp]   = sfval_max
		self.sfval_avg[iesp]   = sfval_avg
		self.sfval_cov[iesp]   = sfval_cov




class TrainSetParameter(object):
	"""
	Information from the Trainset file (*.train.ascii)
		filename    :: Name of the trainset file
		normalized  :: Whether the training set energies are normalized to (-1,1)
		E_scaling   :: Energy scaling factor
		E_shift     :: Energy shift factor
		N_species   :: Number of species in the system
		sys_species :: List of species in the system
		E_atomic    :: List of atomic energies of the isolated atoms
		N_atom      :: Number of atoms in total in the data set
		N_struc     :: Number of structures in the data set
		E_min       :: Minimum energy of the data set
		E_max       :: Maximum energy of the data set
		E_avg       :: Average energy of the data set
	"""
	def __init__(self, filename, normalized, E_scaling, E_shift, N_species, sys_species,
		         E_atomic, N_atom, N_struc, E_min, E_max, E_avg):
		self.filename = filename
		self.normalized = normalized
		self.E_scaling = E_scaling
		self.E_shift = E_shift
		self.N_species = N_species
		self.sys_species = sys_species
		self.E_atomic = E_atomic
		self.N_atom = N_atom
		self.N_struc = N_struc
		self.E_min = E_min
		self.E_max = E_max
		self.E_avg = E_avg

	def get_E_normalization(self):
		"""
		Compute the scaling and shifting factors to normalize the energies
		"""

		E_scaling = 2/(self.E_max - self.E_min)
		E_shift   = 0.5*(self.E_max + self.E_min)

		self.E_min_norm = (self.E_min - E_shift)*E_scaling
		self.E_max_norm = (self.E_max - E_shift)*E_scaling
		self.E_avg_norm = (self.E_avg - E_shift)*E_scaling

		self.E_scaling  = E_scaling
		self.E_shift    = E_shift
		self.normalized = True

		return E_scaling, E_shift




class Structure(object):
	"""
	Information of a structure from the dataset
		name         :: Name of the "xsf" file
		energy       :: Cohesive energy (normalized or not)
		E_atomic_str :: Sum of the isolated energies of the atoms
		train_forces :: Whether to include in force training or not
		N_ions       :: Number of atoms of each species 
		N_atom       :: Total number of atoms
		species      :: (N_atom)                       List of species of each atom
		descriptor   :: (N_species, N_ions[iesp], nsf) Descriptors of each atom, ordered by element
		forces       :: (N_species, N_ions[iesp],3)    Reference forces
		coords       :: (N_species, N_ions[iesp],3)    Coordinates

		max_nb_struc   :: Maximum number of neighbors in the structure
		list_nblist    :: List of first neighbors of each atom
		list_sfderiv_i :: Derivatives of the decriptors with respect to the central atom
		list_sfderiv_j :: Derivatives of the decriptors with respect to the neighbor atoms
	"""
	def __init__(self, name, species, descriptor, energy, energy_atom_struc, sys_species, coords, forces, 
		         input_size, train_forces=False, list_nblist=None, list_sfderiv_i=None, list_sfderiv_j=None,):
		self.name = name
		self.energy = energy
		self.E_atomic_structure = energy_atom_struc
		self.train_forces = train_forces



		self.device = "cpu"
		# if torch.cuda.is_available():
		# 	self.device = "cuda:0"

		self.N_species = len(sys_species)
		self.N_ions = np.array([0 for iesp in range(self.N_species)])
		for iesp in species:
			self.N_ions[iesp] += 1
		self.N_atom = np.sum(self.N_ions)

		self.species   = []
		self.descriptor = [ [ [] for iat in range(self.N_ions[iesp]) ] for iesp in range(self.N_species) ]
		self.forces     = [ 0 for iat in range(self.N_atom) ]
		self.coords     = [ 0 for iat in range(self.N_atom) ]
		cont_iesp = [0 for i in range(self.N_species)]
		cont = 0
		for iesp in range(self.N_species):
			for iat in range(self.N_atom):
				if species[iat] == iesp:

					self.species.append(iesp)
					self.descriptor[iesp][cont_iesp[iesp]] = descriptor[iat]
					self.forces[cont] = forces[iat]
					self.coords[cont] = coords[iat]

					cont_iesp[iesp] += 1
					cont += 1

			self.descriptor[iesp] = torch.tensor(self.descriptor[iesp])
		self.forces = torch.tensor(self.forces)
		self.coords = torch.tensor(self.coords)


		if train_forces:

			self.max_nb_struc = np.max( np.array([ len(x) for x in list_nblist ]) )

			self.order          = torch.zeros(self.N_atom, dtype=int)
			self.list_nblist    = [ [] for iesp in range(self.N_species) ]
			self.list_sfderiv_i = [ torch.zeros(self.N_ions[iesp],input_size[iesp],3) for iesp in range(self.N_species) ]
			self.list_sfderiv_j = [ torch.zeros(self.N_ions[iesp],self.max_nb_struc,input_size[iesp],3) for iesp in range(self.N_species) ]

			cont_iesp = [0 for i in range(self.N_species)]
			cont      = 0
			for iesp in range(self.N_species):
				for iat in range(self.N_atom):
					if species[iat] == iesp:

						nnb = len(list_nblist[iat])
						self.list_sfderiv_i[iesp][cont_iesp[iesp]][:,:]      = list_sfderiv_i[iat]
						self.list_sfderiv_j[iesp][cont_iesp[iesp]][:nnb,:,:] = list_sfderiv_j[iat]
						self.list_nblist[iesp].append(list_nblist[iat])
						self.order[iat] = cont

						cont_iesp[iesp] += 1
						cont += 1

	def padding(self, max_nnb, input_size):
		"""
		Add trailing zeros to make all tensors of the same size
			max_nb_struc :: Maximum number of neighbors in the whole data set
			input_size   :: Size of the descriptors for each species
		"""
		
		# Take care if N_ions == 0:
		for iesp in range(self.N_species):
			if self.N_ions[iesp] == 0:
				self.descriptor[iesp] = torch.zeros( 0, input_size[iesp] )


		if self.train_forces:
			for iesp in range(self.N_species):
				if self.N_ions[iesp] == 0:
					self.list_sfderiv_i[iesp] = torch.zeros(self.N_ions[iesp],input_size[iesp],3)
					self.list_sfderiv_j[iesp] = torch.zeros(self.N_ions[iesp],self.max_nb_struc,input_size[iesp],3)

			# Add trailing zeros
			aux_nblist    = [ torch.ones(self.N_ions[iesp],max_nnb, dtype=int)* -1000000000 for iesp in range(self.N_species) ] 
			aux_sfderiv_j = [ torch.zeros(self.N_ions[iesp],max_nnb,input_size[iesp],3) for iesp in range(self.N_species) ]

			for iesp in range(self.N_species):
				for iat in range(self.N_ions[iesp]):
					nnb = len(self.list_nblist[iesp][iat])
					aux_nblist[iesp][iat,:nnb]        = torch.tensor(self.list_nblist[iesp][iat])-1
					aux_sfderiv_j[iesp][iat,:nnb,:,:] = self.list_sfderiv_j[iesp][iat][:nnb,:,:]


			sorted_nblist = [ torch.clone( aux_nblist[iesp] ) for iesp in range(self.N_species) ]
			N_atom_total = len(self.order)
			for iat in range(N_atom_total):
				if self.order[iat] != iat:
					for iesp in range(self.N_species):
						sorted_nblist[iesp] = torch.where( aux_nblist[iesp] == iat, self.order[iat], sorted_nblist[iesp] )


			self.list_nblist = sorted_nblist
			self.list_sfderiv_j = aux_sfderiv_j
