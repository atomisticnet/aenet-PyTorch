import numpy as np
import torch


class StructureDataset(object):
	"""
	Class containing the list of data_classes.Structure objects.
	Defines methods to normalize energy, forces, descriptors and its derivatives.
	"""
	def __init__(self, list_structures, sys_species, input_size, max_nnb):

		self.list_struc  = list_structures
		self.sys_species = sys_species
		self.N_species   = len(sys_species)
		self.input_size  = input_size
		self.max_nnb     = max_nnb

		self.device = "cpu"
		if torch.cuda.is_available():
			self.device = "cuda:0"


	def __len__ (self):
		return len(self.list_struc)

	def __getitem__(self, index):
		return self.list_struc[index]

	def get_names_and_E_atomic_structure(self):
		dataset_names = []
		dataset_E_atomic_structure = []
		for struc in self.list_struc:
			dataset_names.append(struc.name)
			dataset_E_atomic_structure.append(struc.E_atomic_structure)

		return dataset_names,dataset_E_atomic_structure

	def get_species(self):
		return self.species


	def normalize_E(self, E_scaling, E_shift):

		for struc in self.list_struc:
			struc.energy = (struc.energy - struc.N_atom*E_shift)*E_scaling


	def normalize_F(self, E_scaling, E_shift):

		with torch.no_grad():
			for struc in self.list_struc:
				struc.forces = struc.forces*E_scaling


	def normalize_stp(self, sfval_avg, sfval_cov):

		shift = [ ]
		scale = [ ]
		for iesp in range(self.N_species):

			sh = torch.Tensor(sfval_avg[iesp]).double()
			sc = 1/torch.sqrt(torch.Tensor(sfval_cov[iesp]).double() - sh**2)

			shift.append(sh)
			scale.append(sc)


		for struc in self.list_struc:
			for iesp in range(self.N_species):

				struc.descriptor[iesp] = (struc.descriptor[iesp] - shift[iesp])*scale[iesp]

				if struc.train_forces:
					struc.list_sfderiv_i[iesp] = torch.einsum( "ijk,j->ijk", struc.list_sfderiv_i[iesp], scale[iesp] )
					struc.list_sfderiv_j[iesp] = torch.einsum( "ijkl,k->ijkl", struc.list_sfderiv_j[iesp], scale[iesp] )

		return shift, scale




class GroupedDataset(object):
	"""
	Dataset object containing the data of the structures select for force training and the ones
	for only energy training.
	It is ready to be used with torch.utils.data.DataLoader. Each item of this dataset is one
	batch of data of the size tin.batch_size
	The input are data_loader.PrepDataloader() objects, with the atoms already ordered per element
	"""

	def __init__(self, energy_data, forces_data, generate=True, memory_mode="cpu", device="cpu", dataname=""):
		self.memory_mode = memory_mode
		self.device      = device
		self.dataname    = dataname

		self.train_energy = False
		self.train_forces = False
		if energy_data != None:
			self.N_batch = energy_data.N_batch
			self.train_energy = True
		if forces_data != None:
			self.N_batch = forces_data.N_batch
			self.train_forces = True

		if generate:
			gather_data = self.gather_data(energy_data, forces_data)


	def __len__ (self):
		return self.N_batch

	def __getitem__(self, index):
		"""
		Returns the information needed for training one batch, in the training device.
		It considers three options: GPU(gpu), GPU(cpu), disk (not public yet)
		"""
		if self.memory_mode == "cpu" or self.memory_mode == "gpu":
			data= [self.F_group_descrp[index], self.F_group_energy[index], self.F_logic_tensor_reduce[index], \
			       self.F_index_from_database[index], self.F_group_N_atom[index], self.F_group_forces[index], \
			       self.F_group_sfderiv_i[index], self.F_group_sfderiv_j[index], self.F_group_indices_F[index], \
			       self.F_group_indices_F_i[index],
			       self.E_group_descrp[index], self.E_group_energy[index], self.E_logic_tensor_reduce[index], \
			       self.E_index_from_database[index], self.E_group_N_atom[index], ]

		elif self.memory_mode == "disk":
			if self.train_energy:
				data_E = torch.load( self.E_batch_names[index])
			else:
				data_E = [ None for i in range(5) ]
			if self.train_forces:
				data_F = torch.load( self.F_batch_names[index])
			else:
				data_F = [ None for i in range(10) ]

			data = data_F + data_E

		if self.device == "cuda:0":
			data = self.batch_data_cpu_to_gpu(data)

		return data


	def batch_data_cpu_to_gpu(self, data):
		"""
		Move information of a batch (an item of the dataset) from cpu to gpu memory
		"""
		data_gpu = [ [] for i in range(len(data))]
		if self.train_forces:
			data_gpu[3] = data[3]
			data_gpu[1] = data[1].to(self.device)
			data_gpu[4] = data[4].to(self.device)
			data_gpu[5] = data[5].to(self.device)
			data_gpu[8] = data[8].to(self.device)
			data_gpu[9] = data[9].to(self.device)

			for iesp in range(len(data[0])):
				data_gpu[0].append(data[0][iesp].to(self.device))
				data_gpu[2].append(data[2][iesp].to(self.device))
				data_gpu[6].append(data[6][iesp].to(self.device))
				data_gpu[7].append(data[7][iesp].to(self.device))

		if self.train_energy:
			data_gpu[13] = data[13]
			data_gpu[11] = data[11].to(self.device)
			data_gpu[14] = data[14].to(self.device)

			for iesp in range(len(data[10])):
				data_gpu[10].append(data[10][iesp].to(self.device))
				data_gpu[12].append(data[12][iesp].to(self.device))

		return data_gpu



	def gather_data(self, energy_data, forces_data):
		"""
		Join data of the energy and forces data_loader.PrepDataloader() objects into a single object
		"""

		if self.memory_mode == "cpu" or self.memory_mode == "gpu":
			self.E_group_energy        = [ None for ibatch in range(self.N_batch)]
			self.E_group_descrp        = [ None for ibatch in range(self.N_batch)]
			self.E_logic_tensor_reduce = [ None for ibatch in range(self.N_batch)]
			self.E_index_from_database = [ None for ibatch in range(self.N_batch)]
			self.E_group_N_atom        = [ None for ibatch in range(self.N_batch)]

			self.F_group_energy        = [ None for ibatch in range(self.N_batch)]
			self.F_group_descrp        = [ None for ibatch in range(self.N_batch)]
			self.F_logic_tensor_reduce = [ None for ibatch in range(self.N_batch)]
			self.F_index_from_database = [ None for ibatch in range(self.N_batch)]
			self.F_group_N_atom        = [ None for ibatch in range(self.N_batch)]
	                                                               
			self.F_group_sfderiv_i     = [ None for ibatch in range(self.N_batch)]
			self.F_group_sfderiv_j     = [ None for ibatch in range(self.N_batch)]
			self.F_group_forces        = [ None for ibatch in range(self.N_batch)]
			self.F_group_indices_F     = [ None for ibatch in range(self.N_batch)]
			self.F_group_indices_F_i   = [ None for ibatch in range(self.N_batch)]

			if self.train_energy:
				for ibatch in range(energy_data.N_batch):

					self.E_group_energy[ibatch]        = energy_data.group_energy[ibatch]
					self.E_group_descrp[ibatch]        = energy_data.group_descrp[ibatch]
					self.E_logic_tensor_reduce[ibatch] = energy_data.logic_tensor_reduce[ibatch]
					self.E_index_from_database[ibatch] = energy_data.index_from_database[ibatch]
					self.E_group_N_atom[ibatch]        = energy_data.group_N_atom[ibatch]

			if self.train_forces:
				for ibatch in range(forces_data.N_batch):

					self.F_group_energy[ibatch] = forces_data.group_energy[ibatch]
					self.F_group_descrp[ibatch] = forces_data.group_descrp[ibatch]
					self.F_logic_tensor_reduce[ibatch] = forces_data.logic_tensor_reduce[ibatch]
					self.F_index_from_database[ibatch] = forces_data.index_from_database[ibatch]
					self.F_group_N_atom[ibatch] = forces_data.group_N_atom[ibatch]

					self.F_group_sfderiv_i[ibatch] =  forces_data.group_sfderiv_i[ibatch] 
					self.F_group_sfderiv_j[ibatch] =  forces_data.group_sfderiv_j[ibatch] 
					self.F_group_forces[ibatch] =  forces_data.group_forces[ibatch] 
					self.F_group_indices_F[ibatch] =  forces_data.group_indices_F[ibatch] 
					self.F_group_indices_F_i[ibatch] =  forces_data.group_indices_F_i[ibatch]

		if True: #self.memory_mode == "disk":
			self.E_batch_names = [ None for ibatch in range(self.N_batch) ]
			self.F_batch_names = [ None for ibatch in range(self.N_batch) ]
			if self.train_energy:
				self.E_batch_names = energy_data.batch_names
			if self.train_forces:
				self.F_batch_names = forces_data.batch_names


	def save_batches(self, N_remove, N_struc_E, N_struc_F, max_nnb, trainset_params, setup_params, networks_param):
		"""
		Save information of each batch to disk, for future use in other trainings
		"""
		if not os.path.exists("tmp_batches"):
			os.makedirs("tmp_batches")

		# Save necessary information that is only available from read_trainfile:
		save = {"E_batch_names"   : self.E_batch_names,
				"F_batch_names"   : self.F_batch_names, 
				"N_batch"         : self.N_batch,
				"train_energy"    : self.train_energy,
				"train_forces"    : self.train_forces,
				"max_nnb"         : max_nnb, 
				"N_remove"        : N_remove, 
				"N_struc_E"       : N_struc_E, 
				"N_struc_F"       : N_struc_F, 
				"trainset_params" : trainset_params,
				"setup_params"    : setup_params,
				"networks_param"  : networks_param}

		name = "./tmp_batches/"+self.dataname+"_info"
		torch.save(save, name, pickle_protocol=pickle.HIGHEST_PROTOCOL)

		if self.memory_mode in ["cpu", "gpu"]:

			if self.train_forces:
				for ibatch in range(self.N_batch):
					save = [self.F_group_descrp[ibatch]       , 
							self.F_group_energy[ibatch]       , 
							self.F_logic_tensor_reduce[ibatch],
							self.F_index_from_database[ibatch],
							self.F_group_N_atom[ibatch]       , 
							self.F_group_forces[ibatch]       , 
							self.F_group_sfderiv_i[ibatch]    , 
							self.F_group_sfderiv_j[ibatch]    , 
							self.F_group_indices_F[ibatch]    , 
							self.F_group_indices_F_i[ibatch]  ,]

					name = self.F_batch_names[ibatch]
					torch.save(save, name, pickle_protocol=pickle.HIGHEST_PROTOCOL)

			if self.train_energy:
				for ibatch in range(self.N_batch):
					save = [self.E_group_descrp[ibatch]       , 
							self.E_group_energy[ibatch]       , 
							self.E_logic_tensor_reduce[ibatch],
							self.E_index_from_database[ibatch],
							self.E_group_N_atom[ibatch]       , ]

					name = self.E_batch_names[ibatch]
					torch.save(save, name, pickle_protocol=pickle.HIGHEST_PROTOCOL)



	def load_batches(self):
		"""
		Load information of each batch to disk, saved by save_batches
		"""
		name = "./tmp_batches/"+self.dataname+"_info"
		save = torch.load(name)
		
		self.E_batch_names = save["E_batch_names"]
		self.F_batch_names = save["F_batch_names"]
		self.N_batch       = save["N_batch"]
		self.train_energy  = save["train_energy"]
		self.train_forces  = save["train_forces"]

		N_remove         = save["N_remove"]
		N_struc_E        = save["N_struc_E"]
		N_struc_F        = save["N_struc_F"]
		max_nnb          = save["max_nnb"]
		trainset_params  = save["trainset_params"]
		setup_params     = save["setup_params"]
		networks_param   = save["networks_param"]

		if self.memory_mode == "cpu" or self.memory_mode == "gpu":
			self.E_group_energy        = [ None for ibatch in range(self.N_batch)]
			self.E_group_descrp        = [ None for ibatch in range(self.N_batch)]
			self.E_logic_tensor_reduce = [ None for ibatch in range(self.N_batch)]
			self.E_index_from_database = [ None for ibatch in range(self.N_batch)]
			self.E_group_N_atom        = [ None for ibatch in range(self.N_batch)]

			self.F_group_energy        = [ None for ibatch in range(self.N_batch)]
			self.F_group_descrp        = [ None for ibatch in range(self.N_batch)]
			self.F_logic_tensor_reduce = [ None for ibatch in range(self.N_batch)]
			self.F_index_from_database = [ None for ibatch in range(self.N_batch)]
			self.F_group_N_atom        = [ None for ibatch in range(self.N_batch)]
	                                                               
			self.F_group_sfderiv_i     = [ None for ibatch in range(self.N_batch)]
			self.F_group_sfderiv_j     = [ None for ibatch in range(self.N_batch)]
			self.F_group_forces        = [ None for ibatch in range(self.N_batch)]
			self.F_group_indices_F     = [ None for ibatch in range(self.N_batch)]
			self.F_group_indices_F_i   = [ None for ibatch in range(self.N_batch)]

			for ibatch in range(self.N_batch):

				data = torch.load(self.E_batch_names[ibatch])
				if self.train_energy:
					data_E = torch.load( self.E_batch_names[ibatch])
				else:
					data_E = [ None for i in range(5) ]
				if self.train_forces:
					data_F = torch.load( self.F_batch_names[ibatch])
				else:
					data_F = [ None for i in range(10) ]

				data = data_F + data_E

				if self.memory_mode == "gpu": batch_data_cpu_to_gpu(data)

				self.F_group_descrp[ibatch] = data[0]
				self.F_group_energy[ibatch] = data[1]
				self.F_logic_tensor_reduce[ibatch] = data[2]
				self.F_index_from_database[ibatch] = data[3]
				self.F_group_N_atom[ibatch] = data[4]
				self.F_group_forces[ibatch] = data[5]
				self.F_group_sfderiv_i[ibatch] = data[6]
				self.F_group_sfderiv_j[ibatch] = data[7]
				self.F_group_indices_F[ibatch] = data[8]
				self.F_group_indices_F_i[ibatch] = data[9]
				self.E_group_descrp[ibatch] = data[10]
				self.E_group_energy[ibatch] = data[11]
				self.E_logic_tensor_reduce[ibatch] = data[12]
				self.E_index_from_database[ibatch] = data[13]
				self.E_group_N_atom[ibatch] = data[14]

		return N_remove, N_struc_E, N_struc_F, max_nnb, trainset_params, setup_params, networks_param