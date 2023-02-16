import numpy as np
import torch
import pickle
import os


def custom_collate(batch):
	return batch


class PrepDataloader(object):
	"""
	Transforms the lists of data_classes.Structure objects, stored in data_set.StructureDataset.
	It gathers all the descriptors and their derivatives per element, and provides auxiliar tensors
	to transform the results of the ANN (atomic energy and forces) to their respective structures.
	"""
	def __init__(self, dataset, train_forces=False, batch_size=1, N_batch=1, sampler=None,
		         memory_mode="cpu", device="cpu", dataname="", generate=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.sampler = sampler
		self.train_forces = train_forces

		self.device = device
		self.dataname = dataname
		self.memory_mode = memory_mode

		if memory_mode == "cpu":
			self.device = "cpu"

		# Generate batches
		if generate:
			if self.sampler == None:
				self.sampler = list(range(len(dataset)))
				np.random.shuffle(self.sampler)

			self.N_batch = N_batch
			self.indexes = self.get_batch_indexes_N_batch()
			self.initialize()

			if memory_mode == "disk":
				if not os.path.exists("tmp_batches"):
					os.makedirs("tmp_batches")

			if train_forces:
				for ibatch in range(self.N_batch):
					self.prepare_batch_forces(ibatch)
					
					if memory_mode == "disk":
						self.save_batch(ibatch)
						self.del_batch(ibatch)

			else:
				self.prepare_batches()

				if memory_mode == "disk":
					for ibatch in range(self.N_batch):
						self.save_batch(ibatch)
						self.del_batch(ibatch)


	def __len__(self):
		return len(self.indexes)


	def __getitem__(self, index):
		"""
		Returns a list of the data needed for training one batch, depending if it is energy or force training
		[For each batch]
			group_descrp        :: 
			group_energy        :: 
			logic_tensor_reduce :: 
			index_from_database :: 
			group_N_atom        :: 
			group_forces        :: 
			group_sfderiv_i     :: 
			group_sfderiv_j     :: 
			group_indices_F     :: 
			group_indices_F_i   :: 
		"""
		if self.train_forces:
			return self.group_descrp[index], self.group_energy[index], self.logic_tensor_reduce[index], \
			       self.index_from_database[index], self.group_N_atom[index], self.group_forces[index], \
			       self.group_sfderiv_i[index], self.group_sfderiv_j[index], self.group_indices_F[index], \
			       self.group_indices_F_i[index]
		else:
			return self.group_descrp[index], self.group_energy[index], self.logic_tensor_reduce[index], \
			       self.index_from_database[index], self.group_N_atom[index]


	def get_number_data(self):
		return len(self.dataset)


	def get_batch_indexes_N_batch(self):
		"""
		Returns the indexes of the structures in StructureDataset that belong to each batch
		"""
		finish = 0
		indexes = []

		base, extra = divmod(len(self.sampler), self.N_batch)
		N_per_batch = np.array([base + (i < extra) for i in range(self.N_batch)])

		np.random.shuffle(N_per_batch)

		for i in range(self.N_batch):
			start = finish
			finish = start + N_per_batch[i]
			indexes.append([start,finish])

		indexes = np.array(indexes)

		return indexes


	def initialize(self):
		"""
		Initialize all the data to None for each batch
		"""
		self.group_energy        = [ None for ibatch in range(self.N_batch) ]
		self.group_descrp        = [ None for ibatch in range(self.N_batch) ]
		self.logic_tensor_reduce = [ None for ibatch in range(self.N_batch) ]
		self.index_from_database = [ None for ibatch in range(self.N_batch) ]
		self.group_N_atom        = [ None for ibatch in range(self.N_batch) ]
                                                               
		self.group_sfderiv_i     = [ None for ibatch in range(self.N_batch) ]
		self.group_sfderiv_j     = [ None for ibatch in range(self.N_batch) ]
		self.group_forces        = [ None for ibatch in range(self.N_batch) ]
		self.group_indices_F     = [ None for ibatch in range(self.N_batch) ]
		self.group_indices_F_i   = [ None for ibatch in range(self.N_batch) ]

		self.batch_names = [ "./tmp_batches/"+self.dataname+"batch_energy{:}.pkl".format(ibatch) for ibatch in range(self.N_batch) ]


	def save_batch(self, ibatch):
		if self.train_forces:
			save = [self.group_descrp[ibatch]       , 
					self.group_energy[ibatch]       , 
					self.logic_tensor_reduce[ibatch],
					self.index_from_database[ibatch],
					self.group_N_atom[ibatch]       , 
					self.group_forces[ibatch]       , 
					self.group_sfderiv_i[ibatch]    , 
					self.group_sfderiv_j[ibatch]    , 
					self.group_indices_F[ibatch]    , 
					self.group_indices_F_i[ibatch]  ,]
		else:
			save = [self.group_descrp[ibatch]       , 
					self.group_energy[ibatch]       , 
					self.logic_tensor_reduce[ibatch],
					self.index_from_database[ibatch],
					self.group_N_atom[ibatch]       ,]

		name = self.batch_names[ibatch]
		torch.save(save, name, pickle_protocol=pickle.HIGHEST_PROTOCOL)


	def load_batch(self, ibatch):
		name = self.batch_names[ibatch]
		data = torch.load(name)

		self.group_descrp[ibatch] = data[0]
		self.group_energy[ibatch] = data[1]
		self.logic_tensor_reduce[ibatch] = data[2]
		self.index_from_database[ibatch] = data[3]
		self.group_N_atom[ibatch] = data[4]

		if self.train_forces:
			self.group_forces[ibatch] = data[5]
			self.group_sfderiv_i[ibatch] = data[6]
			self.group_sfderiv_j[ibatch] = data[7]
			self.group_indices_F[ibatch] = data[8]
			self.group_indices_F_i[ibatch] = data[9]


	def del_batch(self,ibatch):
		self.group_energy[ibatch] = None
		self.group_descrp[ibatch] = None
		self.logic_tensor_reduce[ibatch] = None
		self.index_from_database[ibatch] = None
		self.group_N_atom[ibatch] = None
		self.group_sfderiv_i[ibatch] = None
		self.group_sfderiv_j[ibatch] = None
		self.group_forces[ibatch] = None
		self.group_indices_F[ibatch] = None
		self.group_indices_F_i[ibatch] = None


	def gather_data(self, memory_mode):

		old_memory_mode  = self.memory_mode
		self.memory_mode = memory_mode

		if old_memory_mode == "disk" and self.memory_mode == "cpu":
			for ibatch in range(self.N_batch):
				self.load_batch(ibatch)
				self.batch_data_cpu_to_gpu(ibatch, "cpu")

		elif old_memory_mode == "disk" and self.memory_mode == "gpu":
			for ibatch in range(self.N_batch):
				self.load_batch(ibatch)
				self.batch_data_cpu_to_gpu(ibatch, self.device)


		elif old_memory_mode in ["cpu", "gpu"] and self.memory_mode == "disk":
			for ibatch in range(self.N_batch):
				self.save_batch(ibatch)
				self.del_batch(ibatch)



	def batch_data_cpu_to_gpu(self, ibatch, device):

		if self.train_forces:
			self.group_energy[ibatch] = self.group_energy[ibatch].to(device)
			self.group_N_atom[ibatch] = self.group_N_atom[ibatch].to(device)
			self.group_forces[ibatch] = self.group_forces[ibatch].to(device)
			self.group_indices_F[ibatch] = self.group_indices_F[ibatch].to(device)
			self.group_indices_F_i[ibatch] = self.group_indices_F_i[ibatch].to(device)

			for iesp in range( len(self.group_descrp[ibatch]) ):
				self.group_descrp[ibatch][iesp] = self.group_descrp[ibatch][iesp].to(device)
				self.logic_tensor_reduce[ibatch][iesp] = self.logic_tensor_reduce[ibatch][iesp].to(device)
				self.group_sfderiv_i[ibatch][iesp] = self.group_sfderiv_i[ibatch][iesp].to(device)
				self.group_sfderiv_j[ibatch][iesp] = self.group_sfderiv_j[ibatch][iesp].to(device)

		else:
			self.group_energy[ibatch] = self.group_energy[ibatch].to(device)
			self.group_N_atom[ibatch] = self.group_N_atom[ibatch].to(device)

			for iesp in range( len(self.group_descrp[ibatch]) ):
				self.group_descrp[ibatch][iesp] = self.group_descrp[ibatch][iesp].to(device)
				self.logic_tensor_reduce[ibatch][iesp] = self.logic_tensor_reduce[ibatch][iesp].to(device)



	def prepare_batch_forces(self, ibatch):
		"""
		Prepare one batch for force training. Group per element and compute tensors to regroup in the end.
			group_descrp        :: (N_species,N_atom_iesp,Nsf)         List of descriptors
			group_energy        :: (N_species,N_atom_iesp)             List of energies
			logic_tensor_reduce :: (N_species,N_structure,N_atom_iesp) Tensor to regroup energies in the end
			index_from_database :: (N_structure)                       
			group_N_atom        :: (N_structure)                       Number of atoms in each structure
			group_forces        :: (N_atom_batch,3)                    Forces of all the atoms in the batch
			group_sfderiv_i     :: (N_specie,N_atom_iesp,nsf,3)        Derivatives wrt the central atom
			group_sfderiv_j     :: (N_specie,N_atom_iesp,nnb,nsf,3)    Derivatives wrt neighbor atoms
			group_indices_F     :: (N_atom_batch,N_max)                Order F_ij
			group_indices_F_i   :: (N_atom_batch)                      Order F_ii
		"""

		index = self.indexes[ibatch]

		# (N_species) Number of atoms of each element in each structure
		group_N_atom_iesp = torch.zeros(self.dataset.N_species, dtype=int)
		# (N_struc, N_especie) Number of atomos of each element in the batch
		group_N_ions      = []
		# (N_struc) Number of atoms in each structure
		group_N_atom      = [] 

		for istruc in range(index[0], index[1]):
			index_struc = self.sampler[istruc]
			group_N_atom.append( self.dataset[index_struc].N_atom )
			group_N_ions.append( self.dataset[index_struc].N_ions )

			for iesp in range(self.dataset.N_species):
				group_N_atom_iesp[iesp] = group_N_atom_iesp[iesp] + self.dataset[index_struc].N_ions[iesp]
		
		group_N_atom = torch.tensor( np.array(group_N_atom), dtype=int )
		group_N_ions = torch.tensor( np.array(group_N_ions), dtype=int )


		group_energy        = []
		index_from_database = []
		group_descrp      = [ torch.empty( 0,self.dataset.input_size[iesp] ).double() for iesp in range(self.dataset.N_species) ]
		group_forces      = torch.empty( 0,3 ).double()
		group_sfderiv_i   = [ torch.empty( 0, self.dataset.input_size[iesp], 3) for iesp in range(self.dataset.N_species) ]
		group_sfderiv_j   = [ torch.empty( 0, self.dataset.max_nnb, self.dataset.input_size[iesp], 3) for iesp in range(self.dataset.N_species) ]
		group_indices_F_i = torch.zeros( torch.sum(group_N_atom_iesp).item(),dtype=int ) - 1

		# (N_specie, N_atom_iesp, max_nnb)
		group_nblist = [ torch.empty( 0, self.dataset.max_nnb, dtype=int) for iesp in range(self.dataset.N_species) ]

		# Group descriptors and derivatives
		# group_indices_F_i :: 

		commands_i = [ "torch.cat((" for iesp in range(self.dataset.N_species) ]
		commands_j = [ "torch.cat((" for iesp in range(self.dataset.N_species) ]
		ind_start_struc = 0
		cont = 0
		for istruc in range(index[0], index[1]):

			index_struc = self.sampler[istruc]
			group_energy.append( self.dataset[index_struc].energy )
			group_forces = torch.cat( ( group_forces, self.dataset[index_struc].forces ) )

			index_from_database.append( [self.dataset[index_struc].name, self.dataset[index_struc].E_atomic_structure ] )

			ind_start_iesp = 0

			for iesp in range(self.dataset.N_species):
				group_descrp[iesp]    = torch.cat( ( group_descrp[iesp], self.dataset[index_struc].descriptor[iesp] ) )
				group_nblist[iesp]    = torch.cat( ( group_nblist[iesp], self.dataset[index_struc].list_nblist[iesp] + ind_start_struc ) )

				commands_i[iesp] += "self.dataset[{:}].list_sfderiv_i[{:}], ".format(index_struc,iesp)
				commands_j[iesp] += "self.dataset[{:}].list_sfderiv_j[{:}], ".format(index_struc,iesp)

				ind1 = ind_start_iesp  + torch.sum(group_N_ions[:cont,iesp]).item()
				ind2 = ind_start_struc + torch.sum(group_N_ions[cont,:iesp]).item()

				N_ions_istruc = group_N_ions[cont,iesp].item()
				group_indices_F_i[ind2:ind2+N_ions_istruc] = torch.arange(ind1, ind1+N_ions_istruc)

				ind_start_iesp += group_N_atom_iesp[iesp].item()

			ind_start_struc += self.dataset[index_struc].N_atom
			cont += 1

		for iesp in range(self.dataset.N_species):
			commands_i[iesp] += "))"
			commands_j[iesp] += "))"
			group_sfderiv_i[iesp] = eval(commands_i[iesp])
			group_sfderiv_j[iesp] = eval(commands_j[iesp])

		group_energy = torch.tensor( np.array(group_energy) ).double()
		group_N_atom = torch.tensor( np.array(group_N_atom) ).double()


		# Logical tensor to reorder energy output into their corresponding structures
		# list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", E_atomic_ann[iesp], logic_reduce[iesp] )
		logic_tensor_reduce = []
		for iesp in range(self.dataset.N_species):

			aux_iesp = []

			ind1 = 0
			cont_index_struc = index[0]
			for istruc in range( len(group_energy) ):
				aux = np.zeros(len(group_descrp[iesp]))

				index_struc = self.sampler[cont_index_struc]
				ind2 = ind1 + self.dataset[index_struc].N_ions[iesp]

				aux[ind1:ind2] += 1
				ind1 = ind2
				cont_index_struc += 1

				aux_iesp.append(aux)

			aux_iesp = torch.tensor( np.array(aux_iesp) ).double()
			logic_tensor_reduce.append(aux_iesp)



		# Indices to reorder self forces to their structures: F_ii
		N_atom_batch  = int( torch.sum( group_N_atom ).item() )

		command = "torch.cat(("
		group_nblist_flat = torch.empty(0, dtype=int)
		for iesp in range(self.dataset.N_species):
			command += "torch.flatten(group_nblist[{:}]), ".format(iesp)
		command += "))"
		group_nblist_flat = eval(command)

		group_nblist_flat = torch.where( group_nblist_flat < 0, -1, group_nblist_flat )

		N_max = 0
		aux_indices = []
		for i_atom in range(N_atom_batch):
			mask = (group_nblist_flat == i_atom)
			aux_indices.append(mask.nonzero().squeeze())
			auxn = len(aux_indices[i_atom])
			N_max = max(N_max,auxn)

		group_indices_F = torch.zeros((N_atom_batch, N_max),dtype=int) - 1
		for i_atom in range(N_atom_batch):
			auxn = len(aux_indices[i_atom])
			group_indices_F[i_atom,:auxn] = aux_indices[i_atom]


		self.group_energy[ibatch]        = group_energy.to(self.device)
		self.group_descrp[ibatch]        = group_descrp
		self.logic_tensor_reduce[ibatch] = logic_tensor_reduce
		self.index_from_database[ibatch] = index_from_database#.to(self.device)
		self.group_N_atom[ibatch]        = group_N_atom.to(self.device)
                                                               
		self.group_sfderiv_i[ibatch]     = group_sfderiv_i
		self.group_sfderiv_j[ibatch]     = group_sfderiv_j
		self.group_forces[ibatch]        = group_forces.to(self.device)
		self.group_indices_F[ibatch]     = group_indices_F.to(self.device)
		self.group_indices_F_i[ibatch]   = group_indices_F_i.to(self.device)


		for iesp in range(self.dataset.N_species):
			self.group_descrp[ibatch][iesp] = self.group_descrp[ibatch][iesp].to(self.device)
			self.logic_tensor_reduce[ibatch][iesp] = self.logic_tensor_reduce[ibatch][iesp].to(self.device)
			self.group_sfderiv_i[ibatch][iesp] = self.group_sfderiv_i[ibatch][iesp].to(self.device)
			self.group_sfderiv_j[ibatch][iesp] = self.group_sfderiv_j[ibatch][iesp].to(self.device)

		#return group_energy, group_descrp, logic_tensor_reduce, index_from_database, group_N_atom, \
		#	   group_sfderiv_i, group_sfderiv_j, group_forces, group_indices_F, group_indices_F_i



	def prepare_batches(self):
		"""
		Prepare batches for only energy training. See previous method for a more detailed explanation
		"""

		group_descrp = [ [ torch.empty( 0,self.dataset.input_size[iesp] ).double() for iesp in range(self.dataset.N_species) ] for ibatch in range(self.N_batch)]
		group_energy = [ [] for ibatch in range(self.N_batch) ]
		index_from_database = [ [] for ibatch in range(self.N_batch) ]
		group_N_atom = [ [] for ibatch in range(self.N_batch) ]


		for ibatch in range(self.N_batch):
			index = self.indexes[ibatch]

			ind_start_struc = 0

			for istruc in range(index[0], index[1]):

				index_struc = self.sampler[istruc]
				group_energy[ibatch].append( self.dataset[index_struc].energy )
				group_N_atom[ibatch].append( self.dataset[index_struc].N_atom )

				index_from_database[ibatch].append( [self.dataset[index_struc].name, self.dataset[index_struc].E_atomic_structure ] )

				ind_start_iesp = 0

				for iesp in range(self.dataset.N_species):
					group_descrp[ibatch][iesp]    = torch.cat( ( group_descrp[ibatch][iesp], self.dataset[index_struc].descriptor[iesp] ) )

			group_energy[ibatch] = torch.tensor( np.array(group_energy[ibatch]) ).double()
			group_N_atom[ibatch] = torch.tensor( np.array(group_N_atom[ibatch]) ).double()

		# Logical tensor:
		logic_tensor_reduce = []

		for ibatch in range(self.N_batch):
			index = self.indexes[ibatch]

			aux_batch = []

			for iesp in range(self.dataset.N_species):

				aux_iesp = []

				ind1 = 0
				cont_index_struc = index[0]
				for istruc in range( len(group_energy[ibatch]) ):
					aux = np.zeros(len(group_descrp[ibatch][iesp]))

					index_struc = self.sampler[cont_index_struc]
					ind2 = ind1 + self.dataset[index_struc].N_ions[iesp]

					aux[ind1:ind2] += 1


					ind1 = ind2
					cont_index_struc += 1

					aux_iesp.append(aux)

				aux_iesp = np.array(aux_iesp)
				aux_iesp = torch.Tensor(aux_iesp).double()
				aux_batch.append(aux_iesp)
			logic_tensor_reduce.append(aux_batch)


		for ibatch in range(self.N_batch):
			self.group_energy[ibatch]        = group_energy[ibatch].to(self.device)
			self.group_descrp[ibatch]        = group_descrp[ibatch]
			self.logic_tensor_reduce[ibatch] = logic_tensor_reduce[ibatch]
			self.index_from_database[ibatch] = index_from_database[ibatch]#.to(self.device)
			self.group_N_atom[ibatch]        = group_N_atom[ibatch].to(self.device)
	                                                               
			for iesp in range(self.dataset.N_species):
				self.group_descrp[ibatch][iesp] = self.group_descrp[ibatch][iesp].to(self.device)
				self.logic_tensor_reduce[ibatch][iesp] = self.logic_tensor_reduce[ibatch][iesp].to(self.device)
