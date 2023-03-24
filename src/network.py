import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class NetAtom(nn.Module):
	"""
	ANN for each atomic element present in the training set
		input_size  :: Dimension of the descriptor vectors
		hidden_size :: Number of nodes in the hidden layers
		activations :: Activation functions of each layer
		functions   :: List of functions that are applied in the ANN. A series of
						Linear + Activation + Linear + Activation + ... + Linear
	"""
	def __init__(self, input_size, hidden_size, species, activations, alpha, device):
		super(NetAtom, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.species = species
		self.active_names = activations
		self.alpha = torch.tensor(alpha)
		self.device = device

		N_fun = [len(hidden_size[i])+1 for i in range(len(species)) ]
	
		self.linear  = nn.Identity()
		self.tanh    = nn.Tanh()
		self.sigmoid = nn.Sigmoid()
		self.activations = []
		for i in range(len(species)):
			aux = []
			for j in range(len(hidden_size[i])):
				if activations[i][j] == "linear":
					aux.append(self.linear)
				if activations[i][j] == "tanh":
					aux.append(self.tanh)
				if activations[i][j] == "sigmoid":
					aux.append(self.sigmoid)
			self.activations.append(aux)

		self.functions = []
		for i in range(len(species)):
			function_i = OrderedDict()
			name1 = "Linear_Sp"+str(i+1)+"_F"+str(1)
			name2 = "Active_Sp"+str(i+1)+"_F"+str(1)

			function_i[name1] = nn.Linear(input_size[i], hidden_size[i][0])
			function_i[name2] = self.activations[i][0]
			for j in range(1,N_fun[i]-1):
				name1 = "Linear_Sp"+str(i+1)+"_F"+str(j+1)
				name2 = "Active_Sp"+str(i+1)+"_F"+str(j+1)
				function_i[name1] = nn.Linear(hidden_size[i][j-1], hidden_size[i][j])
				function_i[name2] = self.activations[i][j]
			name1 = "Linear_Sp"+str(i+1)+"_F"+str(N_fun[i])
			function_i[name1] = nn.Linear(hidden_size[i][-1], 1)

			self.functions.append( nn.Sequential(function_i) ) 
		self.functions = nn.ModuleList(self.functions)


	def forward(self, grp_descrp, logic_reduce):
		"""
		[Energy training] Compute atomic energy for each atom in the current batch.
		INPUT:
			grp_descrp    :: Descriptors of the atoms of the batch, ordered by element, without considering to which structure belongs each
			logic_reduce  :: Auxiliar tensor to reorder the atomic contributions back to each structure
		OUTPUT:
			partial_E_ann :: atomic energies of all the atoms of all the structures grouped by species for the whole batch
			list_E_ann    :: total ANN energies of each structure in the batch
		"""

		# Compute atomic energies of all the atoms of each element
		partial_E_ann = [0 for i in range(len(self.species))]
		for iesp in range(len(self.species)):
			partial_E_ann[iesp] = self.functions[iesp](grp_descrp[iesp])

		# Gather back all atoms corresponding to the same strucuture from partial_E_ann
		list_E_ann = torch.zeros( (len(logic_reduce[0])), device=self.device ).double()
		for iesp in range(len(self.species)):
			list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", partial_E_ann[iesp], logic_reduce[iesp] )

		return list_E_ann


	def forward_F(self, group_descrp, group_sfderiv_i, group_sfderiv_j, group_indices_F,
				  grp_indices_F_i, logic_reduce, input_size, max_nnb):
		"""
		[Force training] Compute atomic energy and forces for each atom in the current batch.
		"""

		E_atomic_ann = [0 for i in range(len(self.species))] #torch.empty((0,1), requires_grad=True).double()
		aux_F_i      = torch.empty((0, 3 ), requires_grad=True, device=self.device).double()
		aux_F_j      = torch.empty((0, max_nnb, 3), device=self.device , requires_grad=True).double()

		for iesp in range( len(self.species) ):

			group_descrp[iesp].requires_grad_(True)

			partial_E_ann = self.functions[iesp]( group_descrp[iesp] )

			aux_dE_dG, = torch.autograd.grad(partial_E_ann, group_descrp[iesp],
                                    grad_outputs=partial_E_ann.data.new(partial_E_ann.shape).fill_(1),
                                    create_graph=True)#, retain_graph = True)

			E_atomic_ann[iesp] = partial_E_ann

			aux_F_j = torch.cat( (aux_F_j, torch.einsum("ik,ijkl->ijl", aux_dE_dG.double(), group_sfderiv_j[iesp].double())) )
			aux_F_i = torch.cat( (aux_F_i, torch.einsum("ik,ikl->il", aux_dE_dG.double(), group_sfderiv_i[iesp].double())) )



		aux_F_j    = torch.cat( ( aux_F_j, torch.zeros(1, aux_F_j.shape[1], 3, device=self.device) ) )
		aux_F_flat = aux_F_j.reshape( (aux_F_j.shape[0]*aux_F_j.shape[1],3) )
		
		aux_shape =  (group_indices_F.shape[0], group_indices_F.shape[1], 3)
		flatt_indices = torch.where(group_indices_F==-1,len(aux_F_flat)-1,group_indices_F)
		flatt_indices = flatt_indices.flatten()

		F_ann = -torch.index_select(aux_F_i,0,grp_indices_F_i) \
		        -torch.sum( torch.index_select(aux_F_flat,0,flatt_indices).reshape(aux_shape), dim=1 )


		list_E_ann = torch.zeros( (len(logic_reduce[0])), device=self.device ).double()
		for iesp in range(len(self.species)):
			list_E_ann = list_E_ann + torch.einsum( "ij,ki->k", E_atomic_ann[iesp], logic_reduce[iesp] )

		return list_E_ann, F_ann


	def get_loss_RMSE(self, grp_descrp, grp_energy, logic_reduce, grp_N_atom):
		"""
		[Energy training] Compute root mean squared error of energies in the batch
		"""

		list_E_ann = self.forward(grp_descrp, logic_reduce  )
		N_data = len(list_E_ann)

		differences = (list_E_ann - grp_energy)
		l2 = torch.sum( differences**2/grp_N_atom**2 )

		return l2, N_data


	def get_loss_RMSE_F(self, group_energy, group_N_atom,  group_descrp, group_sfderiv_i,
		                group_sfderiv_j, group_indices_F, grp_indices_F_i, logic_reduce,
		                group_forces, input_size, max_nnb, E_scaling):
		"""
		[Force training] Compute root mean squared error of energies and forces in the batch
		"""

		list_E_ann, list_F_ann = self.forward_F(group_descrp, group_sfderiv_i, group_sfderiv_j,
			                                    group_indices_F, grp_indices_F_i,
			                                    logic_reduce, input_size, max_nnb)

		N_data_E = len(list_E_ann)
		N_data_F = len(list_F_ann)
		
		diff_E = list_E_ann - group_energy
		diff_F = list_F_ann - group_forces

		l2_E = torch.sum( diff_E**2/group_N_atom**2 )
		l_F  = torch.sqrt( torch.sum( diff_F**2) / N_data_F )

		return l2_E, l_F, N_data_E, N_data_F
