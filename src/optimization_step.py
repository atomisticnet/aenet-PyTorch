from network import *


def step_train_any(net, train_data, E_scaling, input_size, max_nnb):
	"""
	Run one training step or epoch, i.e., go once over the whole training set
	Returns:
		train_error :: Total training error of the whole set
		E_error     :: Energy RMSE of the training set
		F_error     :: Force RMSE of the training set
	"""
	train_error = 0.0
	N_data_forces = 0
	N_data_energy = 0
	E_error = 0.0
	F_error = 0.0

	for i_batch, data_batch in enumerate(train_data):
		"""
		data_batch[0]  :: group_descrp
		data_batch[1]  :: group_energy
		data_batch[2]  :: logic_tensor_reduce
		data_batch[3]  :: index_from_database
		data_batch[4]  :: group_N_atom
		data_batch[5]  :: group_forces
		data_batch[6]  :: group_sfderiv_i
		data_batch[7]  :: group_sfderiv_j
		data_batch[8]  :: group_indices_F
		data_batch[9]  :: group_indices_F_i
		data_batch[10] :: train_forces_batch
		"""

		# If forces
		l1_E = torch.zeros(1, device=net.device)
		l1_F = torch.zeros(1, device=net.device)
		l2_E = torch.zeros(1, device=net.device)
		N_data1_F = 0
		N_data1_E = 0
		N_data2_E = 0
		if train_data.dataset.train_forces:
			l1_E, l1_F, N_data1_E, N_data1_F = net.get_loss_RMSE_F(data_batch[0][1], data_batch[0][4],  data_batch[0][0], data_batch[0][6],
		                                                         data_batch[0][7], data_batch[0][8], data_batch[0][9], data_batch[0][2],
		                                                         data_batch[0][5], input_size, max_nnb, E_scaling)

			N_data_forces += N_data1_F
			N_data_energy += N_data1_E

		if train_data.dataset.train_energy:
			l2_E, N_data2_E = net.get_loss_RMSE(data_batch[0][10], data_batch[0][11], data_batch[0][12], data_batch[0][14])

			N_data_energy += N_data2_E

		N_data_E_tot = N_data1_E + N_data2_E
		loss_tot = (1.0-net.alpha)*torch.sqrt((l1_E + l2_E)/N_data_E_tot) + net.alpha*l1_F
		net.optimizer.zero_grad()
		loss_tot.backward()
		net.optimizer.step()

		E_error += l1_E.item() + l2_E.item()
		F_error += l1_F.item()**2*N_data1_F

	E_error = np.sqrt(E_error/N_data_energy)
	if N_data_forces != 0:
		F_error = np.sqrt(F_error/N_data_forces)
	else:
		F_error = 0.0
	train_error = (1.0-net.alpha.item())*E_error + net.alpha.item()*F_error

	return train_error, E_error, F_error


def step_valid_any(net, valid_data, E_scaling, input_size, max_nnb):
	"""
	Run one testing step or epoch, i.e., go once over the whole testing set
	Returns:
		valid_error :: Total testing error of the whole set
		MAE         :: Deprecated
		E_error     :: Energy RMSE of the testing set
		F_error     :: Force RMSE of the testing set
	"""
	valid_error = 0.0
	N_data_forces = 0
	N_data_energy = 0
	E_error = 0.0
	F_error = 0.0

	for i_batch, data_batch in enumerate(valid_data):
		"""
		data_batch[10+0]  :: group_descrp
		data_batch[10+1]  :: group_energy
		data_batch[10+2]  :: logic_tensor_reduce
		data_batch[10+3]  :: index_from_database
		data_batch[10+4]  :: group_N_atom
		"""
		l1_E = torch.zeros(1, device=net.device)
		l1_F = torch.zeros(1, device=net.device)
		l2_E = torch.zeros(1, device=net.device)
		N_data1_F = 0
		N_data1_E = 0
		N_data2_E = 0
		if valid_data.dataset.train_forces:
			l1_E, l1_F, N_data1_E, N_data1_F = net.get_loss_RMSE_F(data_batch[0][1], data_batch[0][4],  data_batch[0][0], data_batch[0][6],
		                                                         data_batch[0][7], data_batch[0][8], data_batch[0][9], data_batch[0][2],
		                                                         data_batch[0][5], input_size, max_nnb, E_scaling)

			N_data_forces += N_data1_F
			N_data_energy += N_data1_E

		if valid_data.dataset.train_energy:
			l2_E, N_data2_E = net.get_loss_RMSE(data_batch[0][10], data_batch[0][11], data_batch[0][12], data_batch[0][14])

			N_data_energy += N_data2_E

		N_data_E_tot = N_data1_E + N_data2_E
		net.optimizer.zero_grad()

		E_error += l1_E.item() + l2_E.item()
		F_error += l1_F.item()**2*N_data1_F
	
	E_error = np.sqrt(E_error/N_data_energy)
	if N_data_forces != 0:
		F_error = np.sqrt(F_error/N_data_forces)
	else:
		F_error = 0.0
	valid_error = (1.0-net.alpha.item())*E_error + net.alpha.item()*F_error

	return valid_error, E_error, F_error




def save_energies_any(net, any_data, E_scaling, E_shift, dataset="train"):
	"""
	Compute and save the energies of all the structures in the training/testing sets
	"""

	with open("energies."+dataset, "w") as f:
		ind = 1
		for i_batch, data_batch in enumerate(any_data):
			"""
			data_batch[0]  :: group_descrp
			data_batch[1]  :: group_energy
			data_batch[2]  :: logic_tensor_reduce
			data_batch[3]  :: index_from_database
			data_batch[4]  :: group_N_atom
			data_batch[5]  :: group_forces
			data_batch[6]  :: group_sfderiv_i
			data_batch[7]  :: group_sfderiv_j
			data_batch[8]  :: group_indices_F
			data_batch[9]  :: group_indices_F_i
			"""

			with torch.no_grad():
				if any_data.dataset.train_forces:
					list_E_ann = net.forward(data_batch[0][0], data_batch[0][2]  )
					list_E_ann = list_E_ann/E_scaling + E_shift*data_batch[0][4]
					group_energy = data_batch[0][1]/E_scaling + E_shift*data_batch[0][4]

					for i_struc in range(len(group_energy)):
						N_atoms = data_batch[0][4][i_struc]
						name    = data_batch[0][3][i_struc][0]
						E       = data_batch[0][3][i_struc][1]

						E_ann   = list_E_ann[i_struc].item()
						E_ref   = group_energy[i_struc].item()

						fmt = "{:12d}    {: 12.6f}  {: 12.6f}  {: 12.6f}  {: 12.6f}      {:}\n"
						f.write( fmt.format(int(N_atoms), E_ref, E_ann, E_ref/N_atoms, E_ann/N_atoms, name ))

				# If energy
				if any_data.dataset.train_energy:
					list_E_ann = net.forward(data_batch[0][10], data_batch[0][12]  )
					list_E_ann = list_E_ann/E_scaling + E_shift*data_batch[0][14]
					group_energy = data_batch[0][11]/E_scaling + E_shift*data_batch[0][14]

					for i_struc in range(len(group_energy)):
						N_atoms = data_batch[0][14][i_struc]
						name    = data_batch[0][13][i_struc][0]
						E       = data_batch[0][13][i_struc][1]

						E_ann = list_E_ann[i_struc].item()
						E_ref = group_energy[i_struc].item()

						fmt = "{:12d}    {: 12.6f}  {: 12.6f}  {: 12.6f}  {: 12.6f}      {:}\n"
						f.write( fmt.format(int(N_atoms), E_ref, E_ann, E_ref/N_atoms, E_ann/N_atoms, name ))



def save_forces_any(net, any_data, E_scaling, input_size, max_nnb, dataset="train"):
	"""
	Compute and save the forces of all the atoms of each structure in the training/testing sets
	"""
	with open("forces."+dataset, "w") as f:
		ind = 1
		for i_batch, data_batch in enumerate(any_data):
			"""
			data_batch[0]  :: group_descrp
			data_batch[1]  :: group_energy
			data_batch[2]  :: logic_tensor_reduce
			data_batch[3]  :: index_from_database
			data_batch[4]  :: group_N_atom
			data_batch[5]  :: group_forces
			data_batch[6]  :: group_sfderiv_i
			data_batch[7]  :: group_sfderiv_j
			data_batch[8]  :: group_indices_F
			data_batch[9]  :: group_indices_F_i
			"""

			# If forces
			if any_data.dataset.train_forces:
				list_E_ann, list_F_ann = net.forward_F( data_batch[0][0], data_batch[0][6], data_batch[0][7],
				                            			data_batch[0][8], data_batch[0][9], data_batch[0][2],
				                            			input_size, max_nnb)

				ind1 = 0
				for i_struc in range(len(list_E_ann)):
					N_atoms = int(data_batch[0][4][i_struc])
					name    = data_batch[0][3][i_struc][0]

					ind2 = ind1 + N_atoms

					F_struc_ann = list_F_ann[ind1:ind2]/E_scaling
					F_struc_ref = data_batch[0][5][ind1:ind2]/E_scaling

					diff_F = F_struc_ann - F_struc_ref
					l_F  = torch.sqrt( torch.sum( diff_F**2) ) / N_atoms

					f.write("# {:12d}  {: 12.6f}   {:}\n".format(int(N_atoms), l_F, name))

					ind1 = ind2

					for i_atom in range(N_atoms):
						diff_F = F_struc_ann[i_atom] - F_struc_ref[i_atom]
						l_F = torch.sqrt( torch.sum( diff_F**2) )
						fmt = "{:4d}   {: 12.6f}   {: 12.6f} {: 12.6f} {: 12.6f}   {: 12.6f} {: 12.6f} {: 12.6f} \n"
						
						f.write( fmt.format(i_atom, l_F, *F_struc_ref[i_atom], *F_struc_ann[i_atom] ))
