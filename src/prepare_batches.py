from data_set import *
from data_loader import *
from read_trainset import *


def read_list_structures(tin):
	"""
	Read Training set files (*.train.ascii and *.train.forces)
	"""
	if tin.train_forces:
		list_structures_forces, list_structures_energy, list_removed, max_nnb, tin = read_train_forces_together(tin)
	else:
		list_structures_energy, list_removed, max_nnb, tin = read_train(tin)
		list_structures_forces = []
		max_nnb = 0

	input_size = tin.networks_param["input_size"]
	for struc in list_structures_energy:
		struc.padding(max_nnb, input_size)
	for struc in list_structures_forces:
		struc.padding(max_nnb, input_size)

	return list_structures_energy, list_structures_forces, list_removed, max_nnb, tin


def get_N_batch(len_dataset, batch_size):
	"""
	Returns the number of batches for a given batch size and dataset size
	"""
	N_batch = int(len_dataset/batch_size)
	residue = len_dataset - N_batch*batch_size

	if residue >= int(batch_size/2) or N_batch == 0:
		if residue != 0:
			N_batch += 1

	return N_batch


def split_database(dataset_size, test_split):
	"""
	Returns indices of the structures in the training and testing sets
	"""
	indices = list(range(dataset_size))
	np.random.shuffle(indices)
	split   = int(np.floor(test_split * dataset_size))

	train_indices, test_indices = indices[split:], indices[:split]

	return train_indices, test_indices


def select_batch_size(tin, list_structures_energy, list_structures_forces):
	"""
	Select batch size that best matches the requested size, avoiding the last batch being too small
	"""
	N_data_E = len(list_structures_energy)
	N_data_F = len(list_structures_forces)
	train_sampler_E, valid_sampler_E = split_database(N_data_E, tin.test_split)
	train_sampler_F, valid_sampler_F = split_database(N_data_F, tin.test_split)

	N_data_train_E = len(train_sampler_E)
	N_data_valid_E = len(valid_sampler_E)
	N_data_train_F = len(train_sampler_F)
	N_data_valid_F = len(valid_sampler_F)

	forcespercent  = N_data_F/(N_data_F + N_data_E)

	if forcespercent <= 0.5:
		tin.batch_size = round((1 - forcespercent)*tin.batch_size)

		N_batch_train = get_N_batch(N_data_train_E, tin.batch_size)
		N_batch_valid = get_N_batch(N_data_valid_E, tin.batch_size)
	else:
		tin.batch_size = forcespercent*tin.batch_size

		N_batch_train = get_N_batch(N_data_train_F, tin.batch_size)
		N_batch_valid = get_N_batch(N_data_valid_F, tin.batch_size)

	return N_batch_train, N_batch_valid


def select_batches(tin, trainset_params, device, list_structures_energy, list_structures_forces,
				   max_nnb, N_batch_train, N_batch_valid):
	"""
	Select which structures belong to each batch for training.
	Returns: four objects of the class data_set_loader.PrepDataloader(), for train/test and energy/forces
	"""
	if len(list_structures_energy) != 0:
		dataset_energy = StructureDataset(list_structures_energy, tin.sys_species, tin.networks_param["input_size"], max_nnb)
		dataset_energy_size = len(dataset_energy)

		# Normalize
		E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
		sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
		dataset_energy.normalize_E(trainset_params.E_scaling, trainset_params.E_shift)
		stp_shift, stp_scale = dataset_energy.normalize_stp(sfval_avg, sfval_cov)

		# Split in train/test
		train_sampler_E, valid_sampler_E = split_database(dataset_energy_size, tin.test_split)

		train_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_train,
		                               sampler=train_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="train_energy")
		valid_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_valid,
		                               sampler=valid_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="valid_energy")

	else:
		dataset_energy = None
		train_energy_data, valid_energy_data = None, None


	if len(list_structures_forces) != 0:
		dataset_forces = StructureDataset(list_structures_forces, tin.sys_species, tin.networks_param["input_size"], max_nnb)
		dataset_forces_size = len(dataset_forces)

		# Normalize
		E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
		sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
		dataset_forces.normalize_E(trainset_params.E_scaling, trainset_params.E_shift)
		dataset_forces.normalize_F(trainset_params.E_scaling, trainset_params.E_shift)
		stp_shift, stp_scale = dataset_forces.normalize_stp(sfval_avg, sfval_cov)

		# Split in train/test
		train_sampler_F, valid_sampler_F = split_database(dataset_forces_size, tin.test_split)

		train_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=True, N_batch=N_batch_train,
		                               sampler=train_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="train_forces")
		valid_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=True, N_batch=N_batch_valid,
		                               sampler=valid_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="valid_forces")

	else:
		dataset_forces = None
		train_forces_data, valid_forces_data = None, None

	return train_forces_data, valid_forces_data, train_energy_data, valid_energy_data


def save_datsets(save, train_forces_data, valid_forces_data, train_energy_data, valid_energy_data):
	"""
	Saves datasets created by select_batches
	"""
	torch.save(save, "tmp_batches/trainset_info")
	torch.save(train_forces_data, "tmp_batches/train_forces_data.ph")
	torch.save(valid_forces_data, "tmp_batches/valid_forces_data.ph")
	torch.save(train_energy_data, "tmp_batches/train_energy_data.ph")
	torch.save(valid_energy_data, "tmp_batches/valid_energy_data.ph")
	#torch.save(grouped_train_data, "tmp_batches/grouped_train_data.ph")


def load_datasets():
	"""
	Loads saved datasets instead of preparing them
	"""
	save = torch.load("tmp_batches/trainset_info")
	N_removed, N_struc_E, N_struc_F, max_nnb, tin.trainset_params, tin.setup_params, tin.networks_param = save[:]

	train_forces_data = torch.load("tmp_batches/train_forces_data.ph")
	train_energy_data = torch.load("tmp_batches/train_energy_data.ph")
	train_forces_data.gather_data(tin.memory_mode)
	train_energy_data.gather_data(tin.memory_mode)

	grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
									 	memory_mode=tin.memory_mode, device=device, dataname="train")
	del train_forces_data
	del train_energy_data

	valid_forces_data = torch.load("tmp_batches/valid_forces_data.ph")
	valid_energy_data = torch.load("tmp_batches/valid_energy_data.ph")
	valid_forces_data.gather_data(tin.memory_mode)
	valid_energy_data.gather_data(tin.memory_mode)

	grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
										memory_mode=tin.memory_mode, device=device, dataname="valid")

	del valid_forces_data
	del valid_energy_data

	return N_removed, N_struc_E, N_struc_F, max_nnb, grouped_train_data, grouped_valid_data