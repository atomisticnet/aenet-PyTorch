"""
Main program of aenet-PyTorch. The code is divided in the following scheme:
	1. Read train.in input file
	2. Prepare batches for training
	3. Initialize model and optimizer (either scratch or restart)
	4. Initialize data loader
	5. Train
	6. Save results
"""

import numpy as np
import torch
import time
import sys
import resource
from torch.utils.data import DataLoader

from data_classes import *
from read_input import *
from read_trainset import *
from network import *
from prepare_batches import *
from traininit import *
from data_set import *
from data_loader import *
from optimization_step import *
from output_nn import *
from py_aeio import *



device = "cpu"
if torch.cuda.is_available(): device = "cuda:0"

io_print_header()


# 1. Read train input
if len(sys.argv) > 1:
	tin_file = sys.argv[1]
else:
	tin_file = "train.in"
tin = read_train_in(tin_file)
tin.device = device
torch.manual_seed(tin.pytorch_seed)
np.random.seed(tin.numpy_seed)
if tin.verbose: io_input_reading(tin)


# 2. Get batches
if tin.load_batches:

	if tin.verbose : io_load_batches()

	save = torch.load("tmp_batches/trainset_info")
	train_forces_data = torch.load("tmp_batches/train_forces_data.ph")
	valid_forces_data = torch.load("tmp_batches/valid_forces_data.ph")
	train_energy_data = torch.load("tmp_batches/train_energy_data.ph")
	valid_energy_data = torch.load("tmp_batches/valid_energy_data.ph")

	if train_forces_data != None: train_forces_data.gather_data(tin.memory_mode)
	if valid_forces_data != None: valid_forces_data.gather_data(tin.memory_mode)
	if train_energy_data != None: train_energy_data.gather_data(tin.memory_mode)
	if valid_energy_data != None: valid_energy_data.gather_data(tin.memory_mode)

	N_removed, N_struc_E, N_struc_F, max_nnb, tin.trainset_params, tin.setup_params, tin.networks_param = save[:]

	if tin.verbose : io_trainingset_information_done(tin, tin.trainset_params, N_struc_E, N_struc_F, N_removed)


else:
	# Read datasets
	if tin.verbose : io_trainingset_information()
	list_structures_energy, list_structures_forces, list_removed, max_nnb, tin = read_list_structures(tin)

	N_removed = len(list_removed)
	N_struc_E = len(list_structures_energy)
	N_struc_F = len(list_structures_forces)
	if tin.verbose : io_trainingset_information_done(tin, tin.trainset_params, N_struc_E, N_struc_F, N_removed)

	if tin.verbose : io_prepare_batches()

	N_batch_train, N_batch_valid = select_batch_size(tin, list_structures_energy, list_structures_forces)

	# Join datasets with forces and only energies in a single torch dataset AND prepare batches
	train_forces_data, valid_forces_data, train_energy_data, valid_energy_data = select_batches(tin, tin.trainset_params, device, list_structures_energy, list_structures_forces,
																							max_nnb, N_batch_train, N_batch_valid)

	del list_structures_energy
	del list_structures_forces

	if tin.verbose : io_prepare_batches_done(tin, train_energy_data, train_forces_data)


if tin.save_batches:

	if tin.verbose : io_save_batches()
	if not os.path.exists("tmp_batches"): os.makedirs("tmp_batches")
	save = [N_removed, N_struc_E, N_struc_F, max_nnb, tin.trainset_params, tin.setup_params, tin.networks_param]		
	save_datsets(save, train_forces_data, valid_forces_data, train_energy_data, valid_energy_data)

if tin.mode == "batches":
	io_footer()
	sys.exit()


grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
									 memory_mode=tin.memory_mode, device=device, dataname="train")
grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
									 memory_mode=tin.memory_mode, device=device, dataname="valid")


del train_forces_data
del valid_forces_data
del train_energy_data
del valid_energy_data

# 3. Initialize model and optimizer
if tin.verbose: io_network_initialize(tin)

model = NetAtom(tin.networks_param["input_size"], tin.networks_param["hidden_size"],
			    tin.sys_species, tin.networks_param["activations"], tin.alpha, device).double()
model.to(device)
init_optimizer(tin, model)
if tin.mode == "train":
	init_train(tin, model)
elif tin.mode == "transfer":
	init_transfer(tin, tin.trainset_params, model)
sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
E_scaling, E_shift   = tin.trainset_params.get_E_normalization()


# 4. Initialize dataloader
grouped_train_loader = DataLoader(grouped_train_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0)
grouped_valid_loader = DataLoader(grouped_valid_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0)


# 5. Train
if tin.verbose: io_train_details(tin, device)
if tin.verbose: io_train_start()
t = time.time()
iter_error_trn = []
iter_error_tst = []
for epoch in range(tin.epoch_size):

	train_error, train_E_error, train_F_error = step_train_any(model, grouped_train_loader, E_scaling, tin.networks_param["input_size"], max_nnb)
	#iter_error_trn.append([epoch, train_error, train_E_error, train_F_error])

	if epoch%tin.epoch_write == 0:
		valid_error, valid_E_error, valid_F_error = step_valid_any(model, grouped_valid_loader, E_scaling, tin.networks_param["input_size"], max_nnb)
		io_train_step(epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error, E_scaling)
		iter_error_tst.append([epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error])

io_train_finalize(t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,  torch.cuda.max_memory_allocated()/1024**3)


if tin.save_energies:
	save_energies_any(model, grouped_train_loader, E_scaling, E_shift, dataset="train")
	save_energies_any(model, grouped_valid_loader, E_scaling, E_shift, dataset="test")

if tin.train_forces and tin.save_forces:
	save_forces_any(model, grouped_train_loader, E_scaling, tin.networks_param["input_size"], max_nnb, dataset="train")
	save_forces_any(model, grouped_valid_loader, E_scaling, tin.networks_param["input_size"], max_nnb, dataset="test")


# 6. Save networks
if tin.verbose: io_save_networks(tin)
save_results(tin, model)


# 7. Save error per iteration in separate unformatted file
io_save_error_iteration(tin, E_scaling, iter_error_trn, iter_error_tst)


io_footer()