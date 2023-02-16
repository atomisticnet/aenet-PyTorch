from read_trainset import *
import torch
import os


def init_optimizer(tin, model):
	if tin.method == "adam":
		model.optimizer = torch.optim.Adam(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adadelta":
		model.optimizer = torch.optim.Adadelta(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adagrad":
		model.optimizer = torch.optim.Adagrad(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adamw":
		model.optimizer = torch.optim.AdamW(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)

	if tin.method == "adamax":
		model.optimizer = torch.optim.Adamax(model.parameters(), lr = tin.lr, weight_decay=tin.regularization)



def init_train(tin, model):
	# If RESTART, read previous model and optimizer state_dict
	if os.path.exists("./model.restart"):
		#print("Restarting...")
		model_restart = torch.load("./model.restart")
		model.load_state_dict( model_restart["model"] )
		model.optimizer.load_state_dict( model_restart["optimizer"] )

		model.eval()

		model.alpha = torch.tensor(tin.alpha, device=model.device)

	#else:
		#print("Starting from scratch...")




def init_transfer(tin, trainset_params, model):
	E_min, E_max, E_avg          = trainset_params.E_min, trainset_params.E_max, trainset_params.E_avg
	E_scaling, E_shift, E_atomic = trainset_params.E_scaling, trainset_params.E_shift, trainset_params.E_atomic

	# If TRANSFER, read previous model parameters, and SFT/energy normalization parameters
	if os.path.exists("./model.transfered.restart"):

		#print("Restarting...")
		model_restart = torch.load("./model.transfered.restart")
		model.load_state_dict( model_restart["model"] )
		model.optimizer.load_state_dict( model_restart["optimizer"] )

		model.eval()

		model.alpha = torch.tensor(tin.alpha, device=model.device)

		# Read normalization parameters from *.nn.ascii
		for iesp in range( tin.N_species ):
			with open(tin.networks_param["names"][iesp]+".transfered.ascii", "r") as f:
				read_network_iesp(model, iesp, f)
				neval_i, sfval_min_i, sfval_max_i, sfval_avg_i, sfval_cov_i = read_setup_iesp(tin.sf_setups_param, iesp, f)
				E_scaling, E_shift, E_min, E_max, E_avg = read_trainset_info(trainset_params, f)

				trainset_params.normalized = True
				trainset_params.E_scaling = E_scaling
				trainset_params.E_shift = E_shift
				trainset_params.E_avg = E_avg
				trainset_params.E_max = E_max
				trainset_params.E_min = E_min

				tin.sf_setups_param[iesp].neval     = neval_i
				tin.sf_setups_param[iesp].sfval_min = sfval_min_i
				tin.sf_setups_param[iesp].sfval_max = sfval_max_i
				tin.sf_setups_param[iesp].sfval_avg = sfval_avg_i
				tin.sf_setups_param[iesp].sfval_cov = sfval_cov_i

	else:
		print("Reading for transfer...")
		model.load_state_dict(torch.load("./model", map_location=device))
		model.eval()

		# Compute energy normalization parameters
		E_scaling, E_shift, E_min, E_max, E_avg = get_E_norm_parameters( E_min, E_max, E_avg )


		# Read STP normalization parameters from *.nn.ascii
		for iesp in range( tin.N_species ):
			with open(tin.networks_param["names"][iesp]+".ascii", "r") as f:
				read_network_iesp(model, iesp, f)
				neval_i, sfval_min_i, sfval_max_i, sfval_avg_i, sfval_cov_i = read_setup_iesp(tin.sf_setups_param, iesp, f)
				E_scaling, E_shift, E_min, E_max, E_avg = read_trainset_info(trainset_params, f)

				trainset_params.normalized = True
				trainset_params.E_scaling = E_scaling
				trainset_params.E_shift = E_shift
				trainset_params.E_avg = E_avg
				trainset_params.E_max = E_max
				trainset_params.E_min = E_min

				tin.sf_setups_param[iesp].neval     = neval_i
				tin.sf_setups_param[iesp].sfval_min = sfval_min_i
				tin.sf_setups_param[iesp].sfval_max = sfval_max_i
				tin.sf_setups_param[iesp].sfval_avg = sfval_avg_i
				tin.sf_setups_param[iesp].sfval_cov = sfval_cov_i



	# Freeze layer
	for iesp in range(tin.N_species):
		for ifun in  [0]:#range(1):
			weight = model.functions[iesp][2*ifun].weight.requires_grad = False
			bias   = model.functions[iesp][2*ifun].bias.requires_grad = False



	for i in model.functions:
		print(i)

	for n, p in model.named_parameters():
		print("{:40} {:} {:}".format(n, p.requires_grad, tuple(p.shape)))
