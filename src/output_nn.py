import torch
import numpy as np

from read_trainset import *

def save_network_iesp(model, iesp, f):
	"""
	Save the ANN parameters of one atomic specie in the '*.nn.ascii' format. Same as aenet.
	"""
	species = model.species

	nnodes = [ model.input_size[iesp] ]
	nnodes += model.hidden_size[iesp]
	nnodes.append(1)
	nnodes = np.array(nnodes)

	nlayers = len(nnodes)

	nnodesmax = np.max( nnodes )

	iw = np.zeros(nlayers, dtype=int)
	iv = np.zeros(nlayers, dtype=int)
	wsize = 0
	nvalues = 0

	for ilayer in range(0, nlayers-1):
		wsize += (nnodes[ilayer] + 1)*nnodes[ilayer+1]
		iw[ilayer+1] = wsize

		nvalues += nnodes[ilayer] + 1
		iv[ilayer+1] = nvalues
	nvalues += nnodes[-1]

	fun = [1 for i in range(len(model.hidden_size[iesp]))]
	fun.append(0)

	W = []
	kk = 0

	cont = 0
	for ifun in range(nlayers-1):

		weight = model.functions[iesp][2*ifun].weight
		bias   = model.functions[iesp][2*ifun].bias

		weight = weight.cpu().numpy()
		bias   = bias.cpu().numpy()

		nnodes1 = nnodes[ifun]
		nnodes2 = nnodes[ifun+1]

		bias = np.reshape(bias, (nnodes2,1))

		aux = np.concatenate( (weight,bias), axis=1 )
		aux = np.reshape( aux, ((nnodes1+1)*nnodes2), order="F" )
		W = W + list(aux)

	W = np.array(W)

	f.write("{:17d}\n".format(nlayers))
	f.write("{:17d}\n".format(nnodesmax))
	f.write("{:17d}\n".format(wsize))
	f.write("{:17d}\n".format(nvalues))
	fmt = "{:17d} "*nlayers+"\n"
	f.write(fmt.format(*nnodes))
	fmt = "{:17d} "*(nlayers-1)+"\n"
	f.write(fmt.format(*fun))
	fmt = "{:17d} "*nlayers+"\n"
	f.write(fmt.format(*iw))
	fmt = "{:17d} "*nlayers+"\n"
	f.write( fmt.format(*iv))
	fmt = "{:24.17f} "*wsize+"\n"
	f.write(fmt.format(*W))


def save_setup_iesp(setup_params, iesp, f):
	"""
	Save the Descripto setup parameters of one atomic specie in the '*.nn.ascii' format. Same as aenet.
	"""
	nenv = setup_params.nenv[iesp]
	nsf = setup_params.nsf[iesp]
	nsfparam = setup_params.nsfparam[iesp]

	sfparam = np.reshape(setup_params.sfparam[iesp], (nsfparam*nsf))

	sfenv = np.reshape(setup_params.sfenv[iesp], (2*nsf))

	f.write("{:}\n".format(setup_params.description[iesp]))
	f.write("{:}\n".format(setup_params.atomtype[iesp]))
	f.write("{:17d}\n".format(setup_params.nenv[iesp]))
	fmt = "{:} "*nenv+"\n"
	f.write(fmt.format(*setup_params.envtypes[iesp]))
	f.write("{:24.17f}\n".format(setup_params.rcmin[iesp]))
	f.write("{:24.17f}\n".format(setup_params.rcmax[iesp]))
	f.write("{:}\n".format(setup_params.sftype[iesp]))
	f.write("{:17d}\n".format(setup_params.nsf[iesp]))
	f.write("{:17d}\n".format(setup_params.nsfparam[iesp]))
	fmt = "{:17d} "*nsf+"\n"
	f.write(fmt.format(*setup_params.sf[iesp]))
	fmt = "{:24.17f} "*nsfparam*nsf+"\n"
	f.write(fmt.format(*sfparam))
	fmt = "{:17d} "*2*nsf+"\n"
	f.write(fmt.format(*sfenv))
	f.write("{:17d}\n".format(setup_params.neval[iesp]))
	fmt = "{:24.17f} "*nsf+"\n"
	f.write(fmt.format(*setup_params.sfval_min[iesp]))
	f.write(fmt.format(*setup_params.sfval_max[iesp]))
	f.write(fmt.format(*setup_params.sfval_avg[iesp]))
	f.write(fmt.format(*setup_params.sfval_cov[iesp]))


def save_trainset_info(trainset_params, f):
	"""
	Save traingset information of one atomic specie in the '*.nn.ascii' format. Same as aenet.
	"""
	f.write("{:}\n".format(trainset_params.filename))
	f.write("{:}\n".format(trainset_params.normalized))
	f.write("{:}\n".format(trainset_params.E_scaling))
	f.write("{:}\n".format(trainset_params.E_shift))
	f.write("{:}\n".format(trainset_params.N_species))
	fmt = "{:} "*trainset_params.N_species+"\n"
	f.write(fmt.format(*trainset_params.sys_species))
	fmt = "{:} "*trainset_params.N_species+"\n"
	f.write(fmt.format(*trainset_params.E_atomic))
	f.write("{:}\n".format(trainset_params.N_atom))
	f.write("{:}\n".format(trainset_params.N_struc))
	f.write("{:} {:} {:}\n".format(trainset_params.E_min, trainset_params.E_max, trainset_params.E_avg))


def save_results(tin, model):
	"""
	Save the ANN to '*.nn.ascii' format
	"""
	model_restart = {"model" : model.state_dict(),  "optimizer" : model.optimizer.state_dict()}
	if tin.mode == "train":
		torch.save(model_restart, "./model.restart")
		with torch.no_grad():
			for iesp in range( tin.N_species ):
				with open(tin.networks_param["names"][iesp]+".ascii", "w") as f:
					save_network_iesp(model,iesp, f)
					save_setup_iesp(tin.setup_params, iesp, f)
					save_trainset_info(tin.trainset_params, f)

	elif tin.mode == "transfer":
		torch.save(model_restart, "./model.transfered.restart")
		with torch.no_grad():
			for iesp in range( tin.N_species ):
				with open(tin.networks_param["names"][iesp]+".transfered.ascii", "w") as f:
					save_network_iesp(model,iesp, f)
					save_setup_iesp(tin.setup_params, iesp, f)
					save_trainset_info(tin.trainset_params, f)



def read_network_iesp(model, iesp, f):
	"""
	Read the ANN parameters of one atomic specie in the '*.nn.ascii' format. Same as aenet.
	"""
	nlayers   = int( f.readline() )
	nnodesmax = int( f.readline() )
	wsize     = int( f.readline() )
	nvalues   = int( f.readline() )
	nnodes    = np.array([ int(x) for x in f.readline().split() ])
	fun       = np.array([ int(x) for x in f.readline().split() ])
	iw        = np.array([ int(x) for x in f.readline().split() ])
	iv        = np.array([ int(x) for x in f.readline().split() ])
	W         = np.array([ float(x) for x in f.readline().split() ])

	list_weight = []
	list_bias   = []

	input_size  = [0]
	hidden_size = nnodes[1:-1]

	iw1 = 0
	nnodes1 = nnodes[0]
	for ilayer in range(nlayers-1):

		iw2 = iw[ilayer+1] #- 1
		nnodes2 = nnodes[ilayer+1]

		Wshape = (nnodes2, nnodes1+1)

		work = np.reshape( W[iw1:iw2], Wshape, order="F" )

		weight = work[:,:-1]
		bias = work[:,-1]

		iw1 = iw2 
		nnodes1 = nnodes2


def read_setup_iesp(setup_params, iesp, f):
	"""
	Read the Descripto setup parameters of one atomic specie in the '*.nn.ascii' format. Same as aenet.
	"""
	description = f.readline().strip()
	atomtype    = f.readline().strip()
	nenv        = int( f.readline() )
	envtypes    = [ x for x in f.readline().split() ]
	rcmin       = float( f.readline() )
	rcmax       = float( f.readline() )
	sftype      = f.readline().strip()
	nsf         = int( f.readline() )
	nsfparam    = int( f.readline() )
	sf          = [ int(x) for x in f.readline().split() ]
	sfparam     = [ float(x) for x in f.readline().split() ]
	sfenv       = [ int(x) for x in f.readline().split() ]
	neval       = int( f.readline() )
	sfval_min   = [ float(x) for x in f.readline().split() ]
	sfval_max   = [ float(x) for x in f.readline().split() ]
	sfval_avg   = [ float(x) for x in f.readline().split() ]
	sfval_cov   = [ float(x) for x in f.readline().split() ]

	return neval, sfval_min, sfval_max, sfval_avg, sfval_cov


def read_trainset_info(trainset_params, f):
	"""
	Save training set information in the '*.nn.ascii' format. Same as aenet.
	"""
	filename            = f.readline().strip()
	normalized          = f.readline().strip()
	E_scaling           = float( f.readline() )
	E_shift             = float( f.readline() )
	N_species           = int( f.readline() )
	sys_species         = [ x for x in f.readline().split() ]
	E_atomic            = [ float(x) for x in f.readline().split() ]
	N_atom              = int( f.readline() )
	N_struc             = int( f.readline() )
	E_min, E_max, E_avg = [ float(x) for x in f.readline().split() ]

	return E_scaling, E_shift, E_min, E_max, E_avg