"""
Read 'train.in' input file
--------------------------

read_train_in(infile) :: Returns an object of the class InputParameters()
"""

from data_classes import *


def read_keyword_logical(keyword, lines):
	value = None
	found = False
	for line in lines:
		read_keyword = line.split()[0].lower()

		if read_keyword == keyword:
			found = True
			value = True
			break
	return value, found


def read_keyword_argument_same_line(keyword, lines):
	value = None
	found = False
	for iline in range(len(lines)):
		read_keyword = lines[iline].split()[0].lower()

		if read_keyword == keyword:
			found = True
			value = lines[iline].split()[1]
	return value, found


def read_keyword_argument_next_line(keyword, lines):
	value = None
	found = False
	params = {}
	for iline in range(len(lines)):
		read_keyword = lines[iline].split()[0].lower()

		if read_keyword == keyword:
			found = True

			for param in lines[iline+1].split():
				p, val = param.split("=")[0], param.split("=")[1]
				params[p.lower()] = val
			break
	return params, found


def read_keyword_networks(lines):
	value = None
	found = False
	for iline in range(len(lines)):
		read_keyword = lines[iline].split()[0].lower()

		if read_keyword == "networks":
			found = True
			networks_line = iline+1

	sys_species = []
	for line in lines[networks_line:]:
		sys_species.append( line.split()[0] )
	N_species = len(sys_species)

	# Networks
	input_size  = [ 0  for i in range(N_species)]
	hidden_size = [ [] for i in range(N_species)]
	activations = [ [] for i in range(N_species)]
	names       = [ "" for i in range(N_species)]

	for line in lines[networks_line:networks_line+N_species]:
		aux = line.split()
		specie = aux[0]
		name_i = aux[1]

		N_hidden = int(aux[2])
		hidden_size_i = []
		activations_i = []
		for i in aux[3:]:
			hidden_size_i.append( int(i.split(":")[0]) )
			activations_i.append( i.split(":")[1].lower() )	

		specie_index = sys_species.index(specie)

		hidden_size[specie_index] = hidden_size_i
		activations[specie_index] = activations_i
		names[specie_index]       = name_i


	networks_param = {}
	networks_param["hidden_size"] = hidden_size
	networks_param["activations"] = activations
	networks_param["names"]       = names
	networks_param["input_size"]  = input_size
	return sys_species, networks_param



def read_train_in(infile):
	with open(infile, "r") as f:

		# Initiialize InputParameters with default values
		tin = InputParameters()

		# Remove comments from input file:
		lines = f.readlines()
		list_comments = []
		for i in range(len(lines)-1, -1, -1):
			if lines[i][0] in ["!", "#"] or len(lines[i].split()) == 0:
				list_comments.append(i)

		for i in list_comments:
			lines.pop(i)


		# Compulsory parameters:
		tin.train_file   ,_ = read_keyword_argument_same_line("trainingset", lines)
		tin.test_split   ,_ = read_keyword_argument_same_line("testpercent", lines)
		tin.epoch_size   ,_ = read_keyword_argument_same_line("iterations", lines)
		tin.method_param ,_ = read_keyword_argument_next_line("method", lines)
		tin.sys_species, tin.networks_param = read_keyword_networks(lines)

		# Optional parameters:
		pytorch_seed      , found = read_keyword_argument_same_line("phseed", lines)
		if found: tin.pytorch_seed = int(pytorch_seed)

		numpy_seed      , found = read_keyword_argument_same_line("npseed", lines)
		if found: tin.numpy_seed = int(numpy_seed)

		epoch_write      , found = read_keyword_argument_same_line("iterwrite", lines)
		if found: tin.epoch_write = int(epoch_write)

		batch_size       , found = read_keyword_argument_same_line("batchsize", lines)
		if found: tin.batch_size = int(batch_size)

		max_energy       , found = read_keyword_argument_same_line("maxenergy", lines)
		if found: tin.max_energy = float(max_energy)

		max_forces       , found = read_keyword_argument_same_line("maxforces", lines)
		if found: tin.max_forces = float(max_forces)

		mode             , found = read_keyword_argument_same_line("mode", lines)
		if found: tin.mode = mode

		forces_param     , found = read_keyword_argument_next_line("forces", lines)
		if found: tin.train_forces = True
		if found: tin.forces_param = forces_param

		memory_mode      , found = read_keyword_argument_same_line("memory_mode", lines)
		if found: tin.memory_mode = memory_mode

		save_energies    , found = read_keyword_logical("save_energies", lines)
		if found: tin.save_energies = save_energies

		save_forces      , found = read_keyword_logical("save_forces", lines)
		if found: tin.save_forces = save_forces

		verbose          , found = read_keyword_logical("verbose", lines)
		if found: tin.verbose = verbose

		save_batches          , found = read_keyword_logical("save_batches", lines)
		if found: tin.save_batches = save_batches

		load_batches          , found = read_keyword_logical("load_batches", lines)
		if found: tin.load_batches = load_batches

		regularization   , found = read_keyword_argument_same_line("regularization", lines)
		if found: tin.regularization = float(regularization)

		tin.initialize()

		return tin
