import datetime
import time
import numpy as np
import os



def io_print(text):
	print(text, flush=True)



def io_print_center(text):
	lenght  = 70
	N_blank = lenght - len(text)

	aux = divmod(N_blank, 2)
	a = aux[0]
	b = aux[0]
	if aux[1] != 0: a += 1

	io_print(" "*a+text+" "*b)


def io_current_time():
	aux = str(datetime.datetime.now())[:-6]
	io_print_center(aux)


def io_line():
	io_print("----------------------------------------------------------------------")


def io_print_title(text):
	io_line()
	io_print_center(text)
	io_line()
	io_print("")


def io_double_line():
	io_print("======================================================================")


def io_print_header():
	io_double_line()
	io_print_center("Training with aenet-PyTorch")
	io_double_line()
	io_print("")
	io_print("")
	io_current_time()
	io_print("")
	io_print("")
	io_print("Developed by Jon Lopez-Zorrilla")
	io_print("")
	io_print("This program is distributed in the hope that it will be useful,")
	io_print("but WITHOUT ANY WARRANTY; without even the implied warranty of")
	io_print("MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the")
	io_print("GNU General Public License in file 'LICENSE' for more details.")
	io_print("")


def io_input_reading(tin):
	io_print_title("Reading input information")

	io_print("Reading input parameters.")
	io_print("These are the parameters selected for training:")
	io_print("        - TRAININGSET: "+str(tin.train_file))
	io_print("        - TESTPERCENT: "+str(tin.test_split))
	io_print("        - ITERATIONS:  "+str(tin.epoch_size))
	io_print("        - ITERWRITE:   "+str(tin.epoch_write))
	io_print("        - BATCHSIZE:   "+str(tin.batch_size))
	io_print("        - MEMORY_MODE: "+str(tin.memory_mode))
	io_print("")

	if tin.train_forces:
		io_print("        - FORCES:      "+str(tin.train_forces))
		io_print("        - alpha:       "+str(tin.alpha))
		if tin.max_forces: io_print("        - maxforces:   "+str(tin.max_forces))
		io_print("")


def io_network_initialize(tin):
	io_print_title("Networks")

	if os.path.exists("./model.restart"):
		io_print("Previous run files found. The training will be restarted from")
		io_print("that checkpoint.")
	else:
		io_print("Training will be started from scratch.")
	io_print("Initializing networks:")
	for iesp in range(tin.N_species):
		io_print("")
		io_print("Creating a network for "+str(tin.sys_species[iesp]))
		io_print("")
		io_print("Number of layers: {:4d}".format( len(tin.networks_param["hidden_size"][iesp])+2 ))
		io_print("")
		io_print("Number of nodes and activation type per layer:")
		io_print("")
		io_print("    1 : {:6d} ".format(tin.networks_param["input_size"][iesp]))
		ilayer = -1
		for ilayer in range(len(tin.networks_param["hidden_size"][iesp])):
			io_print("   {:2d} : {:6d}   {:}".format(ilayer+2,tin.networks_param["hidden_size"][iesp][ilayer], tin.networks_param["activations"][iesp][ilayer]))
		io_print("   {:2d} : {:6d} ".format(ilayer+3,1))
		io_print("")


def io_trainingset_information():
	io_print_title("Reading training set information")

	io_print("Training set information will be read now. If force training is")
	io_print("required this proccess may take some time.")
	io_print("")


def io_trainingset_information_done(tin, trainset_params, N_struc_E, N_struc_F=0, N_removed=0):
	io_print("The network output energy will be normalized to the interval [-1,1].")
	io_print("    Energy scaling factor:  f = {: 12.6f}".format(trainset_params.E_scaling))
	io_print("    Atomic energy shift  :  s = {: 12.6f}".format(trainset_params.E_shift))
	io_print("")
	if N_removed != 0:
		io_print(str(N_removed)+" high-energy structures will be removed from the training set.")
	io_print("")
	io_print("Number of structures in the data set:        {:12d}".format(N_struc_E+N_struc_F))
	if N_struc_F != 0:
		io_print("Number of structures with force information: {:12d}".format(N_struc_F))
	io_print("")

	fmt = "{:}  "*tin.N_species
	fmt = fmt.format(*tin.sys_species)[:-2]
	io_print("Atomic species in the training set: {:}".format(fmt))
	io_print("")
	io_print("Average energy (eV/atom) : {:12.6f}".format(trainset_params.E_avg))
	io_print("Minimum energy (eV/atom) : {:12.6f}".format(trainset_params.E_min))
	io_print("Maximum energy (eV/atom) : {:12.6f}".format(trainset_params.E_max))
	io_print("")


def io_prepare_batches():
	io_print_title("Preparing batches for training")

	io_print("Batches for training are being prepared now. If force training is")
	io_print("required, this may take some time.")
	io_print("")
	io_print("If the number of structures is not divisible by the batch size, the actual")
	io_print("batch size may be slightly changed.")
	io_print("")


def io_prepare_batches_done(tin, train_energy_data, train_forces_data):

	mean_batch_size_E = 0
	mean_batch_size_F = 0
	if train_energy_data != None:
		aux     = train_energy_data.indexes[:,1] - train_energy_data.indexes[:,0]
		N_batch = train_energy_data.N_batch

		mean_batch_size_E = round(np.mean(aux))

	if train_forces_data != None:
		aux     = train_forces_data.indexes[:,1] - train_forces_data.indexes[:,0]
		N_batch = train_forces_data.N_batch

		mean_batch_size_F = round(np.mean(aux))

	io_print("Requested batch size: "+str(tin.original_batch_size))
	io_print("Actual batch size   : "+str(mean_batch_size_F+mean_batch_size_E))
	io_print("Number of batches   : "+str(N_batch))
	io_print("")

	if train_energy_data != None and train_forces_data != None:
		io_print("Energy batch size   : "+str(mean_batch_size_E))
		io_print("Forces batch size   : "+str(mean_batch_size_F))
		io_print("")


def io_save_batches():

	io_print_title("Saving batch information")

	io_print("Information about the batches will be saved now. If in the next run")
	io_print("load_batches=True and the folder 'tmp_batches' is present, ")
	io_print("the information will be read from there, saving time.")
	io_print("")
	io_print("If the batch size is changed, please remove that folder, or the training")
	io_print("may fail or not work as expected.")
	io_print("")


def io_load_batches():

	io_print_title("Loading batch information")

	io_print("Loading batch information from 'tmp_batches'. If the batch size has")
	io_print("been changed, the program will not work as expected.")
	io_print("")




def io_train_details(tin, device):
	io_print_title("Training details")

	io_print("Training method : {:>12}".format(tin.method))
	io_print("Learning rate   : {: 12.6f}".format(tin.lr))
	io_print("Regularization  : {: 12.6f}".format(tin.regularization))
	io_print("")
	io_print("Training device : {:}".format(device))
	io_print("Memory mode     : {:}".format(tin.memory_mode))
	io_print("")



def io_train_start():
	io_print_title("Training process")

	fmt2 = "{:>10} :  {:>12}  {:>12}   |{:>12}  {:>12}   |{:>12}  {:>12}"

	io_print(fmt2.format( "epoch", "ERROR(train)", "ERROR(test)", "E (train)", "E (test)", "F (train)", "F (test)" ))
	fmt2 =    "     epoch :  ERROR(train)   ERROR(test)   |   E (train)      E (test)   |   F (train)      F (test)"
	io_print( "     -----    ------------   -----------        ---------      --------        ---------      --------" )


def io_train_step(epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error, E_scaling):

	fmt2 = "{: 10d} :  {: 12.6f}  {: 12.6f}   |{: 12.6f}  {: 12.6f}   |{: 12.6f}  {: 12.6f}"

	train_error = train_error / E_scaling
	valid_error = valid_error / E_scaling
	train_E_error = train_E_error / E_scaling
	valid_E_error = valid_E_error / E_scaling
	train_F_error = train_F_error / E_scaling
	valid_F_error = valid_F_error / E_scaling
	io_print(fmt2.format(epoch, train_error, valid_error, train_E_error, valid_E_error, train_F_error, valid_F_error ))



def io_train_finalize(t, mem_CPU, mem_GPU):

	io_print("")
	io_print("Time needed for training:  {: 18.6f} s".format(time.time() - t))
	io_print("Maximum CPU memory used:   {: 18.6f} GB".format(mem_CPU))
	io_print("Maximum GPU memory used:   {: 18.6f} GB".format(mem_GPU))
	io_print("")



def io_save_networks(tin):
	io_print_title("Storing results")

	io_print("saving train energy error to : energy.train")
	io_print("saving test energy error to  : energy.test")
	io_print("")

	for iesp in range(tin.N_species):
		io_print("Saving the {:} network to file : {:}".format(tin.sys_species[iesp], tin.networks_param["names"][iesp]+".ascii"))



def io_save_error_iteration(tin, E_scaling, iter_error_trn, iter_error_tst):

	iter_error_tst = np.array(iter_error_tst)
	with open("train.error", "w") as f:
		fmt1 = "{: 10d}   {: 12.6f}  {: 12.6f}   {: 12.6f}  {: 12.6f}   {: 12.6f}  {: 12.6f}\n"
		fmt2 = "{: 10d}   {: 12.6f}  {: 12.6f}\n"
		iter_i = np.array(iter_error_tst[:,0],dtype=int)
		error  = np.array(iter_error_tst[:,1:])/E_scaling
		for i in range(len(iter_i)):

			if tin.train_forces:
				f.write(fmt1.format(iter_i[i], *error[i]))
			else:
				f.write(fmt2.format(iter_i[i], error[i][0], error[i][1]))



def io_footer():
	io_print("")
	io_print("")
	io_current_time()
	io_print("")
	io_print("")
	io_double_line()
	io_print_center("Neural Network training done.")
	io_double_line()	