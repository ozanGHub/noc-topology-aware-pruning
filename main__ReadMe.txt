#FILE NAME EXPLENATIONS for main files:
# many different tests were performed for different DNN models, pruning types and 
# comparison types.
#main_[DNNModel]_[Dataset-Optional]_[PruningType]and/or[ComparisonType]

#DNNModel					MNIST		AutoEnc-64		784-128-64-128-784+64-10    Bottleneck
#                           MNIST		LeNet-300-100 	784-300-100-10            	Small Feedforward
#                           MNIST		LeNet-500-300 	784-500-300-10            	Medium Feedforward
#                           MNIST		MLP-1024-1024 	784-1024-1024-10          	Large Feedforward
							
#							FaMNIST		AutoEnc-64		784-128-64-128-784+64-10    Bottleneck	
							
#							CIFAR-10	LeNet-300-100 	3072-300-100-10 			Small Feedforward
#							CIFAR-10	LeNet-500-300 	3072-500-300-10 			Medium Feedforward
#							CIFAR-10	MLP-1024-1024 	3072-1024-1024-10 			Large Feedforward
#							CIFAR-10	AutoEnc-64 		3072-128-64-128-3072+64-10 	Bottleneck
#							CIFAR-10	AutoEnc-128		3072-512-128-512-3072+128-10 Medium Bottleneck
#							CIFAR-10	LeNet-5 		conv1-conv3-c5-f6-out 		Convolutional
#							CIFAR-10	AlexNet 		conv1:conv5-c6-c7-out 		Convolutional


MAIN SCRIPT FILE for end to end comparison:

#main_DNNModel_e2e_AccuracyBudget:   Performs end to end comparison for neuron and edge pruning 
#                                    for best and second best classical methods
#                                    and compare heuristics (comm. cost aware and bridge-link) with these methods 

#OTHER MAIN FILES - previous steps in the process, path through end to end comparison


"""                                    
main_DNNModel:                      Performs heuristic comm.cost pruning & bridge-link edge pruning 
                                    for the DNNModel specified in the file name,
                                    and compares the result with standard L1 pruning only 

main_DNNModel_neuronMatched:        Performs neuron count matched 
                                    Pruning comparison for Comm. Cost Aware Neuron Pruning Heuristic Only
                                    for all classical neuron pruning heuristics (Struct-L1, Struct-L2, 
                                    Taylor, Activation, Random)

main_DNNModel_neuronMatchedBL:      Performs neuron count matched 
                                    Pruning comparison for Comm. Cost Aware Neuron Pruning 
                                    and Bridge Link Heuristic (applied on top of Comm.Cost)
                                    for all classical neuron pruning heuristics (Struct-L1, Struct-L2, 
                                    Taylor, Activation, Random)
                                    

main_DNNModel_EdgeMatched:          Performs edge count matched 
                                    Pruning comparison for Comm. Cost Aware + Bridge Link Heuristics
                                    for all classical edge pruning heuristics (L1, L2, 1st Order Taylor, SNIP)
                                                                      

main_DNNModel_BLonlyEdgeMatched:    Performs edge count matched 
                                    Pruning comparison for Bridge Link Edge Pruning Heuristic Only
                                    for all classical edge pruning heuristics (L1, L2, 1st Order Taylor, SNIP)
                                    Comm. Cost Aware Neuron pruning Heuristic is not executed


main_DNNModel_e2e_Comparison:       Performs end to end comparison 
                                    for neuron and edge pruning where the edge and neuron
                                    prunings are matched accross all methods.
                                    best and second best classical methods are
                                    compared heuristics (comm. cost aware and bridge-link)

main_DNNModel_e2e_Comparison
                    IterativeCC:    Same as main_DNNModel_e2e_Comparison but with 
                                    iterative communication cost heuristic.

main_LeNet_500_300_BudgetSweep:     Performs heuristic comm.cost pruning & bridge-link edge pruning 
                                    for the LeNet-500-300 DNNModel, and compares the result with 
                                    standard Taylor + L2 e2e pruning only 
                                    for a range of accuracy budgets (0.5% to 5% accuracy drop)                                    
                                    

main_Autoenc_3072_64_CIFAR10 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic		 		prunings are matched accross all methods.

main_Autoenc_3072_128_CIFAR10:		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic		 		prunings are matched accross all methods.
		
main_LeNet_300_100_CIFAR10	 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic				prunings are matched accross all methods.
		
main_LeNet_500_300_CIFAR10	 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic				prunings are matched accross all methods.

main_MLP_3072_1024_CIFAR10	 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic		  		prunings are matched accross all methods.

main_AlexNet_CIFAR10		 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic				prunings are matched accross all methods.
									With a run time switch it can also perform neuron count matched 
                                    Pruning for all branches.

main_LeNet5_CIFAR10			 :		Performs end to end comparison for CIFAR-10 dataset
	_e2e_AccuracyBudget				for neuron and edge pruning where the edge and neuron
		_Metaheuristic				prunings are matched accross all methods.
									With a run time switch it can also perform neuron count matched 
                                    Pruning for all branches.
                 
"""
