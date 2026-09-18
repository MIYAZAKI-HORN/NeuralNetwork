# NeuralNetwork
 C言語記述のニューラルネットワークソフトウエア
 (Neural network written in C)
 
 現在、ソースコード中のすべてのコメントは日本語で記述されています。
  (Currently, all comments in the source code are written in Japanese.)
 
## Supports the following layer modules (1 tensor input, 1 tensor output type)
	Dense(Affine)
	SimpleRNN
	Conv2D
	Depthwise Conv2D
	Pointwise Conv2D
	PreConv2D
	MaxPolling2D
	GlobalAveragePooling2D
	ResidualConnection(Sender/Receiver) *skip connection
	BatchNormalization
	LayerNormalization
	Activation(relu/tanh/sigmoid/softmax)
	
# General-purpose sequential neural network model by stacking the above layer modules
	Neural networks can be automatically constructed by loading sequential (stacked) layer data with headers into the prepared SequentialNet API.
	
## Functionality
  	The layer modules are capable of forward propagation/back propagation.
	Optimizer algorithms support : MomentumSGD/RMSProp/Adam
	You can build a sequential neural network by loading sequential layer data with headers into the SequentialNet model.
	When performing additional learning to original model, the learning layers can be fixed by specified the number counted from the final layer.

## System requirement
 	All softwares are written in C language and compatible with MPU/MCP/DSP with 32bit/64bit C compiler available.
	Designed with object-oriented style (encapsulation, virtual functions), it is easy to add new modules.
	Does not use system calls such as malloc (no OS required).
	Operations use single-precision floating point numbers. Maintains the necessary calculation accuracy by using a combination of function tables and algorithms without using math functions.
	Reduce the amount of calculations by writing source code by effectively using pointers.
	Contains API functions necessary for learning with multiple CPU cores.
	
## Sample programs
	classification problem
		Build a fully connected neural network or a convolutional neural network,
		Giving initial values to the parameters of the neural network, perform learning on the specified number of epochs.
		The cross-entropy error and classification rate are displayed sequentially for the training data and validation data.
		Verification data:MNIST in CSV format
	regression problem
		classification problem
		Build a fully connected neural network or a convolutional neural network,
		Giving initial values to the parameters of the neural network, perform learning on the specified number of epochs.
		The cross-entropy error and classification rate are displayed sequentially for the training data and validation data.
		Verification data:MNIST in CSV format
	YOLO
		This is a sample program designed to verify the object detection training capabilities of the YOLO (You Only Look Once) algorithm implemented in C.
		Among the various techniques proposed during the evolution of YOLO, this sample program primarily utilizes the following:
		1. DFL (Distribution Focal Loss)
			Treats the boundary positions of the bounding box as probability distributions rather than continuous values, predicting the ground-truth position probabilistically.
		2. CIoU (Complete IoU) Loss
			Improves box convergence speed and accuracy by accounting for the alignment of center distances and aspect ratios.
		3. Focal Loss for class classification probability calculation
			Renowned for ensuring training stability, particularly regarding the overwhelmingly large number of background grid cells.
