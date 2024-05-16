# NeuralNetwork
 C言語記述のニューラルネットワークソフトウエア
 (Neural network written in C)
 
 -現在、ソースコード中のすべてのコメントは日本語で記述されています。
  (Currently, all comments in the source code are written in Japanese.)
 
 -Supports the following layer modules (1 tensor input, 1 tensor output type)
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
	
 -General-purpose sequential neural network model by stacking the above layer modules
	Neural networks can be automatically constructed by loading sequential (stacked) layer data with headers into the prepared SequentialNet API.
	
 -functionality
  	The layer modules are capable of forward propagation/back propagation (learning).
	Optimizer algorithms support : MomentumSGD/RMSProp/Adam
	You can build a sequential neural network by loading sequential layer data with headers into the SequentialNet model.
	When performing additional learning to original model, the learning layers can be fixed by specified the number counted from the final layer.

 -system requirement
 	All softwares are written in C language and compatible with MPU/MCP/DSP with 32bit/64bit C compiler available.
	Designed with object orientation (encapsulation, virtual functions), it is easy to add new modules.
	Does not use system calls such as malloc (no OS required).
	Operations use single-precision floating point numbers. Maintains the necessary calculation accuracy by using a combination of function tables and algorithms without using math functions.
	Reduce the amount of calculations by writing source code by effectively using pointers.
	Contains API functions necessary for learning with multiple CPU cores.
	
 -sample programs
 	classification problem
		Build a fully connected neural network or a convolutional neural network,
		Giving initial values to the parameters of the neural network, perform learning on the specified number of epochs.
		The cross-entropy error and classification rate are displayed sequentially for the training data and validation data.
		Verification data:MNIST in CSV format
	regression problem
		Build a fully connected neural network,
		Giving initial values to the neural network parameters, perform training on the specified number of epochs.
		Sequentially display the mean squared error on the training data and validation data.
		Validation data:Boston House Prices in CSV format
		