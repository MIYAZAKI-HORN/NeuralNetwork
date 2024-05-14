CC	:= gcc

CFLAGS	:= -c -O2 -W -I./Model

SOURCE	:= 	./Model/NeuralNetLayer.c		\
		./Model/NeuralNetLayerActivation.c	\
		./Model/NeuralNetLayerBatchNormalization.c	\
		./Model/NeuralNetLayerConv2D.c	\
		./Model/NeuralNetLayerDense.c	\
		./Model/NeuralNetLayerDepthWiseConv2D.c	\
		./Model/NeuralNetLayerFunction.c	\
		./Model/NeuralNetLayerGlobalAveragePooling2D.c	\
		./Model/NeuralNetLayerLayerNormalization.c	\
		./Model/NeuralNetLayerMaxPooling2D.c	\
		./Model/NeuralNetLayerPointWiseConv2D.c	\
		./Model/NeuralNetLayerPreDeconv2D.c	\
		./Model/NeuralNetLayerResidualConnection.c	\
		./Model/NeuralNetLayerSimpleRNN.c	\
		./Model/NeuralNetOptimizer.c	\
		./Model/NeuralNetLayerType.c	\
		./Model/RandomValueGenerator.c	\
		./Model/SequentialNet.c


OBJS	:= 	./NeuralNetLayer.o		\
		./NeuralNetLayerActivation.o	\
		./NeuralNetLayerBatchNormalization.o	\
		./NeuralNetLayerConv2D.o	\
		./NeuralNetLayerDense.o	\
		./NeuralNetLayerDepthWiseConv2D.o	\
		./NeuralNetLayerFunction.o	\
		./NeuralNetLayerGlobalAveragePooling2D.o	\
		./NeuralNetLayerLayerNormalization.o	\
		./NeuralNetLayerMaxPooling2D.o	\
		./NeuralNetLayerPointWiseConv2D.o	\
		./NeuralNetLayerPreDeconv2D.o	\
		./NeuralNetLayerResidualConnection.o	\
		./NeuralNetLayerSimpleRNN.o	\
		./NeuralNetOptimizer.o	\
		./NeuralNetLayerType.o	\
		./RandomValueGenerator.o	\
		./SequentialNet.o

LIBRARY	  := ./SequentialNet.lib


sample:
	$(CC) $(CFLAGS) $(SOURCE)
	ar -r $(LIBRARY) $(OBJS)
	rm *.o

clean:
	rm *.o
