#ifndef NEURAL_NET_LAYER_OPTIMIZER_H
#define NEURAL_NET_LAYER_OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"

//-------------------------------------------------------------------------
// 最適化手法定義
//-------------------------------------------------------------------------
typedef enum tagNeuralNetworkOptimizerType {
	NEURAL_NET_OPTIMIZER_UNDEFINED	= 0x0,
	NEURAL_NET_OPTIMIZER_SGD		= 0x1,
	NEURAL_NET_OPTIMIZER_RMSPROP	= 0x2,
	NEURAL_NET_OPTIMIZER_ADAM		= 0x3
} NeuralNetOptimizerType;

typedef uint32_t(*NeuralNetOptimizer_getSizeIn32BitWord)	(uint32_t parameterSize);
typedef handle_t(*NeuralNetOptimizer_construct)				(uint32_t parameterSize, uint32_t batchSize, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord);

//=====================================================================================
//  クラス関数定義
//=====================================================================================
typedef NeuralNetOptimizerType(*NeuralNetOptimizer_getType)		(handle_t hOptimizer);
typedef uint32_t(*NeuralNetOptimizer_getParameterSize)			(handle_t hOptimizer);
typedef flt32_t*(*NeuralNetOptimizer_getDeltaParameterBuffer)	(handle_t hOptimizer);
typedef bool_t(*NeuralNetOptimizer_update)						(handle_t hOptimizer, flt32_t* pParameter);

typedef struct tagOptimizerFunctionTable {
	NeuralNetOptimizer_getSizeIn32BitWord		pGetSizeIn32BitWord;
	NeuralNetOptimizer_construct				pConstruct;
	NeuralNetOptimizer_getType					pGetType;
	NeuralNetOptimizer_getParameterSize			pGetParameterSize;
	NeuralNetOptimizer_getDeltaParameterBuffer	pGetDeltaParameterBuffer;
	NeuralNetOptimizer_update					pUpdate;
} *POptimizerFunctionTable, OptimizerFunctionTable;

//-------------------------------------------------------------------------
//  インターフェース取得関数
//-------------------------------------------------------------------------
bool_t	NeuralNetOptimizer_getInterfaceByType(NeuralNetOptimizerType type, OptimizerFunctionTable* pInterface);
bool_t	NeuralNetOptimizer_getInterface(handle_t hOptimizer, OptimizerFunctionTable* pInterface);

//-------------------------------------------------------------------------
//  SGDパラメタセット
//-------------------------------------------------------------------------
bool_t	NeuralNetworkOptimizerSGD_setParameters(handle_t hOptimizer, flt32_t momeNeuralNetOptimizer_getInterfaceByHandlentum, flt32_t learningRate);

//=====================================================================================
//  RMSpropパラメタセット
//=====================================================================================
bool_t	NeuralNetworkOptimizerRMSprop_setParameters(handle_t hOptimizer, flt32_t decayRate, flt32_t learningRate);

//=====================================================================================
//  Adamパラメタセット
//=====================================================================================
bool_t	NeuralNetworkOptimizerAdam_setParameters(handle_t hOptimizer, flt32_t beta1, flt32_t beta2, flt32_t learningRate);

#ifdef __cplusplus
}
#endif


#endif
