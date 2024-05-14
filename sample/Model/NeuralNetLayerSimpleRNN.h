#ifndef NEURAL_NET_LAYER_SIMPLE_RNN_H
#define NEURAL_NET_LAYER_SIMPLE_RNN_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"
#include "NeuralNetLayerActivation.h"

//=====================================================================================
//  SimpleRNN層インターフェース取得
//=====================================================================================
void
NeuralNetLayerSimpleRNN_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  SimpleRNN層　最大誤差伝播時間の設定
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_setMaxPropagationTime(handle_t hLayer, uint32_t maxPropagationTime);

//=====================================================================================
//  SimpleRNN層作成
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord, 
	uint32_t*	pInputHeight, 
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	unit,
	NeuralNetActivationType activation,
	bool_t		returnSequence,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
