#ifndef NEURAL_NET_LAYER_POINTWISE_CONV2D_H
#define NEURAL_NET_LAYER_POINTWISE_CONV2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  PointwiseConv2D層インターフェース取得
//=====================================================================================
void
NeuralNetLayerPointwiseConv2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  SeparableConv2D層作成
//=====================================================================================
bool_t
NeuralNetLayerPointwiseConv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	nFilter,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
