#ifndef NEURAL_NET_LAYER_DEPTHWISE_CONV2D_H
#define NEURAL_NET_LAYER_DEPTHWISE_CONV2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  DepthwiseConv2D層インターフェース取得
//=====================================================================================
void
NeuralNetLayerDepthwiseConv2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  DepthwiseConv2D層作成
//=====================================================================================
bool_t
NeuralNetLayerDepthwiseConv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	nFilter,
	uint32_t	kernelHeight,
	uint32_t	kernelWidth,
	uint32_t	trideHeight,
	uint32_t	trideWidth,
	bool_t		fPadding,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
