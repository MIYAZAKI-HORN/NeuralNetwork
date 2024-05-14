#ifndef NEURAL_NET_LAYER_MAX_POOLING2D_H
#define NEURAL_NET_LAYER_MAX_POOLING2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  MaxPooling2D層インターフェース取得
//=====================================================================================
void
NeuralNetLayerMaxPooling2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  MaxPooling2D層作成
//=====================================================================================
bool_t
NeuralNetLayerMaxPooling2D_constructLayerData(	
	uint32_t*	ppBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	poolingHeight,
	uint32_t	poolingWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
