#ifndef NEURAL_NET_LAYER_CONV2D_H
#define NEURAL_NET_LAYER_CONV2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  Conv2D層インターフェース取得
//=====================================================================================
void
NeuralNetLayerConv2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  Conv2D層 作成
//=====================================================================================
bool_t
NeuralNetLayerConv2D_constructLayerData(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord,uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel,uint32_t nFilter,uint32_t	kernelHeight,uint32_t kernelWidth,uint32_t strideHeight,uint32_t	strideWidth,bool_t fPadding,uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
