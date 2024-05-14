#ifndef NEURAL_NET_LAYER_LAYER_NORMALIZATION_H
#define NEURAL_NET_LAYER_LAYER_NORMALIZATION_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  LayerNormalization層インターフェース取得
//=====================================================================================
void
NeuralNetLayerLayerNormalization_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  LayerNormalization層作成
//=====================================================================================
bool_t
NeuralNetLayerLayerNormalization_constructLayerData(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord,uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel,uint32_t* pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
