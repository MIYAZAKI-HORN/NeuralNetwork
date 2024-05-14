#ifndef NEURAL_NET_LAYER_BATCH_NORMALIZATION_H
#define NEURAL_NET_LAYER_BATCH_NORMALIZATION_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  BatchNormalization層インターフェース取得
//=====================================================================================
void
NeuralNetLayerBatchNormalization_getInterface(LayerFuncTable* pInterface);
	
//=====================================================================================
//  BatchNormalization層　更新係数設定
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_setMomentum(handle_t hLayer, flt32_t momentum);

//=====================================================================================
//  BatchNormalization層作成
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_constructLayerData(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord,uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel,uint32_t* pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
