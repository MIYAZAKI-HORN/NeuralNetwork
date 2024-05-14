#ifndef NEURAL_NET_LAYER_ACTIVATION_H
#define NEURAL_NET_LAYER_ACTIVATION_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  Activation関数タイプ
//=====================================================================================
typedef enum tagNeuralNetActivationType {
	NEURAL_NET_ACTIVATION_RELU		= 0x1,
	NEURAL_NET_ACTIVATION_TANH		= 0x2,
	NEURAL_NET_ACTIVATION_SIGMOID	= 0x3,
	NEURAL_NET_ACTIVATION_SOFTMAX	= 0x4
} NeuralNetActivationType;

//=====================================================================================
//  Activation層インターフェース取得
//=====================================================================================
//relu
void	NeuralNetLayerReluActivation_getInterface(LayerFuncTable* pInterface);
//tanh
void	NeuralNetLayerTanhActivation_getInterface(LayerFuncTable* pInterface);
//sigmoid
void	NeuralNetLayerSigmoidActivation_getInterface(LayerFuncTable* pInterface);
//softmax
void	NeuralNetLayerSoftmaxActivation_getInterface(LayerFuncTable* pInterface);
//helper
void	NeuralNetLayerActivation_getInterface(LayerFuncTable* pInterface, uint32_t* pLayerData);

//=====================================================================================
//  Activation層作成
//=====================================================================================
//relu
bool_t	NeuralNetLayerReluActivation_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//tanh
bool_t	NeuralNetLayerTanhActivation_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//sigmoid
bool_t	NeuralNetLayerSigmoidActivation_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//softmax
bool_t	NeuralNetLayerSoftmaxActivation_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//helper
bool_t	NeuralNetLayerActivation_constructLayerData(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord,uint32_t* pInputHeight,uint32_t* pInputWidth, uint32_t* pInputChannel, NeuralNetActivationType activation, uint32_t* pSizeOfLayerIn32BitWord);

//=====================================================================================
//  パラメタ設定
//=====================================================================================
//relu
bool_t
NeuralNetLayerReluActivation_setParameter(handle_t hLayer, flt32_t negativeSlope);

//=====================================================================================
//  活性化関数タイプを取得する
//=====================================================================================
bool_t	NeuralNetLayerActivation_getType(handle_t hLayer, NeuralNetActivationType* pType);

#ifdef __cplusplus
}
#endif

#endif
