#ifndef NEURAL_NET_LAYER_RESIDUAL_CONNECTION_H
#define NEURAL_NET_LAYER_RESIDUAL_CONNECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//
//  ResidualConnectionSender層
//
//=====================================================================================

//=====================================================================================
//  ResidualConnectionSender層インターフェース取得
//=====================================================================================
void
NeuralNetLayerResidualConnectionSender_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  追加インターフェイス定義
//=====================================================================================
typedef bool_t(*ResidualConnectionSender_setReceiver)		(handle_t hLayer, handle_t hReceiver);
typedef bool_t(*ResidualConnectionSender_setBackwardData)	(handle_t hLayer, PropagationInfo* pPropagationInfo);

typedef struct tagResidualConnectionSenderInterface {
	ResidualConnectionSender_setReceiver		pSetReceiver;
	ResidualConnectionSender_setBackwardData	pSetBackwardData;
} ResidualConnectionSenderInterface;

//=====================================================================================
//  追加インターフェイス取得
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_getSenderInterface(handle_t hLayer, ResidualConnectionSenderInterface* pInterface);

//=====================================================================================
//  ResidualConnectionSender層作成
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);

//=====================================================================================
//
//  ResidualConnectionReceiver層
//
//=====================================================================================

//=====================================================================================
//  ResidualConnectionReceiver層インターフェース取得
//=====================================================================================
void
NeuralNetLayerResidualConnectionReceiver_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  追加インターフェイス定義
//=====================================================================================
typedef bool_t(*ResidualConnectionReceiver_setForwardData)	(handle_t hLayer, handle_t hSender, PropagationInfo* pPropagationInfo);

typedef struct tagResidualConnectionReceiverInterface {
	ResidualConnectionReceiver_setForwardData	pSetForwardData;
} ResidualConnectionReceiverInterface;

//=====================================================================================
//  追加インターフェイス取得
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(handle_t hLayer, ResidualConnectionReceiverInterface* pInterface);

//=====================================================================================
//  ResidualConnectionReceiver層作成
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
