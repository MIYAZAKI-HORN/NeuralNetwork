#ifndef NEURAL_NET_LAYER_RESIDUAL_CONNECTION_H
#define NEURAL_NET_LAYER_RESIDUAL_CONNECTION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//
//  ResidualConnectionSender�w
//
//=====================================================================================

//=====================================================================================
//  ResidualConnectionSender�w�C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerResidualConnectionSender_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X��`
//=====================================================================================
typedef bool_t(*ResidualConnectionSender_setReceiver)		(handle_t hLayer, handle_t hReceiver);
typedef bool_t(*ResidualConnectionSender_setBackwardData)	(handle_t hLayer, PropagationInfo* pPropagationInfo);

typedef struct tagResidualConnectionSenderInterface {
	ResidualConnectionSender_setReceiver		pSetReceiver;
	ResidualConnectionSender_setBackwardData	pSetBackwardData;
} ResidualConnectionSenderInterface;

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X�擾
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_getSenderInterface(handle_t hLayer, ResidualConnectionSenderInterface* pInterface);

//=====================================================================================
//  ResidualConnectionSender�w�쐬
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);

//=====================================================================================
//
//  ResidualConnectionReceiver�w
//
//=====================================================================================

//=====================================================================================
//  ResidualConnectionReceiver�w�C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerResidualConnectionReceiver_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X��`
//=====================================================================================
typedef bool_t(*ResidualConnectionReceiver_setForwardData)	(handle_t hLayer, handle_t hSender, PropagationInfo* pPropagationInfo);

typedef struct tagResidualConnectionReceiverInterface {
	ResidualConnectionReceiver_setForwardData	pSetForwardData;
} ResidualConnectionReceiverInterface;

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X�擾
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(handle_t hLayer, ResidualConnectionReceiverInterface* pInterface);

//=====================================================================================
//  ResidualConnectionReceiver�w�쐬
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_constructLayerData(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
