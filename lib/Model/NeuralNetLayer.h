
#ifndef NEURAL_NET_LAYER_H
#define NEURAL_NET_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//**********************************************************************************
//
// �j���[�����l�b�g���[�N���f���w�p�����^�w�b�_�[
//
//**********************************************************************************

#define		MODEL_FILE_VERSION	(3)

//=====================================================================================
// �w�f�[�^�\����
//=====================================================================================
typedef struct tagNeuralNetHeader {
	NetLayerType	layerType;
	uint32_t		inHeight;
	uint32_t		inWidth;
	uint32_t		inChannel;
	uint32_t		sizeIn32BitWord;
} NeuralNetHeader;

void	NeuralNetHeader_construct(NeuralNetHeader* pHeader, NetLayerType layerType, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel, uint32_t sizeIn32BitWord);

//=====================================================================================
//  �w�x�[�X�N���X
//=====================================================================================
typedef struct tagNeuralNetLayer {
	uint32_t*		pLayerData;
	bool_t			fEnableLearning;
	LayerFuncTable	funcTable;
	uint32_t		layerOrder;			//�v�Z����
} NeuralNetLayer;

void	NeuralNetLayer_construct(NeuralNetLayer* pLayer, uint32_t* pLayerData,bool_t fEnableLearning,LayerFuncTable	funcTable,uint32_t layerOrder);

//=====================================================================================
//  �w�^�C�v�擾
//=====================================================================================
bool_t	NeuralNetLayer_getType(handle_t hLayer, NetLayerType* pNetLayerType);

//=====================================================================================
// �w�v�Z�����擾
//=====================================================================================
bool_t	NeuralNetLayer_getOrder(handle_t hLayer, uint32_t* pOrder);

//=====================================================================================
//  ���o�͌`��擾
//=====================================================================================
bool_t	NeuralNetLayer_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape);

//=====================================================================================
//  ���`���v�Z
//=====================================================================================
bool_t	NeuralNetLayer_forward(handle_t hLayer, PropagationInfo* pPropagationInfo);

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
bool_t	NeuralNetLayer_backward(handle_t hLayer, PropagationInfo* pPropagationInfo);

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
bool_t	NeuralNetLayer_update(handle_t hLayer);

//=====================================================================================
//  �p�����^������
//=====================================================================================
bool_t	NeuralNetLayer_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator);

//=====================================================================================
//  �w�K�p�����^���擾
//=====================================================================================
bool_t	NeuralNetLayer_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters);

#ifdef __cplusplus
}
#endif

#endif
