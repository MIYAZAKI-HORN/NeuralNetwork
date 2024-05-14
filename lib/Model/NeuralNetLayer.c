
#include "NeuralNetLayer.h"

//=====================================================================================
// �w�f�[�^�\����
//=====================================================================================
void
NeuralNetHeader_construct(NeuralNetHeader* pHeader, NetLayerType layerType,uint32_t inHeight,uint32_t inWidth,uint32_t inChannel,uint32_t sizeIn32BitWord) {
	pHeader->layerType = layerType;
	pHeader->inHeight = inHeight;
	pHeader->inWidth = inWidth;
	pHeader->inChannel = inChannel;
	pHeader->sizeIn32BitWord = sizeIn32BitWord;
}

//=====================================================================================
// �w�\����
//=====================================================================================
void	
NeuralNetLayer_construct(NeuralNetLayer* pLayer, uint32_t* pLayerData, bool_t fEnableLearning, LayerFuncTable funcTable, uint32_t layerOrder) {
	pLayer->pLayerData = pLayerData;
	pLayer->fEnableLearning = fEnableLearning;
	pLayer->funcTable = funcTable;
	pLayer->layerOrder = layerOrder;
}

//=====================================================================================
// �w�^�C�v�擾
//=====================================================================================
bool_t
NeuralNetLayer_getType(handle_t hLayer,NetLayerType* pNetLayerType) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (pLayer == NULL) {
		return FALSE;
	}
	if (pNetLayerType == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�̃^�C�v���Z�b�g
	//---------------------------------------------------------------------------------
	*pNetLayerType = pNeuralNetHeader->layerType;
	return TRUE;
}

//=====================================================================================
// �w�v�Z�����擾
//=====================================================================================
bool_t
NeuralNetLayer_getOrder(handle_t hLayer, uint32_t* pOrder) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (pLayer == NULL) {
		return FALSE;
	}
	if (pOrder == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�v�Z�������Z�b�g
	//---------------------------------------------------------------------------------
	*pOrder = pLayer->layerOrder;
	return TRUE;
}

//=====================================================================================
//  ���o�͌`��擾
//=====================================================================================
bool_t
NeuralNetLayer_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pGetShape(hLayer, pInputShape, pOutputShape);
}

//=====================================================================================
//  ���`���v�Z
//=====================================================================================
bool_t
NeuralNetLayer_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (hLayer == NULL) {
		return FALSE;
	}
	if (pPropagationInfo == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�v�Z����
	//---------------------------------------------------------------------------------
	pPropagationInfo->layerOrder++;
	if (pLayer->layerOrder < pPropagationInfo->layerOrder) {
		pLayer->layerOrder = pPropagationInfo->layerOrder;
	}
	return pLayer->funcTable.pForward(hLayer, pPropagationInfo);
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
bool_t
NeuralNetLayer_backward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	if (pLayer->fEnableLearning == TRUE) {
		return pLayer->funcTable.pBackward(hLayer, pPropagationInfo);
	}
	else {
		return TRUE;
	}
}

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
bool_t
NeuralNetLayer_update(handle_t hLayer) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pUpdate(hLayer);
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
bool_t
NeuralNetLayer_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pInitializeParameters(hLayer,hRandomValueGenerator);
}

//=====================================================================================
//  �w�K�p�����^���擾
//=====================================================================================
bool_t
NeuralNetLayer_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pGetParameters(hLayer, ppParameters, pNumberOfParameters);
}
