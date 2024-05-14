
#include "NeuralNetLayer.h"

//=====================================================================================
// 層データ構造体
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
// 層構造体
//=====================================================================================
void	
NeuralNetLayer_construct(NeuralNetLayer* pLayer, uint32_t* pLayerData, bool_t fEnableLearning, LayerFuncTable funcTable, uint32_t layerOrder) {
	pLayer->pLayerData = pLayerData;
	pLayer->fEnableLearning = fEnableLearning;
	pLayer->funcTable = funcTable;
	pLayer->layerOrder = layerOrder;
}

//=====================================================================================
// 層タイプ取得
//=====================================================================================
bool_t
NeuralNetLayer_getType(handle_t hLayer,NetLayerType* pNetLayerType) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (pLayer == NULL) {
		return FALSE;
	}
	if (pNetLayerType == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層のタイプをセット
	//---------------------------------------------------------------------------------
	*pNetLayerType = pNeuralNetHeader->layerType;
	return TRUE;
}

//=====================================================================================
// 層計算順序取得
//=====================================================================================
bool_t
NeuralNetLayer_getOrder(handle_t hLayer, uint32_t* pOrder) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (pLayer == NULL) {
		return FALSE;
	}
	if (pOrder == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層計算順序をセット
	//---------------------------------------------------------------------------------
	*pOrder = pLayer->layerOrder;
	return TRUE;
}

//=====================================================================================
//  入出力形状取得
//=====================================================================================
bool_t
NeuralNetLayer_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pGetShape(hLayer, pInputShape, pOutputShape);
}

//=====================================================================================
//  順伝搬計算
//=====================================================================================
bool_t
NeuralNetLayer_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (hLayer == NULL) {
		return FALSE;
	}
	if (pPropagationInfo == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//計算順序
	//---------------------------------------------------------------------------------
	pPropagationInfo->layerOrder++;
	if (pLayer->layerOrder < pPropagationInfo->layerOrder) {
		pLayer->layerOrder = pPropagationInfo->layerOrder;
	}
	return pLayer->funcTable.pForward(hLayer, pPropagationInfo);
}

//=====================================================================================
//  逆伝搬計算
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
//  パラメタ更新
//=====================================================================================
bool_t
NeuralNetLayer_update(handle_t hLayer) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pUpdate(hLayer);
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
bool_t
NeuralNetLayer_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pInitializeParameters(hLayer,hRandomValueGenerator);
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
bool_t
NeuralNetLayer_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pLayer = (NeuralNetLayer*)hLayer;
	return pLayer->funcTable.pGetParameters(hLayer, ppParameters, pNumberOfParameters);
}
