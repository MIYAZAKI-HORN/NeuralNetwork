
#ifndef NEURAL_NET_LAYER_H
#define NEURAL_NET_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//**********************************************************************************
//
// ニューラルネットワークモデル層パラメタヘッダー
//
//**********************************************************************************

#define		MODEL_FILE_VERSION	(3)

//=====================================================================================
// 層データ構造体
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
//  層ベースクラス
//=====================================================================================
typedef struct tagNeuralNetLayer {
	uint32_t*		pLayerData;
	bool_t			fEnableLearning;
	LayerFuncTable	funcTable;
	uint32_t		layerOrder;			//計算順序
} NeuralNetLayer;

void	NeuralNetLayer_construct(NeuralNetLayer* pLayer, uint32_t* pLayerData,bool_t fEnableLearning,LayerFuncTable	funcTable,uint32_t layerOrder);

//=====================================================================================
//  層タイプ取得
//=====================================================================================
bool_t	NeuralNetLayer_getType(handle_t hLayer, NetLayerType* pNetLayerType);

//=====================================================================================
// 層計算順序取得
//=====================================================================================
bool_t	NeuralNetLayer_getOrder(handle_t hLayer, uint32_t* pOrder);

//=====================================================================================
//  入出力形状取得
//=====================================================================================
bool_t	NeuralNetLayer_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape);

//=====================================================================================
//  順伝搬計算
//=====================================================================================
bool_t	NeuralNetLayer_forward(handle_t hLayer, PropagationInfo* pPropagationInfo);

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
bool_t	NeuralNetLayer_backward(handle_t hLayer, PropagationInfo* pPropagationInfo);

//=====================================================================================
//  パラメタ更新
//=====================================================================================
bool_t	NeuralNetLayer_update(handle_t hLayer);

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
bool_t	NeuralNetLayer_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator);

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
bool_t	NeuralNetLayer_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters);

#ifdef __cplusplus
}
#endif

#endif
