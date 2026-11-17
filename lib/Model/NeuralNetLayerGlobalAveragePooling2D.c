#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerGlobalAveragePooling2D.h"

//=====================================================================================
//  GlobalAveragePooling2D層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagGlobalAveragePooling2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} GlobalAveragePooling2DNeuralNetHeader;

//=====================================================================================
//  GlobalAveragePooling2D層構造体
//=====================================================================================
typedef struct tagGlobalAveragePooling2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
} GlobalAveragePooling2DNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, 1, 1, pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  MaxPooling層　順伝播
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	bool_t		fEnableLearning;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	fEnableLearning	= pNeuralNetLayer->fEnableLearning;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	//------------------------------------------------------------------------------------------
	//平均値プーリング
	//------------------------------------------------------------------------------------------
	global_average_pooling2D_forward(pInputBuffer, pOutputBuffer, inHeight, inWidth, inChannel);
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, 1, 1, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  逆伝播
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	uint32_t	i;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	flt32_t*	pInput;
	//---------------------------------------------------------------------------------
	//逆伝播対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	//---------------------------------------------------------------------------------
	//バッファーの初期化
	//---------------------------------------------------------------------------------
	//誤差出力バッファー
	pInput = pInputBuffer;
	i = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	while (i--) {
		*pInput++ = 0.0f;
	}
	//------------------------------------------------------------------------------------------
	//平均値プーリング
	//------------------------------------------------------------------------------------------
	global_average_pooling2D_backward(pInputBuffer, pOutputBuffer, inHeight, inWidth, inChannel);
	//---------------------------------------------------------------------------------
	//逆伝播出力データサイズ形状(順伝播の入力データ形状)
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, inChannel);
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = 0;
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(GlobalAveragePooling2DNeuralNetLayer), uint32_t);
	}
	//---------------------------------------------------------------------------------
	//層内部の計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = 0;
	}
	//---------------------------------------------------------------------------------
	//出力形状
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, 1, 1, pNeuralNetHeader->inChannel);
	}
	//---------------------------------------------------------------------------------
	//入力形状
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	if (ppParameters != NULL) {
		*ppParameters = NULL;
	}
	if (pNumberOfParameters != NULL) {
		*pNumberOfParameters = 0;
	}
	return TRUE;
}

//=====================================================================================
//  ハイパーパラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getHyperParameters(handle_t hLayer, flt32_t* pParameterArray, uint32_t* pNumberOfParameters, uint32_t parameterArraySize) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pGlobalAveragePooling2DNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//ハイパーパラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfParameters != NULL) {
		*pNumberOfParameters = 0;
	}
	//---------------------------------------------------------------------------------
	//ハイパーパラメタ内容
	//---------------------------------------------------------------------------------
	if (pParameterArray != NULL) {
		// don't care
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerGlobalAveragePooling2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerGlobalAveragePooling2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerGlobalAveragePooling2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		return (handle_t)pGlobalAveragePooling2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerGlobalAveragePooling2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerGlobalAveragePooling2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerGlobalAveragePooling2D_construct;
	pInterface->pGetShape = NeuralNetLayerGlobalAveragePooling2D_getShape;
	pInterface->pForward = NeuralNetLayerGlobalAveragePooling2D_forward;
	pInterface->pBackward = NeuralNetLayerGlobalAveragePooling2D_backward;
	pInterface->pUpdate = NeuralNetLayerGlobalAveragePooling2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerGlobalAveragePooling2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerGlobalAveragePooling2D_getParameters;
	pInterface->pGetHyperParameters = NeuralNetLayerGlobalAveragePooling2D_getHyperParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	inHeight = *pInputHeight;
	inWidth = *pInputWidth;
	inChannel = *pInputChannel;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(GlobalAveragePooling2DNeuralNetHeader), uint32_t);
	sizeLayer = sizeHeader;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
	}
	//---------------------------------------------------------------------------------
	//層データ構築
	//---------------------------------------------------------------------------------
	if (pBuffer != NULL) {
		//サイズチェック
		if (sizeOfBufferIn32BitWord < sizeLayer) {
			return FALSE;
		}
		//バッファーの先頭をセット
		pLayer = pBuffer;
		// header
		pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pGlobalAveragePooling2DNeuralNetHeader->super, NET_LAYER_GLOBAL_AVERAGE_POOLING2D, inHeight, inWidth, inChannel, sizeLayer);
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= 1;
	*pInputWidth	= 1;
	*pInputChannel	= inChannel;
	return TRUE;
}
