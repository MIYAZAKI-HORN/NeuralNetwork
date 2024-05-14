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
//  MaxPooling層　順伝搬
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
	uint32_t	inPixels;
	//一時変数
	bool_t		fEnableLearning;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	flt32_t*	pChannelInput;
	flt32_t*	pInput;
	flt32_t*	pOutput;
	flt32_t		normalizationFactor;
	flt32_t		averageValue;
	//データ位置
	uint32_t	iP;
	uint32_t	iC;
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
	inPixels = inHeight * inWidth;
	normalizationFactor = 1.0f / (flt32_t)inPixels;
	pChannelInput = pInputBuffer;
	pOutput = pOutputBuffer;
	iC = inChannel;
	while(iC--) {
		//チャネルごとの平均値を計算
		averageValue = 0.0f;
		pInput = pChannelInput;	//入力バッファ(チャンネル先頭)
		iP = inPixels;
		while(iP--) {
			averageValue += *pInput;
			pInput += inChannel;	//次のチャネルデータへ移動
		}
		averageValue *= normalizationFactor;
		*pOutput++ = averageValue;
		pChannelInput++;
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, 1, 1, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  逆伝搬
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
	uint32_t	inPixels;
	//一時変数
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	flt32_t		normalizationFactor;
	flt32_t*	pDLossArray;
	flt32_t*	pChannelInput;
	flt32_t*	pInput;
	//データ位置
	uint32_t	iP;
	uint32_t	iC;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
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
	inPixels = inHeight * inWidth;
	normalizationFactor = 1.0f / (flt32_t)inPixels;
	pDLossArray = pOutputBuffer;
	pChannelInput = pInputBuffer;
	iC = inChannel;
	while (iC--) {
		pInput = pChannelInput;	//入力バッファ(チャンネル先頭)
		iP = inPixels;
		while (iP--) {
			*pInput = (*pDLossArray) * normalizationFactor;
			pInput += inChannel;	//次のチャネルデータへ移動
		}
		pDLossArray++;
		pChannelInput++;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬出力データサイズ形状(順伝搬の入力データ形状)
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
