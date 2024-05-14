#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerPreDeconv2D.h"

//=====================================================================================
//  PreDeconv2D層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagPreDeconv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		strideHeight;	//ストライド高
	uint32_t		strideWidth;	//ストライド幅
	uint32_t		outHeight;		//出力高
	uint32_t		outWidth;		//出力幅
} PreDeconv2DNeuralNetHeader;

//=====================================================================================
//  PreDeconv2D層構造体
//=====================================================================================
typedef struct tagPreDeconv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
} PreDeconv2DNeuralNetLayer;

//=====================================================================================
//  形状関連情報計算
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_getShapeInformation(
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t	outHeight,
	uint32_t	outWidth,
	int32_t*	pPaddingHeight,
	int32_t*	pPaddingWidth,
	DataShape*	pOutputShape)
{
	int32_t		outHeightStride;
	int32_t		outWidthStride;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	//height
	outHeightStride = (inHeight - 1) * strideHeight + 1;
	paddingHeight =  (int32_t)outHeight - outHeightStride;
	//width
	outWidthStride = (inWidth - 1) * strideWidth + 1;
	paddingWidth = (int32_t)outWidth - outWidthStride;
	//---------------------------------------------------------------------------------
	//形状情報出力
	//---------------------------------------------------------------------------------
	if (pPaddingHeight != NULL) {
		*pPaddingHeight = paddingHeight;
	}
	if (pPaddingWidth != NULL) {
		*pPaddingWidth = paddingWidth;
	}
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, outHeight, outWidth, inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	PreDeconv2DNeuralNetLayer* pPreDeconv2DLayer = (PreDeconv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPreDeconv2DLayer;
	PreDeconv2DNeuralNetHeader* pPreDeconv2DNeuralNetHeader = (PreDeconv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPreDeconv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//Paddingサイズ
	//---------------------------------------------------------------------------------
	NeuralNetLayerPreDeconv2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pPreDeconv2DNeuralNetHeader->strideHeight,
		pPreDeconv2DNeuralNetHeader->strideWidth,
		pPreDeconv2DNeuralNetHeader->outHeight,
		pPreDeconv2DNeuralNetHeader->outWidth,
		NULL,
		NULL,
		pOutputShape);
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	PreDeconv2DNeuralNetLayer* pPreDeconv2DLayer = (PreDeconv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPreDeconv2DLayer;
	PreDeconv2DNeuralNetHeader* pPreDeconv2DNeuralNetHeader = (PreDeconv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPreDeconv2DNeuralNetHeader;
	uint32_t	i;
	uint32_t	iH;
	uint32_t	iW;
	uint32_t	iC;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	outChannel;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	flt32_t*	pInputData;
	flt32_t*	pOutputData;
	DataShape	outputShape;
	int32_t		posH;
	int32_t		posW;
	int32_t		pos;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPreDeconv2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pPreDeconv2DNeuralNetHeader->strideHeight,
		pPreDeconv2DNeuralNetHeader->strideWidth,
		pPreDeconv2DNeuralNetHeader->outHeight,
		pPreDeconv2DNeuralNetHeader->outWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	if (fStatus == FALSE) {
		return FALSE;
	}
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	outChannel = outputShape.channel;
	//---------------------------------------------------------------------------------
	//高速化と見やすさのため一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	strideHeight	= pPreDeconv2DNeuralNetHeader->strideHeight;
	strideWidth		= pPreDeconv2DNeuralNetHeader->strideWidth;
	//---------------------------------------------------------------------------------
	//データ初期化
	//---------------------------------------------------------------------------------
	pOutputData = pPropagationInfo->pOutputBuffer;
	i = DataShape_getSize(&outputShape);
	while (i--) {
		*pOutputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//出力データ
	//---------------------------------------------------------------------------------
	paddingHeight /= 2;	//edge padding height
	paddingWidth /= 2;	//edge padding width
	pInputData = pPropagationInfo->pInputBuffer;
	posH = paddingHeight;
	iH = inHeight;
	while (iH--) {
		if (posH < 0) {
			pInputData += inWidth * inChannel;
		}
		else if (posH >= (int16_t)outHeight) {
			//終了
			break;
		}
		else {
			posW = paddingWidth;
			iW = inWidth;
			while (iW--) {
				if (posW < 0) {
					pInputData += inChannel;
				}
				else if (posW >= (int16_t)outWidth) {
					pInputData += inChannel;
				}
				else {
					pos = (posH * outHeight + posW) * outChannel;
					pOutputData = pPropagationInfo->pOutputBuffer + (uint32_t)pos;
					iC = inChannel;
					while (iC--) {
						*pOutputData++ = *pInputData++;
					}
				}
				posW += strideWidth;
			}
		}
		posH += strideHeight;
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape = outputShape;
	return TRUE;
}

//=====================================================================================
//  逆伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	uint32_t	i;
	PreDeconv2DNeuralNetLayer* pPreDeconv2DLayer = (PreDeconv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPreDeconv2DLayer;
	PreDeconv2DNeuralNetHeader* pPreDeconv2DNeuralNetHeader = (PreDeconv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPreDeconv2DNeuralNetHeader;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	outChannel;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	int32_t		iH;
	int32_t		iW;
	uint32_t	iC;
	flt32_t*	pInputData;
	flt32_t*	pOutputData;
	DataShape	outputShape;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	int32_t		posH;
	int32_t		posW;
	int32_t		pos;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPreDeconv2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pPreDeconv2DNeuralNetHeader->strideHeight,
		pPreDeconv2DNeuralNetHeader->strideWidth,
		pPreDeconv2DNeuralNetHeader->outHeight,
		pPreDeconv2DNeuralNetHeader->outWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	if (fStatus == FALSE) {
		return FALSE;
	}
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	outChannel = outputShape.channel;
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	strideHeight = pPreDeconv2DNeuralNetHeader->strideHeight;
	strideWidth = pPreDeconv2DNeuralNetHeader->strideWidth;
	//---------------------------------------------------------------------------------
	//逆伝搬誤差バッファーを初期化する
	//---------------------------------------------------------------------------------
	i = inHeight * inWidth * inChannel;
	pInputData = pPropagationInfo->pInputBuffer;
	while (i--) {
		*pInputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//出力データ
	//---------------------------------------------------------------------------------
	paddingHeight /= 2;	//edge padding height
	paddingWidth /= 2;	//edge padding width
	pInputData = pPropagationInfo->pInputBuffer;
	posH = paddingHeight;
	iH = inHeight;
	while (iH--) {
		if (posH < 0) {
			pInputData += inWidth * inChannel;
		}
		else if (posH >= (int16_t)outHeight) {
			//終了
			break;
		}
		else {
			posW = paddingWidth;
			iW = inWidth;
			while (iW--) {
				if (posW < 0) {
					pInputData += inChannel;
				}
				else if (posW >= (int16_t)outWidth) {
					pInputData += inChannel;
				}
				else {
					pos = (posH * outHeight + posW) * outChannel;
					pOutputData = pPropagationInfo->pOutputBuffer + (uint32_t)pos;	//伝搬誤差
					iC = inChannel;
					while (iC--) {
						*pInputData++ = *pOutputData++;
					}
				}
				posW += strideWidth;
			}
		}
		posH += strideHeight;
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
NeuralNetLayerPreDeconv2D_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerPreDeconv2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	PreDeconv2DNeuralNetHeader* pPreDeconv2DNeuralNetHeader = (PreDeconv2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPreDeconv2DNeuralNetHeader;
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(PreDeconv2DNeuralNetLayer), uint32_t);
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
	NeuralNetLayerPreDeconv2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pPreDeconv2DNeuralNetHeader->strideHeight,
		pPreDeconv2DNeuralNetHeader->strideWidth,
		pPreDeconv2DNeuralNetHeader->outHeight,
		pPreDeconv2DNeuralNetHeader->outWidth,
		NULL,
		NULL,
		pOutputShape);
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
NeuralNetLayerPreDeconv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerPreDeconv2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	LayerFuncTable	funcTable;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	PreDeconv2DNeuralNetLayer* pPreDeconv2DLayer = (PreDeconv2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPreDeconv2DLayer;
	NeuralNetLayerPreDeconv2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerPreDeconv2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		return (handle_t)pPreDeconv2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerPreDeconv2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerPreDeconv2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerPreDeconv2D_construct;
	pInterface->pGetShape = NeuralNetLayerPreDeconv2D_getShape;
	pInterface->pForward = NeuralNetLayerPreDeconv2D_forward;
	pInterface->pBackward = NeuralNetLayerPreDeconv2D_backward;
	pInterface->pUpdate = NeuralNetLayerPreDeconv2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerPreDeconv2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerPreDeconv2D_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerPreDeconv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t	outHeight,
	uint32_t	outWidth,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	DataShape	outputShape;
	PreDeconv2DNeuralNetHeader* pPreDeconv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(PreDeconv2DNeuralNetHeader), uint32_t);
	sizeLayer = sizeHeader;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
	}
	//---------------------------------------------------------------------------------
	//層データ構築
	//---------------------------------------------------------------------------------
	if (pBuffer != NULL) {
		if (sizeOfBufferIn32BitWord < sizeLayer) {
			return FALSE;
		}
		//バッファーの先頭をセット
		pLayer = pBuffer;
		//header
		pPreDeconv2DNeuralNetHeader = (PreDeconv2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pPreDeconv2DNeuralNetHeader->super, NET_LAYER_PREDECONV2D, inHeight, inWidth, inChannel, sizeLayer);
		pPreDeconv2DNeuralNetHeader->strideHeight	= strideHeight;
		pPreDeconv2DNeuralNetHeader->strideWidth	= strideWidth;
		pPreDeconv2DNeuralNetHeader->outHeight		= outHeight;
		pPreDeconv2DNeuralNetHeader->outWidth		= outWidth;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	NeuralNetLayerPreDeconv2D_getShapeInformation(	
		inHeight, 
		inWidth, 
		inChannel,
		strideHeight, 
		strideWidth,
		outHeight,
		outWidth,
		NULL,
		NULL,
		&outputShape);
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = outputShape.height;
	*pInputWidth = outputShape.width;
	*pInputChannel = outputShape.channel;
	return TRUE;
}
