#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerMaxPooling2D.h"

//=====================================================================================
//  MaxPooling2D層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagMaxPooling2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		poolingHeight;	//プーリング高
	uint32_t		poolingWidth;	//プーリング幅
	uint32_t		strideHeight;	//ストライド高
	uint32_t		strideWidth;	//ストライド幅
} MaxPooling2DNeuralNetHeader;

//=====================================================================================
//  MaxPooling2D層構造体
//=====================================================================================
typedef struct tagMaxPooling2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	uint32_t*		pMaxValueIndex;	//誤差逆伝搬用データバッファ
} MaxPooling2DNeuralNetLayer;

//=====================================================================================
//  形状関連情報計算
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getShapeInformation(
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
	uint32_t	poolingHeight,
	uint32_t	poolingWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	DataShape*	pOutputShape)
{
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
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		pOutputShape->height = 1 + (inHeight - poolingHeight) / strideHeight;
		pOutputShape->width = 1 + (inWidth - poolingWidth) / strideWidth;
		pOutputShape->channel = inChannel;
	}
	return TRUE;
}

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		pOutputShape);
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
NeuralNetLayerMaxPooling2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	uint32_t	i, j;
	DataShape	outputShape;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inWidth;
	uint32_t	inChannel;
	//一時変数
	uint32_t	poolingHeight;
	uint32_t	poolingWidth;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	bool_t		fEnableLearning;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	flt32_t*	pOutput;
	flt32_t		pixelValue;
	flt32_t		maxValue;
	uint32_t*	pMaxValueIndexHead;
	//データ位置
	uint32_t	iH;
	uint32_t	iW;
	uint32_t	iC;
	uint32_t	iCornerInHeight;
	uint32_t	iCornerInWidth;
	flt32_t*	pInputBufferCorner;
	flt32_t*	pInputBufferPoolY;
	flt32_t*	pInputBufferPool;
	flt32_t*	pXwithMaxValue;
	uint32_t	indexOfXwithMaxValue;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		if (pMaxPooling2DLayer->pMaxValueIndex == NULL) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		&outputShape);
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	poolingHeight	= pMaxPooling2DNeuralNetHeader->poolingHeight;
	poolingWidth	= pMaxPooling2DNeuralNetHeader->poolingWidth;
	strideHeight	= pMaxPooling2DNeuralNetHeader->strideHeight;
	strideWidth		= pMaxPooling2DNeuralNetHeader->strideWidth;
	fEnableLearning	= pNeuralNetLayer->fEnableLearning;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
	//------------------------------------------------------------------------------------------
	//最大値プーリング
	//------------------------------------------------------------------------------------------
	pOutput = pOutputBuffer;
	pMaxValueIndexHead = pMaxPooling2DLayer->pMaxValueIndex;
	iH = outHeight;
	iCornerInHeight = 0;
	while(iH--) {
		iW = outWidth;
		iCornerInWidth = 0;
		while(iW--) {
			pInputBufferCorner = pInputBuffer + (iCornerInHeight * inWidth + iCornerInWidth) * inChannel;
			iC = inChannel;
			while(iC--) {
				maxValue = *pInputBufferCorner;
				pInputBufferPoolY = pInputBufferCorner;
				pXwithMaxValue = pInputBufferPoolY;	//最大値を保持するXのポインタ
				i = poolingHeight;
				while (i--) {
					pInputBufferPool = pInputBufferPoolY;
					j = poolingWidth;
					while (j--) {
						pixelValue = *pInputBufferPool;
						if (maxValue < pixelValue) {
							maxValue = pixelValue;
							pXwithMaxValue = pInputBufferPool;
						}
						pInputBufferPool += inChannel;
					}
					pInputBufferPoolY += inWidth * inChannel;
				}
				if (pMaxValueIndexHead != NULL) {
					//最大値のインデックス
					indexOfXwithMaxValue = pXwithMaxValue - pInputBuffer;
					*pMaxValueIndexHead++ = indexOfXwithMaxValue;
				}
				pInputBufferCorner++;
				*pOutput++ = maxValue;
			}
			iCornerInWidth += strideWidth;
		}
		iCornerInHeight += strideHeight;
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
bool_t
NeuralNetLayerMaxPooling2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	uint32_t	i;
	DataShape	outputShape;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	poolingHeight;
	uint32_t	poolingWidth;
	//一時変数
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//データ位置
	uint32_t	iH;
	uint32_t	iW;
	uint32_t	iChan;
	uint32_t	iCornerInHeight;
	uint32_t	iCornerInWidth;
	bool_t		fStatus;
	flt32_t*	pDLossArray;
	uint32_t	indexOfXwithMaxValue;
	uint32_t	outDataCounter;
	flt32_t*	pInput;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		&outputShape);
	if (fStatus == FALSE) {
		return FALSE;
	}
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	poolingHeight	= pMaxPooling2DNeuralNetHeader->poolingHeight;
	poolingWidth	= pMaxPooling2DNeuralNetHeader->poolingWidth;
	strideHeight	= pMaxPooling2DNeuralNetHeader->strideHeight;
	strideWidth		= pMaxPooling2DNeuralNetHeader->strideWidth;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
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
	//最大値プーリング
	//------------------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	outDataCounter = 0;
	iH = outHeight;
	iCornerInHeight = 0;
	while (iH--) {
		iW = outWidth;
		iCornerInWidth = 0;
		for (iW = 0; iW < outWidth; iW++) {
			for (iChan = 0; iChan < inChannel; iChan++) {
				//------------------------------------------------------------------------------------------
				//最大値を保持するのXに対して誤差を伝搬
				//------------------------------------------------------------------------------------------
				//最大値のインデックス
				indexOfXwithMaxValue = pMaxPooling2DLayer->pMaxValueIndex[outDataCounter];
				//最大値のXに対し誤差を積算する
				pInputBuffer[indexOfXwithMaxValue] += (*pDLossArray);
				//伝搬誤差微分値を進める
				pDLossArray++;
				outDataCounter++;
			}
			iCornerInWidth += strideWidth;
		}
		iCornerInHeight += strideHeight;
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
NeuralNetLayerMaxPooling2D_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(MaxPooling2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(uint32_t) * nInput, uint32_t);
		}
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
	NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
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
NeuralNetLayerMaxPooling2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerMaxPooling2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerMaxPooling2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerMaxPooling2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(MaxPooling2DNeuralNetLayer), uint32_t);
			pMaxPooling2DLayer->pMaxValueIndex = (uint32_t*)pObjectWork;
		}
		else {
			pMaxPooling2DLayer->pMaxValueIndex = NULL;
		}
		return (handle_t)pMaxPooling2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerMaxPooling2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerMaxPooling2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerMaxPooling2D_construct;
	pInterface->pGetShape = NeuralNetLayerMaxPooling2D_getShape;
	pInterface->pForward = NeuralNetLayerMaxPooling2D_forward;
	pInterface->pBackward = NeuralNetLayerMaxPooling2D_backward;
	pInterface->pUpdate = NeuralNetLayerMaxPooling2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerMaxPooling2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerMaxPooling2D_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerMaxPooling2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	poolingHeight,
	uint32_t	poolingWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader;
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
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	if (inHeight < poolingHeight) {
		return FALSE;
	}
	if (inWidth < poolingWidth) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(MaxPooling2DNeuralNetHeader), uint32_t);
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
		pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pMaxPooling2DNeuralNetHeader->super, NET_LAYER_MAX_POOLING2D, inHeight, inWidth, inChannel, sizeLayer);
		pMaxPooling2DNeuralNetHeader->poolingHeight = poolingHeight;
		pMaxPooling2DNeuralNetHeader->poolingWidth = poolingWidth;
		pMaxPooling2DNeuralNetHeader->strideHeight = strideHeight;
		pMaxPooling2DNeuralNetHeader->strideWidth = strideWidth;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	outHeight		= 1 + (inHeight - poolingHeight) / strideHeight;
	outWidth		= 1 + (inWidth - poolingWidth) / strideWidth;
	*pInputHeight	= outHeight;
	*pInputWidth	= outWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
