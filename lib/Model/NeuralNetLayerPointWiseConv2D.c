#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerPointwiseConv2D.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  PointwiseConv2D層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagPointwiseConv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		nFilter;		//フィルタ数
} PointwiseConv2DNeuralNetHeader;

//=====================================================================================
//  PointwiseConv2D層構造体
//=====================================================================================
typedef struct tagPointwiseConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ
	handle_t		hOptimizer;		//オプティマイザーハンドル
} PointwiseConv2DNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pPointwiseConv2DNeuralNetHeader->nFilter);
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  層パラメタ
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_getLayerParameter(PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader, flt32_t** ppFilter,flt32_t** ppBias)
{
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t* pLayerParam;
	flt32_t* pFilter;
	flt32_t* pBias;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pPointwiseConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pPointwiseConv2DNeuralNetHeader->nFilter * pNeuralNetHeader->inChannel, uint32_t);
	pBias = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pPointwiseConv2DNeuralNetHeader->nFilter, uint32_t);
	//---------------------------------------------------------------------------------
	//パラメタ配列ポインタ
	//---------------------------------------------------------------------------------
	if (ppFilter != NULL) {
		*ppFilter = pFilter;
	}
	if (ppBias != NULL) {
		*ppBias = pBias;
	}
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	uint32_t	size;
	uint32_t	iFilter;
	uint32_t	iChan;
	flt32_t*	pFilterHead;
	flt32_t*	pInputHead;
	flt32_t*	pOutputData;
	flt32_t*	pBiasHead;
	uint32_t	dataSize;
	flt32_t		filterdData;
	flt32_t*	pInput;
	flt32_t*	pX;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//高速化と見やすさのため一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	nFilter		= pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//point方向の畳み込みを実施する
	//---------------------------------------------------------------------------------
	pOutputData = pPropagationInfo->pOutputBuffer;
	pInput = pPropagationInfo->pInputBuffer;
	//重みは（nFilter,nChannel）形状
	size = inHeight * inWidth;
	while (size--) {
		pBiasHead = pBias;
		pFilterHead = pFilter;
		iFilter = nFilter;
		while (iFilter--) {
			pInputHead = pInput;
			filterdData = *pBiasHead++;
			iChan = inChannel;
			while (iChan--) {
				filterdData += (*pInputHead++) * (*pFilterHead++);
			}
			*pOutputData++ = filterdData;
		}
		pInput += inChannel;
	}
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持:X
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pPointwiseConv2DLayer->pX == NULL) {
			return FALSE;
		}
		dataSize = inHeight * inWidth * inChannel;
		pInput = pPropagationInfo->pInputBuffer;
		pX = pPointwiseConv2DLayer->pX;
		while (dataSize--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, nFilter);
	return TRUE;
}

//=====================================================================================
//  逆伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	uint32_t	i;
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	uint32_t	size;
	uint32_t	iChan;
	flt32_t*	pFilterHead;
	uint32_t	dataSize;
	flt32_t*	pInputData;
	flt32_t*	pInputArray;
	flt32_t*	pDLossArray;
	flt32_t		deltaLoss;
	flt32_t*	pDFilter;
	flt32_t*	pDBias;
	flt32_t*	pDFilterHead;
	flt32_t*	pDBiasHead;
	flt32_t*	pXArray;	//入力保存バッファ
	flt32_t*	pXData;		//入力値
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, NULL);
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pPointwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pPointwiseConv2DLayer->hOptimizer);
	pDBias = pDFilter + (pPointwiseConv2DNeuralNetHeader->nFilter * inChannel);
	//---------------------------------------------------------------------------------
	// back propagation用ネットワーク微分パラメタバッファ
	//---------------------------------------------------------------------------------
	pDFilterHead	= pDFilter;
	pDBiasHead		= pDBias;
	//---------------------------------------------------------------------------------
	//誤差出力バッファーを初期化
	//---------------------------------------------------------------------------------
	dataSize = inHeight * inWidth * inChannel;
	pInputData = pPropagationInfo->pInputBuffer;
	while (dataSize--) {
		*pInputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
	pXArray = pPointwiseConv2DLayer->pX;
	pInputArray = pPropagationInfo->pInputBuffer;
	pDLossArray = pPropagationInfo->pOutputBuffer;
	//フィルタの重みが（pw_nFilter,pw_nChannel）で入っているとする
	size = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth;
	while (size--) {
		//-----------------------------------------------------------------
		//nFilter個のフィルターをかける
		//-----------------------------------------------------------------
		pFilterHead		= pFilter;		//フィルターバッファ
		pDFilterHead	= pDFilter;	//フィルター微分値バッファ
		pDBiasHead		= pDBias;		//フィルターバイアス微分値バッファ
		i = nFilter;
		while (i--) {
			//----------------------------------------------------------
			//伝搬誤差微分値
			//----------------------------------------------------------
			deltaLoss = *pDLossArray++;
			//----------------------------------------------------------
			//順伝搬時のフィルターブロックへの入力と、逆伝搬微分
			//----------------------------------------------------------
			pXData = pXArray;
			pInputData = pInputArray;
			//----------------------------------------------------------
			//bias(学習パラメタ)微分値積算
			//----------------------------------------------------------
			*pDBiasHead++ += deltaLoss;
			//----------------------------------------------------------
			//Filter(学習パラメタ)微分値積算
			//----------------------------------------------------------
			iChan = inChannel;
			while (iChan--) {
				//----------------------------------------------------------
				//フィルター係数(学習パラメタ)微分値へ積算
				//----------------------------------------------------------
				*pDFilterHead++ += (*pXData++) * deltaLoss;
				//----------------------------------------------------------
				//逆伝搬出力へ積算
				//----------------------------------------------------------
				*pInputData++ += (*pFilterHead++) * deltaLoss;
			}
		}
		//次のpixelに移動
		pXArray += inChannel;
		pInputArray += inChannel;
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
NeuralNetLayerPointwiseConv2D_update(handle_t hLayer) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	bool_t		fStatus;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pPointwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pParameter, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pPointwiseConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	uint32_t	paramSize;
	uint32_t	normSize;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	//フィルタ係数
	paramSize	= nFilter * inChannel;
	normSize	= paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pFilter, paramSize, normSize);
	//バイアス
	paramSize	= nFilter;
	set_constant_initial_values(pBias, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	if (pPointwiseConv2DNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = inChannel * nFilter;
		*pNumberOfLearningParameters += nFilter;
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(PointwiseConv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
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
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pPointwiseConv2DNeuralNetHeader->nFilter);
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
NeuralNetLayerPointwiseConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerPointwiseConv2D_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerPointwiseConv2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	LayerFuncTable	funcTable;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerPointwiseConv2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerPointwiseConv2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			//学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(PointwiseConv2DNeuralNetLayer), uint32_t);
			pPointwiseConv2DLayer->pX = (flt32_t*)pObjectWork;
			pPointwiseConv2DLayer->hOptimizer = hOptimizer;
		}
		else {
			pPointwiseConv2DLayer->pX = NULL;
			pPointwiseConv2DLayer->hOptimizer = NULL;
		}
		return (handle_t)pPointwiseConv2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerPointwiseConv2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerPointwiseConv2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerPointwiseConv2D_construct;
	pInterface->pGetShape = NeuralNetLayerPointwiseConv2D_getShape;
	pInterface->pForward = NeuralNetLayerPointwiseConv2D_forward;
	pInterface->pBackward = NeuralNetLayerPointwiseConv2D_backward;
	pInterface->pUpdate = NeuralNetLayerPointwiseConv2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerPointwiseConv2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerPointwiseConv2D_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerPointwiseConv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	nFilter,
	uint32_t*	pSizeOfLayerIn32BitWord) 
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamFilter;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
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
	sizeHeader = size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	sizeOfParamFilter = size_in_type(sizeof(flt32_t) * inChannel * nFilter, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * nFilter, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamFilter + sizeOfParamB;
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
		pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pPointwiseConv2DNeuralNetHeader->super, NET_LAYER_POINTWISE_CONV2D, inHeight, inWidth, inChannel, sizeLayer);
		pPointwiseConv2DNeuralNetHeader->nFilter = nFilter;
		pLayer += sizeHeader;
		//Filter
		pLayer += sizeOfParamFilter;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= nFilter;
	return TRUE;
}
