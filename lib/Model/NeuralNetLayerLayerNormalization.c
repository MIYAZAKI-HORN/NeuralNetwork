#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerLayerNormalization.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON	(0.00001f)

//=====================================================================================
//  LayerNormalization層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagLayerNormalizationNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} LayerNormalizationNeuralNetHeader;

//=====================================================================================
//  LayerNormalization層構造体
//=====================================================================================
typedef struct tagLayerNormalizationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ：x
	flt32_t			mean;
	flt32_t			invStd;
	handle_t		hOptimizer;		//オプティマイザーハンドル
} LayerNormalizationNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  順伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_forward_calculation(
	uint32_t	size,
	flt32_t		gamma,
	flt32_t		beta,
	flt32_t*	pMean,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer) {
	uint32_t	i;
	flt32_t		x;
	flt32_t		X;
	flt32_t		y;
	flt32_t		sumOfX		= 0.0f;
	flt32_t		sumOfVar	= 0.0f;
	flt32_t		mean		= 0.0f;
	flt32_t		var			= 1.0f;
	flt32_t		invVar		= 1.0f;
	flt32_t		diff;
	flt32_t*	pInputData;
	//平均値
	pInputData = pInputBuffer;
	i = size;
	while (i--) {
		sumOfX += *pInputData++;
	}
	mean = sumOfX / (flt32_t)size;
	//分散
	pInputData = pInputBuffer;
	i = size;
	while (i--) {
		diff = *pInputData++ - mean;
		sumOfVar += diff * diff;
	}
	var = sumOfVar / (flt32_t)size;
	//分散逆数
	invVar = 1.0f / (var+ EPSILON);
	//平均値と分散を保持
	*pMean = mean;
	*pInvStd = invVar;
	//順伝搬データ
	i = size;
	while (i--) {
		x = *pInputBuffer++;
		X = (x - mean) * invVar;
		//順伝搬出力
		y = gamma * X + beta;
		*pOutputBuffer++ = y;
		//逆伝搬用保持
		if (pX != NULL) {
			*pX++ = X;	//gammaの微分値ととして保持：y = gamma * X + beta
		}
	}
	return TRUE;
}

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_backward_calculation(
	uint32_t	size,
	flt32_t		gamma,
	flt32_t		invStd,
	flt32_t*	pX,
	flt32_t*	pDGamma,
	flt32_t*	pDBeta,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer)
{
	uint32_t	i;
	flt32_t*	pInput;
	flt32_t*	pDLossArray;
	//---------------------------------------------------------------------------------
	//パラメタの誤差逆伝搬
	//---------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	pInput = pInputBuffer;
	i = size;
	while (i--) {
		//パラメタ微分値
		*pDGamma += (*pX++) * (*pDLossArray);
		*pDBeta += (*pDLossArray);
		//逆伝搬
		*pInput++ += gamma * invStd * (*pDLossArray);
		//逆伝入力値伝搬誤差微分値ポインタ更新
		pDLossArray++;
	}
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	flt32_t*	pX;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//重みパラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);	//Header分だけ異動
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pLayerNormalizationLayer->pX == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//入力、入力積算、分散積算
		//-------------------------------------------------------
		pX	= pLayerNormalizationLayer->pX;
	}
	else {
		pX	= NULL;
	}
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//入力次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//正規化処理の対処うデータ
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	//---------------------------------------------------------------------------------
	//正規化処理：channel方向にかける場合とwidth方向にかける場合がある
	//---------------------------------------------------------------------------------
	NeuralNetLayerLayerNormalization_forward_calculation(size,*pGamma,*pBeta,&pLayerNormalizationLayer->mean,&pLayerNormalizationLayer->invStd,pX,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape.height = pPropagationInfo->dataShape.height;
	pPropagationInfo->dataShape.width = pPropagationInfo->dataShape.width;
	pPropagationInfo->dataShape.channel = pPropagationInfo->dataShape.channel;
	return TRUE;
}

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pDGamma;
	flt32_t*	pDBeta;
	flt32_t*	pInput;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Gamma
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//インターフェイス取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pLayerNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得：パラメタは連続して入っている
	//---------------------------------------------------------------------------------
	pDGamma = optimizerFunctionTable.pGetDeltaParameterBuffer(pLayerNormalizationLayer->hOptimizer);
	pDBeta = pDGamma + 1;
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//誤差伝搬先バッファーの初期化
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInput = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInput++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬計算
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	NeuralNetLayerLayerNormalization_backward_calculation(size,*pGamma,pLayerNormalizationLayer->invStd,pLayerNormalizationLayer->pX,pDGamma,pDBeta,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//出力（入力方向）データサイズ形状
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape.height = pPropagationInfo->dataShape.height;
	pPropagationInfo->dataShape.width = pPropagationInfo->dataShape.width;
	pPropagationInfo->dataShape.channel = pPropagationInfo->dataShape.channel;
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_update(handle_t hLayer) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam;
	flt32_t* pGamma;
	flt32_t* pBeta;
	OptimizerFunctionTable	optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//インターフェイス取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pLayerNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//重みパラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//層学習パラメタ更新：Gamma&Beta
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pLayerNormalizationLayer->hOptimizer, pGamma);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t*	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	uint32_t	paramSize;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);	//Header分だけ異動
	//Gamma
	pGamma = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	//Beta
	pBeta = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	//---------------------------------------------------------------------------------
	//層パラメタ初期化
	//---------------------------------------------------------------------------------
	//pGamma
	paramSize = 1;
	set_constant_initial_values(pGamma, paramSize, 1.0f);
	//pBeta
	paramSize = 1;
	set_constant_initial_values(pBeta, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t inputDataDim;
	if (pLayerNormalizationNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = 1;	//Gamma
		*pNumberOfLearningParameters += 1;	//Beta
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(LayerNormalizationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			//X
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//層内部計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = 0;
	}
	//---------------------------------------------------------------------------------
	//入出力形状
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerLayerNormalization_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerLayerNormalization_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	uint32_t parameterSize;
	NeuralNetLayerLayerNormalization_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerLayerNormalization_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		if (fEnableLearning == TRUE) {
			//学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(LayerNormalizationNeuralNetLayer), uint32_t);
			//学習用バッファ
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			pLayerNormalizationLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//メンバ初期化
			pLayerNormalizationLayer->hOptimizer	= hOptimizer;
			//バッファ初期化
			i = inputDataDim;
			while (i--) {
				pLayerNormalizationLayer->pX[i]		= 0.0f;
			}
		}
		else {
			//学習用バッファ
			pLayerNormalizationLayer->pX			= NULL;
			//メンバ初期化
			pLayerNormalizationLayer->hOptimizer	= hOptimizer;
		}
		return (handle_t)pLayerNormalizationLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerLayerNormalization_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerLayerNormalization_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerLayerNormalization_construct;
	pInterface->pGetShape = NeuralNetLayerLayerNormalization_getShape;
	pInterface->pForward = NeuralNetLayerLayerNormalization_forward;
	pInterface->pBackward = NeuralNetLayerLayerNormalization_backward;
	pInterface->pUpdate = NeuralNetLayerLayerNormalization_update;
	pInterface->pInitializeParameters = NeuralNetLayerLayerNormalization_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerLayerNormalization_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerLayerNormalization_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamGamma;
	uint32_t	sizeOfParamBeta;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	sizeOfParamGamma = size_in_type(sizeof(flt32_t) * 1, uint32_t);
	sizeOfParamBeta = size_in_type(sizeof(flt32_t) * 1, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamGamma + sizeOfParamBeta;
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
		//header
		pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pLayerNormalizationNeuralNetHeader->super, NET_LAYER_LAYER_NORMALIZATION, inHeight, inWidth, inChannel, sizeLayer);
		pLayer += sizeHeader;
		//Gamma
		pLayer += sizeOfParamGamma;
		//Beta
		pLayer += sizeOfParamBeta;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
