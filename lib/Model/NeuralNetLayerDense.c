#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDense.h"
#include "RandomValueGenerator.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  Dense層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagDenseNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		unit;			//ユニット数
} DenseNeuralNetHeader;

//=====================================================================================
//  Dense層構造体
//=====================================================================================
typedef struct tagDenseNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ
	handle_t		hOptimizer;		//オプティマイザーハンドル
} DenseNeuralNetLayer;

//=====================================================================================
//  DENSE層　順伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape,1, pDenseNeuralNetHeader->unit,1);
	return TRUE;
}

//=====================================================================================
//  DENSE層　順伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerDense_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit * nInput, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//重みマトリックス計算
	//---------------------------------------------------------------------------------
	weight_matrix_with_bias_forward(pPropagationInfo->pInputBuffer, nInput, pW, pB, pPropagationInfo->pOutputBuffer, pDenseNeuralNetHeader->unit,FALSE);
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		flt32_t*	pInput = pPropagationInfo->pInputBuffer;
		flt32_t*	pX;
		uint32_t	inputDim = nInput;
		//エラーハンドリング
		if (pDenseLayer->pX == NULL) {
			return FALSE;
		}
		//DENSE back propagationデータ
		pX = pDenseLayer->pX;
		while (inputDim--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape,1, pDenseNeuralNetHeader->unit, 1);
	return TRUE;
}

//=====================================================================================
//  DENSE層　逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerDense_backward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pDWeight;
	flt32_t*	pDBias;
	flt32_t*	pInput;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	uint32_t	size;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pDenseLayer->hOptimizer, &optimizerFunctionTable);
	pDWeight = optimizerFunctionTable.pGetDeltaParameterBuffer(pDenseLayer->hOptimizer);
	pDBias = pDWeight + pDenseNeuralNetHeader->unit * nInput;
	//---------------------------------------------------------------------------------
	//誤差伝搬先バッファ初期化
	//---------------------------------------------------------------------------------
	size = nInput;
	pInput = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInput++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//重みパラメタの誤差逆伝搬
	//---------------------------------------------------------------------------------
	weight_matrix_with_bias_backward(pPropagationInfo->pInputBuffer, nInput, pW, pPropagationInfo->pOutputBuffer, pDenseNeuralNetHeader->unit, pDenseLayer->pX, pDWeight, pDBias);
	//---------------------------------------------------------------------------------
	//出力（入力方向）データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, 1, nInput, 1);
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerDense_update(handle_t hLayer) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pDenseLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pParameter = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	//W＆B
	optimizerFunctionTable.pUpdate(pDenseLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerDense_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	uint32_t	paramSize;
	uint32_t	normSize;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit * nInput, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//層パラメタ初期化
	//---------------------------------------------------------------------------------
	//W
	paramSize = pDenseNeuralNetHeader->unit * nInput;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pW, paramSize, normSize);
	//B
	paramSize = pDenseNeuralNetHeader->unit;
	set_constant_initial_values(pB, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	if (pDenseNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pDenseNeuralNetHeader->unit * nInput;	//W
		*pNumberOfLearningParameters += pDenseNeuralNetHeader->unit;			//B
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(DenseNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
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
	//入出力形状
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, 1, pDenseNeuralNetHeader->unit, 1);
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerDense_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerDense_construct(	uint32_t*	pLayerData,
								uint32_t*	pObjectWork,
								uint32_t	sizeObjectIn32BitWord,
								bool_t		fEnableLearning,
								handle_t	hOptimizer) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t parameterSize;
	NeuralNetLayerDense_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL,NULL,NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		LayerFuncTable	funcTable;
		//層インターフェイス取得
		NeuralNetLayerDense_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//　学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			//学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork		+= size_in_type(sizeof(DenseNeuralNetLayer), uint32_t);
			pDenseLayer->pX			= (flt32_t*)pObjectWork;
			pDenseLayer->hOptimizer = hOptimizer;
		}
		else {
			pDenseLayer->pX			= NULL;
			pDenseLayer->hOptimizer	= NULL;
		}
		return (handle_t)pDenseLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerDense_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerDense_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerDense_construct;
	pInterface->pGetShape = NeuralNetLayerDense_getShape;
	pInterface->pForward = NeuralNetLayerDense_forward;
	pInterface->pBackward = NeuralNetLayerDense_backward;
	pInterface->pUpdate = NeuralNetLayerDense_update;
	pInterface->pInitializeParameters = NeuralNetLayerDense_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerDense_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerDense_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	unit,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamW;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nInput;
	DenseNeuralNetHeader* pDenseNeuralNetHeader;
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
	nInput = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	sizeOfParamW = size_in_type(sizeof(flt32_t) * unit * nInput, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamW + sizeOfParamB;
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
		pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pDenseNeuralNetHeader->super, NET_LAYER_DENSE, inHeight, inWidth, inChannel, sizeLayer);
		pDenseNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//W
		pLayer += sizeOfParamW;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = 1;
	*pInputWidth = unit;
	*pInputChannel = 1;
	return TRUE;
}
