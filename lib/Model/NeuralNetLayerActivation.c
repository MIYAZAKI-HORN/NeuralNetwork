#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerActivation.h"

#define DEFAULT_RELU_ALPHA	(0.0f)

//=====================================================================================
//  Activation層ブロック情報ヘッダー
//=====================================================================================
//activation base
typedef struct tagActivationNeuralNetHeader {
	NeuralNetHeader			super;
	NeuralNetActivationType	activation;
} ActivationNeuralNetHeader;

//relu
typedef struct tagReluActivationNeuralNetHeader {
	ActivationNeuralNetHeader	super;
	flt32_t						alpha;
} ReluActivationNeuralNetHeader;

//tanh
typedef struct tagTanhActivationNeuralNetHeader {
	ActivationNeuralNetHeader	super;
} TanhActivationNeuralNetHeader;

//sigmoid
typedef struct tagSigmoidActivationNeuralNetHeader {
	ActivationNeuralNetHeader	super;
} SigmoidActivationNeuralNetHeader;

//softmax
typedef struct tagSoftmaxActivationNeuralNetHeader {
	ActivationNeuralNetHeader	super;
} SoftmaxActivationNeuralNetHeader;

//=====================================================================================
//  Activation層構造体
//=====================================================================================
//relu
typedef struct tagReluActivationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ
} ReluActivationNeuralNetLayer;

//tanh
typedef struct tagTanhActivationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pY;				//誤差逆伝搬用データバッファ
} TanhActivationNeuralNetLayer;

//sigmoid
typedef struct tagSigmoidActivationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pY;				//誤差逆伝搬用データバッファ
} SigmoidActivationNeuralNetLayer;

//softmax
typedef struct tagSoftmaxActivationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pY;				//誤差逆伝搬用データバッファ
} SoftmaxActivationNeuralNetLayer;


//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerActivation_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	NeuralNetLayer*		pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	NeuralNetHeader*	pNeuralNetHeader = (NeuralNetHeader*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape,pNeuralNetHeader->inHeight,pNeuralNetHeader->inWidth,pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
//relu
static
bool_t
NeuralNetLayerReluActivation_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer*					pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	ReluActivationNeuralNetLayer*	pReluNeuralNetLayer = (ReluActivationNeuralNetLayer*)pNeuralNetLayer;
	ReluActivationNeuralNetHeader*	pReluActivationNeuralNetHeader = (ReluActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t	size;
	flt32_t*	pInput;
	uint32_t	inputDim;
	flt32_t*	pX;
	//---------------------------------------------------------------------------------
	//順伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	relu_forward(pPropagationInfo->pInputBuffer, pPropagationInfo->pOutputBuffer, size, pReluActivationNeuralNetHeader->alpha);
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pReluNeuralNetLayer->pX == NULL) {
			return FALSE;
		}
		//逆誤差伝搬用データ保持
		pX = pReluNeuralNetLayer->pX;
		pInput = pPropagationInfo->pInputBuffer;
		inputDim = size;
		while (inputDim--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//tanh
static
bool_t
NeuralNetLayerTanhActivation_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	TanhActivationNeuralNetLayer*	pTanhNeuralNetLayer = (TanhActivationNeuralNetLayer*)pNeuralNetLayer;
	TanhActivationNeuralNetHeader*	pTanhActivationNeuralNetHeader = (TanhActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t	size;
	flt32_t*	pOutput;
	uint32_t	outputDim;
	flt32_t*	pY;
	//---------------------------------------------------------------------------------
	//順伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	tanh_forward(pPropagationInfo->pInputBuffer, pPropagationInfo->pOutputBuffer, size);
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pTanhNeuralNetLayer->pY == NULL) {
			return FALSE;
		}
		//逆誤差伝搬用データ保持
		pY = pTanhNeuralNetLayer->pY;
		pOutput = pPropagationInfo->pOutputBuffer;
		outputDim = size;
		while (outputDim--) {
			*pY++ = *pOutput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//sigmoid
static
bool_t
NeuralNetLayerSigmoidActivation_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	SigmoidActivationNeuralNetLayer*	pSigmoidActivationNeuralNetLayer = (SigmoidActivationNeuralNetLayer*)pNeuralNetLayer;
	SigmoidActivationNeuralNetHeader*	pSigmoidActivationNeuralNetHeader = (SigmoidActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t	size;
	flt32_t*	pOutput;
	uint32_t	outputDim;
	flt32_t*	pY;
	//---------------------------------------------------------------------------------
	//順伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	sigmoid_forward(pPropagationInfo->pInputBuffer, pPropagationInfo->pOutputBuffer, size);
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pSigmoidActivationNeuralNetLayer->pY == NULL) {
			return FALSE;
		}
		//逆誤差伝搬用データ保持
		pY = pSigmoidActivationNeuralNetLayer->pY;
		pOutput = pPropagationInfo->pOutputBuffer;
		outputDim = size;
		while (outputDim--) {
			*pY++ = *pOutput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//softmax
static
bool_t
NeuralNetLayerSoftmaxActivation_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	SoftmaxActivationNeuralNetLayer*	pSoftmaxActivationNeuralNetLayer = (SoftmaxActivationNeuralNetLayer*)pNeuralNetLayer;
	SoftmaxActivationNeuralNetHeader*	pSoftmaxActivationNeuralNetHeader = (SoftmaxActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t	size;
	flt32_t*	pOutput;
	uint32_t	outputDim;
	flt32_t*	pY;
	//---------------------------------------------------------------------------------
	//順伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	softmax_forward(pPropagationInfo->pInputBuffer, pPropagationInfo->pOutputBuffer, size);
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pSoftmaxActivationNeuralNetLayer->pY == NULL) {
			return FALSE;
		}
		//逆誤差伝搬用データ保持
		pY = pSoftmaxActivationNeuralNetLayer->pY;
		pOutput = pPropagationInfo->pOutputBuffer;
		outputDim = size;
		while (outputDim--) {
			*pY++ = *pOutput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//=====================================================================================
//  逆伝搬
//=====================================================================================
//relu
static
bool_t
NeuralNetLayerReluActivation_backward(handle_t hLayer,PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	ReluActivationNeuralNetLayer*	pReluActivationNeuralNetLayer = (ReluActivationNeuralNetLayer*)pNeuralNetLayer;
	ReluActivationNeuralNetHeader*	pReluActivationNeuralNetHeader = (ReluActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	relu_backword(pReluActivationNeuralNetLayer->pX, pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, size, pReluActivationNeuralNetHeader->alpha);
	//---------------------------------------------------------------------------------
	//逆伝搬出力データ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//tanh
static
bool_t
NeuralNetLayerTanhActivation_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	TanhActivationNeuralNetLayer* pTanhActivationNeuralNetLayer = (TanhActivationNeuralNetLayer*)pNeuralNetLayer;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	//微分：1 - y*y
	tanh_backword(pTanhActivationNeuralNetLayer->pY, pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, size);
	//---------------------------------------------------------------------------------
	//逆伝搬出力データ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//sigmoid
static
bool_t
NeuralNetLayerSigmoidActivation_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	SigmoidActivationNeuralNetLayer* pSigmoidActivationNeuralNetLayer = (SigmoidActivationNeuralNetLayer*)pNeuralNetLayer;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	//微分：y(1 - y)
	sigmoid_backword(pSigmoidActivationNeuralNetLayer->pY, pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, size);
	//---------------------------------------------------------------------------------
	//逆伝搬出力データ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//softmax
static
bool_t
NeuralNetLayerSoftmaxActivation_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	SoftmaxActivationNeuralNetLayer* pSoftmaxActivationNeuralNetLayer = (SoftmaxActivationNeuralNetLayer*)pNeuralNetLayer;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	//微分：y(1 - y) //出力と同じ次元の場合はsigmoidと同じ
	softmax_backword(pSoftmaxActivationNeuralNetLayer->pY, pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, size);
	//---------------------------------------------------------------------------------
	//逆伝搬出力データ形状
	//---------------------------------------------------------------------------------
	//変更無し
	return TRUE;
}

//=====================================================================================
//  パラメタ更新：学習パラメタ無し
//=====================================================================================
static
bool_t
NeuralNetLayerActivation_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化：：学習パラメタ無し
//=====================================================================================
static
bool_t
NeuralNetLayerActivation_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}


//=====================================================================================
//  層情報取得
//=====================================================================================
//relu
static
bool_t
NeuralNetLayerReluActivation_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
	ReluActivationNeuralNetHeader* pReluActivationNeuralNetHeader = (ReluActivationNeuralNetHeader*)pNeuralNetHeader;
	uint32_t	nInput;
	if (pReluActivationNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(ReluActivationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
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
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerActivation_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	if (ppParameters != NULL) {
		*ppParameters = NULL;
	}
	if (pNumberOfParameters != NULL) {
		*pNumberOfParameters = 0;
	}
	return TRUE;
}

//tanh
static
bool_t
NeuralNetLayerTanhActivation_getLayerInformation(
	uint32_t* pLayerData,
	bool_t		fEnableLearning,
	uint32_t* pLayerObjectSizeIn32BitWord,
	uint32_t* pNumberOfLearningParameters,
	uint32_t* pTempWorkAreaSizeIn32BitWord,
	DataShape* pInputShape,
	DataShape* pOutputShape) {
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
	TanhActivationNeuralNetHeader* pTanhActivationNeuralNetHeader = (TanhActivationNeuralNetHeader*)pNeuralNetHeader;
	uint32_t	nInput;
	if (pTanhActivationNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(TanhActivationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
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
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//sigmoid
static
bool_t
NeuralNetLayerSigmoidActivation_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
	SigmoidActivationNeuralNetHeader* pSigmoidActivationNeuralNetHeader = (SigmoidActivationNeuralNetHeader*)pNeuralNetHeader;
	uint32_t	nInput;
	if (pSigmoidActivationNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(SigmoidActivationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
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
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//softmax
static
bool_t
NeuralNetLayerSoftmaxActivation_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
	SoftmaxActivationNeuralNetHeader* pSoftmaxActivationNeuralNetHeader = (SoftmaxActivationNeuralNetHeader*)pNeuralNetHeader;
	uint32_t	nInput;
	if (pSoftmaxActivationNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(SoftmaxActivationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
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
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  Activation層構築
//=====================================================================================
//relu
static
handle_t
NeuralNetLayerReluActivation_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	ReluActivationNeuralNetLayer* pReluActivationNeuralNetLayer = (ReluActivationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pReluActivationNeuralNetLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerReluActivation_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerReluActivation_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer,pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(ReluActivationNeuralNetLayer), uint32_t);
			pReluActivationNeuralNetLayer->pX = (flt32_t*)pObjectWork;
		}
		else {
			pReluActivationNeuralNetLayer->pX = NULL;
		}
		return (handle_t)pNeuralNetLayer;
	}
}

//tanh
static
handle_t
NeuralNetLayerTanhActivation_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	TanhActivationNeuralNetLayer* pTanhActivationNeuralNetLayer = (TanhActivationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pTanhActivationNeuralNetLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerTanhActivation_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerTanhActivation_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(TanhActivationNeuralNetLayer), uint32_t);
			pTanhActivationNeuralNetLayer->pY = (flt32_t*)pObjectWork;
		}
		else {
			pTanhActivationNeuralNetLayer->pY = NULL;
		}
		return (handle_t)pNeuralNetLayer;
	}
}

//sigmoid
static
handle_t
NeuralNetLayerSigmoidActivation_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	SigmoidActivationNeuralNetLayer* pSigmoidActivationNeuralNetLayer = (SigmoidActivationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSigmoidActivationNeuralNetLayer;
	LayerFuncTable	funcTable;	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerSigmoidActivation_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerSigmoidActivation_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(SigmoidActivationNeuralNetLayer), uint32_t);
			pSigmoidActivationNeuralNetLayer->pY = (flt32_t*)pObjectWork;
		}
		else {
			pSigmoidActivationNeuralNetLayer->pY = NULL;
		}
		return (handle_t)pNeuralNetLayer;
	}
}

//softmax
static
handle_t
NeuralNetLayerSoftmaxActivation_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	SoftmaxActivationNeuralNetLayer* pSoftmaxActivationNeuralNetLayer = (SoftmaxActivationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSoftmaxActivationNeuralNetLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerSoftmaxActivation_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerSoftmaxActivation_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(SoftmaxActivationNeuralNetLayer), uint32_t);
			pSoftmaxActivationNeuralNetLayer->pY = (flt32_t*)pObjectWork;
		}
		else {
			pSoftmaxActivationNeuralNetLayer->pY = NULL;
		}
		return (handle_t)pNeuralNetLayer;
	}
}
//=====================================================================================
//  Activation層インターフェース取得
//=====================================================================================
//relu
void
NeuralNetLayerReluActivation_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerReluActivation_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerReluActivation_construct;
	pInterface->pGetShape = NeuralNetLayerActivation_getShape;							//super
	pInterface->pForward = NeuralNetLayerReluActivation_forward;
	pInterface->pBackward = NeuralNetLayerReluActivation_backward;
	pInterface->pUpdate = NeuralNetLayerActivation_update;								//super
	pInterface->pInitializeParameters = NeuralNetLayerActivation_initializeParameters;	//super
	pInterface->pGetParameters = NeuralNetLayerActivation_getParameters;				//super
}

//tanh
void
NeuralNetLayerTanhActivation_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerTanhActivation_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerTanhActivation_construct;
	pInterface->pGetShape = NeuralNetLayerActivation_getShape;							//super
	pInterface->pForward = NeuralNetLayerTanhActivation_forward;
	pInterface->pBackward = NeuralNetLayerTanhActivation_backward;
	pInterface->pUpdate = NeuralNetLayerActivation_update;								//super
	pInterface->pInitializeParameters = NeuralNetLayerActivation_initializeParameters;	//super
	pInterface->pGetParameters = NeuralNetLayerActivation_getParameters;				//super
}

//sigmoid
void
NeuralNetLayerSigmoidActivation_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerSigmoidActivation_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerSigmoidActivation_construct;
	pInterface->pGetShape = NeuralNetLayerActivation_getShape;							//super
	pInterface->pForward = NeuralNetLayerSigmoidActivation_forward;
	pInterface->pBackward = NeuralNetLayerSigmoidActivation_backward;
	pInterface->pUpdate = NeuralNetLayerActivation_update;								//super
	pInterface->pInitializeParameters = NeuralNetLayerActivation_initializeParameters;	//super
	pInterface->pGetParameters = NeuralNetLayerActivation_getParameters;				//super
}

//softmax
void
NeuralNetLayerSoftmaxActivation_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerSoftmaxActivation_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerSoftmaxActivation_construct;
	pInterface->pGetShape = NeuralNetLayerActivation_getShape;							//super
	pInterface->pForward = NeuralNetLayerSoftmaxActivation_forward;
	pInterface->pBackward = NeuralNetLayerSoftmaxActivation_backward;
	pInterface->pUpdate = NeuralNetLayerActivation_update;								//super
	pInterface->pInitializeParameters = NeuralNetLayerActivation_initializeParameters;	//super
	pInterface->pGetParameters = NeuralNetLayerActivation_getParameters;				//super
}

//helper
void
NeuralNetLayerActivation_getInterface(LayerFuncTable* pInterface, uint32_t* pLayerData) {
	ActivationNeuralNetHeader* pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pLayerData;
	switch (pActivationNeuralNetHeader->activation) {
	case NEURAL_NET_ACTIVATION_RELU:
		NeuralNetLayerReluActivation_getInterface(pInterface);
		break;
	case NEURAL_NET_ACTIVATION_TANH:
		NeuralNetLayerTanhActivation_getInterface(pInterface);
		break;
	case NEURAL_NET_ACTIVATION_SIGMOID:
		NeuralNetLayerSigmoidActivation_getInterface(pInterface);
		break;
	case NEURAL_NET_ACTIVATION_SOFTMAX:
		NeuralNetLayerSoftmaxActivation_getInterface(pInterface);
		break;
	}
}

//=====================================================================================
//  Activation層作成
//=====================================================================================
//relu
bool_t
NeuralNetLayerReluActivation_constructLayerData(
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
	ActivationNeuralNetHeader* pActivationNeuralNetHeader;
	ReluActivationNeuralNetHeader* pReluActivationNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(ReluActivationNeuralNetHeader), uint32_t);
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
		//header
		pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pActivationNeuralNetHeader->super, NET_LAYER_ACTIVATION, inHeight, inWidth, inChannel, sizeLayer);
		pActivationNeuralNetHeader->activation = NEURAL_NET_ACTIVATION_RELU;
		pReluActivationNeuralNetHeader = (ReluActivationNeuralNetHeader*)pActivationNeuralNetHeader;
		pReluActivationNeuralNetHeader->alpha = DEFAULT_RELU_ALPHA;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}

//tanh
bool_t
NeuralNetLayerTanhActivation_constructLayerData(
	uint32_t* pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t* pInputHeight,
	uint32_t* pInputWidth,
	uint32_t* pInputChannel,
	uint32_t* pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t* pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	ActivationNeuralNetHeader* pActivationNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(TanhActivationNeuralNetHeader), uint32_t);
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
		//header
		pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pActivationNeuralNetHeader->super, NET_LAYER_ACTIVATION, inHeight, inWidth, inChannel, sizeLayer);
		pActivationNeuralNetHeader->activation = NEURAL_NET_ACTIVATION_TANH;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = inHeight;
	*pInputWidth = inWidth;
	*pInputChannel = inChannel;
	return TRUE;
}

//sigmoid
bool_t
NeuralNetLayerSigmoidActivation_constructLayerData(
	uint32_t* pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t* pInputHeight,
	uint32_t* pInputWidth,
	uint32_t* pInputChannel,
	uint32_t* pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t* pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	ActivationNeuralNetHeader* pActivationNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(SigmoidActivationNeuralNetHeader), uint32_t);
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
		//header
		pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pActivationNeuralNetHeader->super, NET_LAYER_ACTIVATION, inHeight, inWidth, inChannel, sizeLayer);
		pActivationNeuralNetHeader->activation = NEURAL_NET_ACTIVATION_SIGMOID;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = inHeight;
	*pInputWidth = inWidth;
	*pInputChannel = inChannel;
	return TRUE;
}

//softmax
bool_t
NeuralNetLayerSoftmaxActivation_constructLayerData(
	uint32_t* pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t* pInputHeight,
	uint32_t* pInputWidth,
	uint32_t* pInputChannel,
	uint32_t* pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t* pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	ActivationNeuralNetHeader* pActivationNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(SoftmaxActivationNeuralNetHeader), uint32_t);
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
		//header
		pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pActivationNeuralNetHeader->super, NET_LAYER_ACTIVATION, inHeight, inWidth, inChannel, sizeLayer);
		pActivationNeuralNetHeader->activation = NEURAL_NET_ACTIVATION_SOFTMAX;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = inHeight;
	*pInputWidth = inWidth;
	*pInputChannel = inChannel;
	return TRUE;
}

//helper
bool_t
NeuralNetLayerActivation_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	NeuralNetActivationType	activation,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	switch (activation) {
	case NEURAL_NET_ACTIVATION_RELU:
		return NeuralNetLayerReluActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
		break;
	case NEURAL_NET_ACTIVATION_TANH:
		return NeuralNetLayerTanhActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
		break;
	case NEURAL_NET_ACTIVATION_SIGMOID:
		return NeuralNetLayerSigmoidActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
		break;
	case NEURAL_NET_ACTIVATION_SOFTMAX:
		return NeuralNetLayerSoftmaxActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
		break;
	}
	return FALSE;
}

//=====================================================================================
//  パラメタ設定
//=====================================================================================
//relu
bool_t
NeuralNetLayerReluActivation_setParameter(handle_t hLayer,flt32_t negativeSlope) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	ReluActivationNeuralNetHeader*	pReluActivationNeuralNetHeader = (ReluActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetActivationType type;
	NeuralNetLayerActivation_getType(hLayer,&type);
	if (type != NEURAL_NET_ACTIVATION_RELU) {
		return FALSE;
	}
	pReluActivationNeuralNetHeader->alpha = negativeSlope;
	return TRUE;
}

//=====================================================================================
//  活性化関数タイプを取得する
//=====================================================================================
bool_t
NeuralNetLayerActivation_getType(handle_t hLayer, NeuralNetActivationType* pType) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	ActivationNeuralNetHeader* pActivationNeuralNetHeader = (ActivationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pType == NULL ) {
		return FALSE;
	}
	*pType = pActivationNeuralNetHeader->activation;
	return TRUE;
}

