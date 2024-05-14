#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON				(0.001f)
#define DEFAULT_MOMENTUM	(0.99f)

#define INITAL_MEAN			(0.0f)
#define INITAL_INVVAR		(1.0f)

//=====================================================================================
//  BatchNormalization層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagBatchNormalizationNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		unit;
} BatchNormalizationNeuralNetHeader;

//=====================================================================================
//  BatchNormalization層構造体
//=====================================================================================
typedef struct tagBatchNormalizationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ：x
	flt32_t*		pSumOfX;		//誤差逆伝搬用データバッファ：x積算
	flt32_t*		pSumOfVar;		//誤差逆伝搬用データバッファ：(x - mean) * (x - mean)積算
	flt32_t			momentum;		//移動平均パラメタ
	uint32_t		accumulation;	//積算回数
	handle_t		hOptimizer;		//オプティマイザーハンドル
} BatchNormalizationNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
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
NeuralNetLayerBatchNormalization_forward_calculation(
	uint32_t	size,
	uint32_t	unit,
	flt32_t*	pGamma,
	flt32_t*	pBeta,
	flt32_t*	pMean,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pSumOfX,
	flt32_t*	pSumOfVar,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer) {
	uint32_t	i, j;
	flt32_t*	pGammaHead;
	flt32_t*	pBetaHead;
	flt32_t*	pMeanHead;
	flt32_t*	pInvStdHead;
	flt32_t		x;
	flt32_t		X;
	flt32_t		y;
	flt32_t		gamma;
	flt32_t		beta;
	flt32_t		mean;
	flt32_t		invStd;
	flt32_t*	pSumOfXHead;
	flt32_t*	pSumOfVarHead;
	i = size;
	while (i--) {
		//順伝搬データ
		pGammaHead = pGamma;
		pBetaHead = pBeta;
		pMeanHead = pMean;
		pInvStdHead = pInvStd;
		//逆伝播データ
		pSumOfXHead = pSumOfX;
		pSumOfVarHead = pSumOfVar;
		j = unit;
		while (j--) {
			x = *pInputBuffer++;
			gamma = *pGammaHead++;
			beta = *pBetaHead++;
			mean = *pMeanHead++;
			invStd = *pInvStdHead++;
			X = (x - mean) * invStd;	//正規化
			//順伝搬出力
			y = gamma * X + beta;
			*pOutputBuffer++ = y;
			//逆伝搬用保持
			if (pX != NULL) {
				*pX++ = X;				//gammaの微分値ととして保持：y = gamma * X + beta
				*pSumOfXHead++ += x;	//入力の平均値
				*pSumOfVarHead++ += (x - mean) * (x - mean);	//入力の分散値
			}
		}
	}
	return TRUE;
}

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_backward_calculation(
	uint32_t	size,
	uint32_t	unit,
	flt32_t*	pGamma,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pDGamma,
	flt32_t*	pDBeta,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer)
{
	uint32_t	i, j;
	flt32_t* pInput;
	flt32_t* pGammaHead;
	flt32_t* pInvStdHead;
	flt32_t* pDG;
	flt32_t* pDB;
	flt32_t* pDLossArray;
	//---------------------------------------------------------------------------------
	//パラメタの誤差逆伝搬
	//---------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	pInput = pInputBuffer;
	i = size;
	while (i--) {
		pGammaHead = pGamma;
		pInvStdHead = pInvStd;
		pDG = pDGamma;
		pDB = pDBeta;
		j = unit;
		while (j--) {
			//パラメタ微分値
			*pDG++ += (*pX++) * (*pDLossArray);
			*pDB++ += (*pDLossArray);
			//逆伝搬
			*pInput++ += (*pGammaHead++) * (*pInvStdHead++) * (*pDLossArray);
			//逆伝入力値伝搬誤差微分値ポインタ更新
			pDLossArray++;
		}
	}
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	flt32_t*	pX;
	flt32_t*	pSumOfX;
	flt32_t*	pSumOfVar;
	uint32_t	size;
	uint32_t	unit;
	//---------------------------------------------------------------------------------
	//重みパラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header分だけ異動
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pBatchNormalizationLayer->pX == NULL) {
			return FALSE;
		}
		if (pBatchNormalizationLayer->pSumOfX == NULL) {
			return FALSE;
		}
		if (pBatchNormalizationLayer->pSumOfVar == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//入力、入力積算、分散積算
		//-------------------------------------------------------
		pX			= pBatchNormalizationLayer->pX;
		pSumOfX		= pBatchNormalizationLayer->pSumOfX;
		pSumOfVar	= pBatchNormalizationLayer->pSumOfVar;
	}
	else {
		pX			= NULL;
		pSumOfX		= NULL;
		pSumOfVar	= NULL;
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
	if (pPropagationInfo->dataShape.channel == pBatchNormalizationNeuralNetHeader->unit) {
		//channel方向：通常
		size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	else {
		//width方向：Dense後など
		if (pBatchNormalizationNeuralNetHeader->unit != (pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel)) {
			return FALSE;
		}
		size = pPropagationInfo->dataShape.height;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	//---------------------------------------------------------------------------------
	//正規化処理：channel方向にかける場合とwidth方向にかける場合がある
	//---------------------------------------------------------------------------------
	NeuralNetLayerBatchNormalization_forward_calculation(size,unit,pGamma,pBeta,pMean,pInvStd,pX,pSumOfX,pSumOfVar,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	if (pX != NULL) {
		//逆伝搬用積算回数
		pBatchNormalizationLayer->accumulation += size;
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	//変更なし
	return TRUE;
}

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	flt32_t*	pDGamma;
	flt32_t*	pDBeta;
	flt32_t*	pInput;
	uint32_t	size;
	uint32_t	unit;
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
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header分だけ異動
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//インターフェイス取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pBatchNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得：パラメタは連続して入っている
	//---------------------------------------------------------------------------------
	pDGamma = optimizerFunctionTable.pGetDeltaParameterBuffer(pBatchNormalizationLayer->hOptimizer);
	pDBeta = pDGamma + pBatchNormalizationNeuralNetHeader->unit;
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
	//正規化処理の対象データ
	//---------------------------------------------------------------------------------
	if (inChannel == pBatchNormalizationNeuralNetHeader->unit) {
		//channel方向：通常
		size = inHeight * inWidth;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	else {
		//width方向：Dense後など
		if (pBatchNormalizationNeuralNetHeader->unit != (inWidth * inChannel)) {
			return FALSE;
		}
		size = inHeight;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬計算
	//---------------------------------------------------------------------------------
	NeuralNetLayerBatchNormalization_backward_calculation(size,unit,pGamma, pInvStd, pBatchNormalizationLayer->pX,pDGamma,pDBeta,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//出力（入力方向）データサイズ形状
	//---------------------------------------------------------------------------------
	//変更なし
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_update(handle_t hLayer) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam;
	flt32_t* pGamma;
	flt32_t* pBeta;
	flt32_t* pMean;
	flt32_t* pInvStd;
	flt32_t factor;
	flt32_t currentVar;
	flt32_t measuredVar;
	flt32_t measuredMean;
	flt32_t sigma;
	flt32_t invStd;
	uint32_t i;
	OptimizerFunctionTable	optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//インターフェイス取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pBatchNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//重みパラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//層学習パラメタ更新：Gamma&Beta
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pBatchNormalizationLayer->hOptimizer, pGamma);
	factor = 1.0f / (flt32_t)pBatchNormalizationLayer->accumulation;
	//---------------------------------------------------------------------------------
	//非学習パラメタ：測定パラメタ
	//---------------------------------------------------------------------------------
	//平均
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		//計測値
		measuredMean = pBatchNormalizationLayer->pSumOfX[i] * factor;
		//計測値で更新
		pMean[i] = pMean[i] * pBatchNormalizationLayer->momentum + measuredMean * (1.0f - pBatchNormalizationLayer->momentum);
		//累積バッファ初期化
		pBatchNormalizationLayer->pSumOfX[i] = 0.0f;
	}
	//分散の逆数
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		//計測値
		measuredVar = pBatchNormalizationLayer->pSumOfVar[i] * factor;
#if 0
		//この実装では、momentumが√の中にある場合と更新スピードが異なる
		//性能が劣化する場合があった。
		sigma = low_cost_sqrt(measuredVar + EPSILON, 2);
		if (sigma > 0.0f) {
			invStd = 1.0f / sigma;
		}
		else {
			invStd = 1.0f;
		}
		pInvStd[i] = pInvStd[i] * pBatchNormalizationLayer->momentum + invStd * (1.0f - pBatchNormalizationLayer->momentum);
#else
		//---------------------------------------------------------
		//Varの移動平均：論文内容に一致
		//pInvStd[i]=1.0/√(Var+EPSILON) ※Var =　(x-mean)*(x-mean)
		//---------------------------------------------------------
		//現行値
		if (pInvStd[i] > 0.0f) {
			currentVar = pInvStd[i];  
		}
		else {
			currentVar = 1.0f;
		}
		currentVar = 1.0f / currentVar;
		currentVar = currentVar * currentVar - EPSILON;
		//計測値で更新
		currentVar = currentVar * pBatchNormalizationLayer->momentum + measuredVar * (1.0f - pBatchNormalizationLayer->momentum);
		sigma = low_cost_sqrt(currentVar + EPSILON, 2);
		if (sigma > 0.0f) {
			invStd = 1.0f / sigma;
		}
		else {
			invStd = 1.0f;
		}
		pInvStd[i] = invStd;
#endif
		//累積バッファ初期化
		pBatchNormalizationLayer->pSumOfVar[i] = 0.0f;
	}
	//積算カウンタ初期化
	pBatchNormalizationLayer->accumulation = 0;
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t*	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	uint32_t	paramSize;
	uint32_t	i;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header分だけ異動
	//Gamma
	pGamma = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//Beta
	pBeta = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//Mean
	pMean = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//invStd
	pInvStd = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//層パラメタ初期化
	//---------------------------------------------------------------------------------
	//pGamma
	paramSize	= pBatchNormalizationNeuralNetHeader->unit;
	set_constant_initial_values(pGamma, paramSize, 1.0f);
	//pBeta
	paramSize = pBatchNormalizationNeuralNetHeader->unit;
	set_constant_initial_values(pBeta, paramSize, 0.0f);
	//pMean,pInvStd
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		pMean[i]	= INITAL_MEAN;
		pInvStd[i]	= INITAL_INVVAR;
	}
	return TRUE;
}

//=====================================================================================
//  更新パラメタ設定
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_setMomentum(handle_t hLayer, flt32_t momentum)
{
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NetLayerType	layerType;
	bool_t			fStatus;
	fStatus = NeuralNetLayer_getType(hLayer,&layerType);
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (layerType != NET_LAYER_BATCH_NORMALIZATION) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ範囲チェック
	//---------------------------------------------------------------------------------
	if (momentum < 0.0f) {
		momentum = 0.0f;
	}
	if (momentum > 1.0f) {
		momentum = 1.0f;
	}
	pBatchNormalizationLayer->momentum = momentum;
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t inputDataDim;
	if (pBatchNormalizationNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pBatchNormalizationNeuralNetHeader->unit;	//Gamma
		*pNumberOfLearningParameters += pBatchNormalizationNeuralNetHeader->unit;	//Beta
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(BatchNormalizationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			//X
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//SumX,SumVar
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit * 2, uint32_t);
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
NeuralNetLayerBatchNormalization_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
#if 0
		//学習パラメタのみ
		NeuralNetLayerBatchNormalization_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
#else
		//学習パラメタ
		*pNumberOfParameters = pBatchNormalizationNeuralNetHeader->unit;	//Gamma
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//Beta
		//学習時統計（測定）パラメタ
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//Mean
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//invStd
#endif
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerBatchNormalization_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	uint32_t parameterSize;
	NeuralNetLayerBatchNormalization_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerBatchNormalization_getInterface(&funcTable);
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
			pObjectWork += size_in_type(sizeof(BatchNormalizationNeuralNetLayer), uint32_t);
			//学習用バッファ
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			pBatchNormalizationLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			pBatchNormalizationLayer->pSumOfX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
			pBatchNormalizationLayer->pSumOfVar = (flt32_t*)pObjectWork;
			//メンバ初期化
			pBatchNormalizationLayer->momentum			= DEFAULT_MOMENTUM;
			pBatchNormalizationLayer->accumulation		= 0;
			pBatchNormalizationLayer->hOptimizer		= hOptimizer;
			//バッファ初期化
			i = inputDataDim;
			while (i--) {
				pBatchNormalizationLayer->pX[i] = 0.0f;
			}
			i = pBatchNormalizationNeuralNetHeader->unit;
			while (i--) {
				pBatchNormalizationLayer->pSumOfX[i] = 0.0f;
				pBatchNormalizationLayer->pSumOfVar[i] = 0.0f;
			}
		}
		else {
			//学習用バッファ
			pBatchNormalizationLayer->pX = NULL;
			pBatchNormalizationLayer->pSumOfX = NULL;
			pBatchNormalizationLayer->pSumOfVar = NULL;
			//メンバ初期化
			pBatchNormalizationLayer->momentum			= DEFAULT_MOMENTUM;
			pBatchNormalizationLayer->accumulation		= 0;
			pBatchNormalizationLayer->hOptimizer		= hOptimizer;
		}
		return (handle_t)pBatchNormalizationLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerBatchNormalization_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerBatchNormalization_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerBatchNormalization_construct;
	pInterface->pGetShape = NeuralNetLayerBatchNormalization_getShape;
	pInterface->pForward = NeuralNetLayerBatchNormalization_forward;
	pInterface->pBackward = NeuralNetLayerBatchNormalization_backward;
	pInterface->pUpdate = NeuralNetLayerBatchNormalization_update;
	pInterface->pInitializeParameters = NeuralNetLayerBatchNormalization_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerBatchNormalization_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_constructLayerData(
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
	uint32_t	sizeOfParamMean;
	uint32_t	sizeOfParamVar;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	unit;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader;
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
	//Normalizationをかける方向
	//---------------------------------------------------------------------------------
	if (inChannel > 1) {
		unit = inChannel;	//in channel direction
	}
	else {
		unit = inWidth;		//in width direction
	}
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	sizeOfParamGamma = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamBeta = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamMean = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamVar = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamGamma + sizeOfParamBeta + sizeOfParamMean + sizeOfParamVar;
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
		pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pBatchNormalizationNeuralNetHeader->super, NET_LAYER_BATCH_NORMALIZATION, inHeight, inWidth, inChannel, sizeLayer);
		pBatchNormalizationNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//Gamma
		pLayer += sizeOfParamGamma;
		//Beta
		pLayer += sizeOfParamBeta;
		//Mean
		pLayer += sizeOfParamMean;
		//Var
		pLayer += sizeOfParamVar;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
