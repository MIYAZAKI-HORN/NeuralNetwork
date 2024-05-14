#include "STDTypeDefinition.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"
#include "NeuralNetLayerFunction.h"

//=====================================================================================
//  Optimizerベースクラス
//=====================================================================================
typedef struct tagNeuralNetworkOptimizer {
	NeuralNetOptimizerType	type;
	uint32_t				parameterSize;
	uint32_t				batchSize;
	flt32_t*				pDParam;
	OptimizerFunctionTable	funcTable;
} NeuralNetOptimizer;

//=====================================================================================
//  SGDクラス
//=====================================================================================
typedef struct tagNeuralNetworkOptimizerSGD {
	NeuralNetOptimizer	base;
	flt32_t*			pV;
	flt32_t				momentum;
	flt32_t				lr;
} NeuralNetworkOptimizerSGD;

//=====================================================================================
//  RMSpropクラス
//=====================================================================================
typedef struct tagNeuralNetworkOptimizerRMSprop {
	NeuralNetOptimizer	base;
	flt32_t*			pH;
	flt32_t				rho;
	flt32_t				lr;
} NeuralNetworkOptimizerRMSprop;

//=====================================================================================
//  Adamクラス
//=====================================================================================
typedef struct tagNeuralNetworkOptimizerAdam {
	NeuralNetOptimizer	base;
	flt32_t*			pM;
	flt32_t*			pV;
	flt32_t				beta_1;
	flt32_t				beta_2;
	flt32_t				lr;
	flt32_t				beta_1t;
	flt32_t				beta_2t;
} NeuralNetworkOptimizerAdam;

#define EPSILON (1e-08f)
#define SQRT_ITERATIONS (3)

//=====================================================================================
// 
//  optimizer constructer
// 
//=====================================================================================
static
void
NeuralNetworkNeuralNetOptimizer_construct(
	NeuralNetOptimizer*		pOptimizer,
	NeuralNetOptimizerType	type,
	uint32_t				parameterSize, 
	uint32_t				batchSize,
	flt32_t*				pDParam) {
	pOptimizer->type			= type;
	pOptimizer->parameterSize	= parameterSize;
	pOptimizer->batchSize		= batchSize;
	pOptimizer->pDParam			= pDParam;
	//funcTable
	pOptimizer->funcTable.pGetSizeIn32BitWord			= NULL;
	pOptimizer->funcTable.pConstruct					= NULL;
	pOptimizer->funcTable.pGetType						= NULL;
	pOptimizer->funcTable.pGetParameterSize				= NULL;
	pOptimizer->funcTable.pGetDeltaParameterBuffer		= NULL;
	pOptimizer->funcTable.pUpdate						= NULL;
}

//=====================================================================================
//  タイプを取得する
//=====================================================================================
static
NeuralNetOptimizerType
NeuralNetworkNeuralNetOptimizer_getType(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	if (This == NULL) {
		return NEURAL_NET_OPTIMIZER_UNDEFINED;
	}
	return This->type;
}

//=====================================================================================
//  パラメタサイズを取得する
//=====================================================================================
static
uint32_t
NeuralNetworkNeuralNetOptimizer_getParameterSize(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	if (This == NULL) {
		return 0;
	}
	return This->parameterSize;
}

//=====================================================================================
//  微分値バッファーのポインタを取得する
//=====================================================================================
static
flt32_t*
NeuralNetworkNeuralNetOptimizer_getDeltaParameterBuffer(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	if (This == NULL) {
		return NULL;
	}
	return This->pDParam;
}

//=====================================================================================
// 
//  SGD 
// 
//=====================================================================================
static void		NeuralNetworkOptimizerSGD_initialize(handle_t hOptimizer);
static bool_t	NeuralNetworkOptimizerSGC_getInterface(OptimizerFunctionTable* pInterface);

//=====================================================================================
//  ワークエリアサイズ取得
//=====================================================================================
static
uint32_t
NeuralNetworkOptimizerSGD_getSizeIn32BitWord(uint32_t parameterSize) {
	uint32_t	workAreaSizeIn32BitWord = 0;
	workAreaSizeIn32BitWord = size_in_type(sizeof(NeuralNetworkOptimizerSGD), uint32_t);
	workAreaSizeIn32BitWord += size_in_type(sizeof(flt32_t) * parameterSize * 2, uint32_t); // parameter + V
	return workAreaSizeIn32BitWord;
}

//=====================================================================================
//  エンジン構築
//=====================================================================================
static
handle_t
NeuralNetworkOptimizerSGD_construct(uint32_t parameterSize, uint32_t batchSize, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord) {
	uint32_t	requiredSizeOfWorkAreaIn32BitWord;
	handle_t	hOptimizer;
	NeuralNetworkOptimizerSGD*  pSGD;
	//----------------------------------------------------------------------------------
	//  ワークエリアチェック
	//----------------------------------------------------------------------------------
	requiredSizeOfWorkAreaIn32BitWord = NeuralNetworkOptimizerSGD_getSizeIn32BitWord(parameterSize);
	if (requiredSizeOfWorkAreaIn32BitWord == 0) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//  エンジン
	//----------------------------------------------------------------------------------
	pSGD = (NeuralNetworkOptimizerSGD*)pWorkArea;
	pWorkArea += size_in_type(sizeof(NeuralNetworkOptimizerSGD), uint32_t);
	//----------------------------------------------------------------------------------
	//  ベースクラス情報セット
	//----------------------------------------------------------------------------------
	NeuralNetworkNeuralNetOptimizer_construct((NeuralNetOptimizer*)pSGD, NEURAL_NET_OPTIMIZER_SGD, parameterSize, batchSize, (flt32_t*)pWorkArea);
	pWorkArea += size_in_type(sizeof(flt32_t) * parameterSize, uint32_t);
	//----------------------------------------------------------------------------------
	//  バッファおよび学習用パラメタ
	//----------------------------------------------------------------------------------
	pSGD->pV		= (flt32_t*)pWorkArea;
	pSGD->momentum	= 0.9f;
	pSGD->lr		= 0.01f;
	//----------------------------------------------------------------------------------
	//インターフェイス設定
	//----------------------------------------------------------------------------------
	NeuralNetworkOptimizerSGC_getInterface(&pSGD->base.funcTable);
	//----------------------------------------------------------------------------------
	//  ハンドルセットと初期化
	//----------------------------------------------------------------------------------
	hOptimizer = pSGD;
	NeuralNetworkOptimizerSGD_initialize(hOptimizer);
	return hOptimizer;
}

//=====================================================================================
//  バッファの初期化
//=====================================================================================
static
void
NeuralNetworkOptimizerSGD_initialize(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerSGD* pSGD = (NeuralNetworkOptimizerSGD*)hOptimizer;
	flt32_t*    pParam	= This->pDParam;
	flt32_t*    pV		= pSGD->pV;
	uint32_t    i		= This->parameterSize;
	while (i--) {
		*pParam++	= 0.0f;
		*pV++		= 0.0f;
	}
}

//=====================================================================================
//  パラメタの更新
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerSGD_update(handle_t hOptimizer,flt32_t* pParameterToUpdate) {
	uint32_t	i;
	flt32_t*	pV;
	flt32_t*	pParameter;
	flt32_t		averageDParam;
	flt32_t*	pDP;
	flt32_t		factor;
	flt32_t		deltaParameter;
	NeuralNetOptimizer*			This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerSGD*  pSGD = (NeuralNetworkOptimizerSGD*)hOptimizer;
	if (This == NULL) {
		return FALSE;
	}
	if (This->batchSize == 0) {
		return FALSE;
	}
	pV              = pSGD->pV;
	pParameter      = pParameterToUpdate;
	factor          = 1.0f / (flt32_t)This->batchSize;
	pDP             = This->pDParam;
	i				= This->parameterSize;
	while (i--) {
		//---------------------------------------------------------------------------------
		// 微分値平均
		//---------------------------------------------------------------------------------
		averageDParam = *pDP++;
		averageDParam *= factor;
		//---------------------------------------------------------------------------------
		// V計算
		//---------------------------------------------------------------------------------
		*pV = (*pV) * pSGD->momentum - averageDParam * pSGD->lr;
		//---------------------------------------------------------------------------------
		// パラメタ更新
		//---------------------------------------------------------------------------------
		deltaParameter = *pV;
		if (deltaParameter > EPSILON || deltaParameter < (-EPSILON)) {
			*pParameter += deltaParameter; //加算
		}
		//---------------------------------------------------------------------------------
		// vとパラメタのポインタを進める
		//---------------------------------------------------------------------------------
		pParameter++;
		pV++;
	}
	//---------------------------------------------------------------------------------
	// バッファー初期化
	//---------------------------------------------------------------------------------
	pDP = This->pDParam;
	i	= This->parameterSize;
	while (i--) {
		*pDP++ = 0.0f;
	}    
	return TRUE;
}

//=====================================================================================
//  インターフェイス設定
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerSGC_getInterface(OptimizerFunctionTable* pInterface) {
	//construct optimizer
	pInterface->pGetSizeIn32BitWord			= NeuralNetworkOptimizerSGD_getSizeIn32BitWord;
	pInterface->pConstruct					= NeuralNetworkOptimizerSGD_construct;
	//base class information
	pInterface->pGetType					= NeuralNetworkNeuralNetOptimizer_getType;
	pInterface->pGetParameterSize			= NeuralNetworkNeuralNetOptimizer_getParameterSize;
	pInterface->pGetDeltaParameterBuffer	= NeuralNetworkNeuralNetOptimizer_getDeltaParameterBuffer;
	//independent function
	pInterface->pUpdate						= NeuralNetworkOptimizerSGD_update;
	return TRUE;
}

//=====================================================================================
//  パラメタセット
//=====================================================================================
bool_t
NeuralNetworkOptimizerSGD_setParameters(handle_t hOptimizer, flt32_t momentum, flt32_t lr) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerSGD* pSGD = (NeuralNetworkOptimizerSGD*)hOptimizer;
	if (pSGD == NULL) {
		return FALSE;
	}
	if (This->type == NEURAL_NET_OPTIMIZER_SGD) {
		pSGD->momentum = momentum;
		pSGD->lr = lr;
	}
	else {
		return FALSE;
	}
	return TRUE;
}

//=====================================================================================
// 
//  RMSprop 
// 
//=====================================================================================
static void		NeuralNetworkOptimizerRMSprop_initialize(handle_t hOptimizer);
static bool_t	NeuralNetworkOptimizerRMSprop_getInterface(OptimizerFunctionTable* pInterface);

//=====================================================================================
//  必要ワークエリアサイズ取得
//=====================================================================================
static
uint32_t
NeuralNetworkOptimizerRMSprop_getSizeIn32BitWord(uint32_t parameterSize) {
	uint32_t	workAreaSizeIn32BitWord = 0;
	workAreaSizeIn32BitWord = size_in_type(sizeof(NeuralNetworkOptimizerRMSprop), uint32_t);
	workAreaSizeIn32BitWord += size_in_type(sizeof(flt32_t) * parameterSize * 2, uint32_t); // parameter + H
	return workAreaSizeIn32BitWord;
}

//=====================================================================================
//  エンジン構築
//=====================================================================================
static
handle_t
NeuralNetworkOptimizerRMSprop_construct(uint32_t parameterSize, uint32_t batchSize, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord) {
	uint32_t	requiredSizeOfWorkAreaIn32BitWord;
	handle_t	hOptimizer;
	NeuralNetworkOptimizerRMSprop*  pRMSprop;
	//----------------------------------------------------------------------------------
	//  ワークエリアチェック
	//----------------------------------------------------------------------------------
	requiredSizeOfWorkAreaIn32BitWord = NeuralNetworkOptimizerRMSprop_getSizeIn32BitWord(parameterSize);
	if (requiredSizeOfWorkAreaIn32BitWord == 0) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//  エンジン
	//----------------------------------------------------------------------------------
	pRMSprop = (NeuralNetworkOptimizerRMSprop*)pWorkArea;
	pWorkArea += size_in_type(sizeof(NeuralNetworkOptimizerRMSprop), uint32_t);
	//----------------------------------------------------------------------------------
	//  ベースクラス情報セット
	//----------------------------------------------------------------------------------
	NeuralNetworkNeuralNetOptimizer_construct((NeuralNetOptimizer*)pRMSprop, NEURAL_NET_OPTIMIZER_RMSPROP, parameterSize, batchSize, (flt32_t*)pWorkArea);
	pWorkArea += size_in_type(sizeof(flt32_t) * parameterSize, uint32_t);
	//----------------------------------------------------------------------------------
	//  バッファおよび学習用パラメタ
	//----------------------------------------------------------------------------------
	pRMSprop->pH	= (flt32_t*)pWorkArea;
	pRMSprop->rho	= 0.9f;
	pRMSprop->lr	= 0.001f;
	//----------------------------------------------------------------------------------
	//インターフェイス設定
	//----------------------------------------------------------------------------------
	NeuralNetworkOptimizerRMSprop_getInterface(&pRMSprop->base.funcTable);
	//----------------------------------------------------------------------------------
	//  ハンドルセットと初期化
	//----------------------------------------------------------------------------------
	hOptimizer = pRMSprop;
	NeuralNetworkOptimizerRMSprop_initialize(hOptimizer);
	return hOptimizer;
}

//=====================================================================================
//  バッファの初期化
//=====================================================================================
static
void
NeuralNetworkOptimizerRMSprop_initialize(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerRMSprop* pRMSprop = (NeuralNetworkOptimizerRMSprop*)hOptimizer;
	flt32_t*	pParam	= This->pDParam;
	flt32_t*	pH		= pRMSprop->pH;
	uint32_t	i		= This->parameterSize;
	while (i--) {
		*pParam++	= 0.0f;
		*pH++		= 0.0f;
	}
}

//=====================================================================================
//  パラメタの更新
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerRMSprop_update(handle_t hOptimizer, flt32_t* pParameterToUpdate) {
	uint32_t	i;
	flt32_t*	pParameter;
	flt32_t		averageDParam;
	flt32_t*	pDP;
	flt32_t*	pH;
	flt32_t		factor;
	flt32_t		sqrt_value;
	flt32_t		effectiveLearningRatio;
	flt32_t		deltaParameter;
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerRMSprop* pRMSprop = (NeuralNetworkOptimizerRMSprop*)hOptimizer;
	if (This == NULL) {
		return FALSE;
	}
	if (This->batchSize == 0) {
		return FALSE;
	}
	pH = pRMSprop->pH;
	pParameter = pParameterToUpdate;
	factor = 1.0f / (flt32_t)This->batchSize;
	pDP = This->pDParam;
	i = This->parameterSize;
	while(i--) {
		//---------------------------------------------------------------------------------
		// 微分値平均
		//---------------------------------------------------------------------------------
		averageDParam = *pDP++;
		averageDParam *= factor;
		//---------------------------------------------------------------------------------
		// 実行学習率
		//---------------------------------------------------------------------------------
		*pH = (*pH) * pRMSprop->rho + (1.0f - pRMSprop->rho) * averageDParam * averageDParam;
		sqrt_value = low_cost_sqrt(*pH, SQRT_ITERATIONS);
		effectiveLearningRatio = pRMSprop->lr / (sqrt_value + EPSILON);
		//---------------------------------------------------------------------------------
		// パラメタappude-to 
		//---------------------------------------------------------------------------------
		deltaParameter = averageDParam * effectiveLearningRatio;
		if (deltaParameter > EPSILON || deltaParameter < (-EPSILON)) {
			*pParameter -= deltaParameter;
		}
		//---------------------------------------------------------------------------------
		// hとパラメタのポインタを進める
		//---------------------------------------------------------------------------------
		pH++;
		pParameter++;
	}
	//---------------------------------------------------------------------------------
	// バッファー初期化
	//---------------------------------------------------------------------------------
	pDP = This->pDParam;
	i	= This->parameterSize;
	while (i--) {
		*pDP++ = 0.0f;
	}
	return TRUE;
}

//=====================================================================================
//  インターフェイス設定
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerRMSprop_getInterface(OptimizerFunctionTable* pInterface) {
	//construct optimizer
	pInterface->pGetSizeIn32BitWord			= NeuralNetworkOptimizerRMSprop_getSizeIn32BitWord;
	pInterface->pConstruct					= NeuralNetworkOptimizerRMSprop_construct;
	//base class information
	pInterface->pGetType					= NeuralNetworkNeuralNetOptimizer_getType;
	pInterface->pGetParameterSize			= NeuralNetworkNeuralNetOptimizer_getParameterSize;
	pInterface->pGetDeltaParameterBuffer	= NeuralNetworkNeuralNetOptimizer_getDeltaParameterBuffer;
	//independent function
	pInterface->pUpdate						= NeuralNetworkOptimizerRMSprop_update;
	return TRUE;
}

//=====================================================================================
//  パラメタセット
//=====================================================================================
bool_t
NeuralNetworkOptimizerRMSprop_setParameters(handle_t hOptimizer, flt32_t rho, flt32_t lr) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerRMSprop* pRMSprop = (NeuralNetworkOptimizerRMSprop*)hOptimizer;
	if (pRMSprop == NULL) {
		return FALSE;
	}
	if (This->type == NEURAL_NET_OPTIMIZER_RMSPROP) {
		pRMSprop->rho	= rho;
		pRMSprop->lr	= lr;
	}
	else {
		return FALSE;
	}
	return TRUE;
}

//=====================================================================================
// 
//  Optimizer Adam 
// 
//=====================================================================================
static void		NeuralNetworkOptimizerAdam_initialize(handle_t hOptimizer);
static bool_t	NeuralNetworkOptimizerAdam_getInterface(OptimizerFunctionTable* pInterface);

//=====================================================================================
//  必要ワークエリアサイズ取得
//=====================================================================================
static
uint32_t
NeuralNetworkOptimizerAdam_getSizeIn32BitWord(uint32_t parameterSize) {
	uint32_t	workAreaSizeIn32BitWord = 0;
	workAreaSizeIn32BitWord = size_in_type(sizeof(NeuralNetworkOptimizerAdam), uint32_t);
	workAreaSizeIn32BitWord += size_in_type(sizeof(flt32_t) * parameterSize * 3, uint32_t); // parameter + M,V
	return workAreaSizeIn32BitWord;
}

//=====================================================================================
//  エンジン構築
//=====================================================================================
static
handle_t
NeuralNetworkOptimizerAdam_construct(uint32_t parameterSize, uint32_t batchSize, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord) {
	uint32_t        requiredSizeOfWorkAreaIn32BitWord;
	handle_t    hOptimizer;
	NeuralNetworkOptimizerAdam* pAdam;
	//----------------------------------------------------------------------------------
	//  ワークエリアチェック
	//----------------------------------------------------------------------------------
	requiredSizeOfWorkAreaIn32BitWord = NeuralNetworkOptimizerAdam_getSizeIn32BitWord(parameterSize);
	if (requiredSizeOfWorkAreaIn32BitWord == 0) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//  エンジン
	//----------------------------------------------------------------------------------
	pAdam = (NeuralNetworkOptimizerAdam*)pWorkArea;
	pWorkArea += size_in_type(sizeof(NeuralNetworkOptimizerAdam), uint32_t);
	//----------------------------------------------------------------------------------
	//  ベースクラス情報セット
	//----------------------------------------------------------------------------------
	NeuralNetworkNeuralNetOptimizer_construct((NeuralNetOptimizer*)pAdam, NEURAL_NET_OPTIMIZER_ADAM, parameterSize, batchSize, (flt32_t*)pWorkArea);
	pWorkArea += size_in_type(sizeof(flt32_t) * parameterSize, uint32_t);
	//----------------------------------------------------------------------------------
	//  バッファおよび学習用パラメタ
	//----------------------------------------------------------------------------------
	pAdam->pM		= (flt32_t*)pWorkArea;
	pWorkArea		+= size_in_type(sizeof(flt32_t) * parameterSize, uint32_t);	//バッファを進める
	pAdam->pV		= (flt32_t*)pWorkArea;
	pAdam->beta_1	= 0.9f;
	pAdam->beta_2	= 0.999f;
	pAdam->lr		= 0.001f;
	pAdam->beta_1t	= pAdam->beta_1;	//t乗
	pAdam->beta_2t	= pAdam->beta_2;	//t乗
	//----------------------------------------------------------------------------------
	//インターフェイス設定
	//----------------------------------------------------------------------------------
	NeuralNetworkOptimizerAdam_getInterface(&pAdam->base.funcTable);
	//----------------------------------------------------------------------------------
	//  ハンドルセットと初期化
	//----------------------------------------------------------------------------------
	hOptimizer = pAdam;
	NeuralNetworkOptimizerAdam_initialize(hOptimizer);
	return hOptimizer;
}

//=====================================================================================
//  バッファの初期化
//=====================================================================================
static
void
NeuralNetworkOptimizerAdam_initialize(handle_t hOptimizer) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerAdam* pAdam = (NeuralNetworkOptimizerAdam*)hOptimizer;
	flt32_t*	pParam	= This->pDParam;
	flt32_t*	pM		= pAdam->pM;
	flt32_t*	pV		= pAdam->pV;
	uint32_t	i		= This->parameterSize;
	while (i--) {
		*pParam++	= 0.0f;
		*pM++		= 0.0f;
		*pV++		= 0.0f;
	}
}

//=====================================================================================
//  パラメタの更新
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerAdam_update(handle_t hOptimizer, flt32_t* pParameterToUpdate) {
	uint32_t	i;
	flt32_t*	pParameter;
	flt32_t		averageDParam;
	flt32_t*	pDP;
	flt32_t*	pM;
	flt32_t*	pV;
	flt32_t		factor;
	flt32_t		numerator;      // 分子
	flt32_t		denominator;    // 分母
	flt32_t		deltaParameter;
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerAdam* pAdam = (NeuralNetworkOptimizerAdam*)hOptimizer;
	if (This == NULL) {
		return FALSE;
	}
	if (This->batchSize == 0) {
		return FALSE;
	}
	pM = pAdam->pM;
	pV = pAdam->pV;
	pParameter = pParameterToUpdate;
	factor = 1.0f / (flt32_t)This->batchSize;
	pDP = This->pDParam;
	i = This->parameterSize;
	while (i--) {
		//---------------------------------------------------------------------------------
		// 微分値平均
		//---------------------------------------------------------------------------------
		averageDParam = *pDP++;
		averageDParam *= factor;
		//---------------------------------------------------------------------------------
		// 実行学習率
		//---------------------------------------------------------------------------------
		*pM = (*pM) * pAdam->beta_1 + (1.0f - pAdam->beta_1) * averageDParam;
		*pV = (*pV) * pAdam->beta_2 + (1.0f - pAdam->beta_2) * averageDParam * averageDParam;
		numerator = pAdam->lr * low_cost_sqrt(1.0f-pAdam->beta_2t, SQRT_ITERATIONS) * (*pM);
		denominator = (1.0f - pAdam->beta_1t) * (low_cost_sqrt(*pV, SQRT_ITERATIONS) + EPSILON);
		//---------------------------------------------------------------------------------
		// パラメタ更新
		//---------------------------------------------------------------------------------
		deltaParameter = numerator / denominator;
		if (deltaParameter > EPSILON || deltaParameter < (-EPSILON)) {
			*pParameter -= deltaParameter;
		}
		//---------------------------------------------------------------------------------
		// ポインタを進める
		//---------------------------------------------------------------------------------
		pM++;
		pV++;
		pParameter++;
	}
	//---------------------------------------------------------------------------------
	// バッファー初期化
	//---------------------------------------------------------------------------------
	pDP = This->pDParam;
	i = This->parameterSize;
	while(i--) {
		*pDP++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	// betaべき乗
	//---------------------------------------------------------------------------------
	pAdam->beta_1t *= pAdam->beta_1;
	pAdam->beta_2t *= pAdam->beta_2;
	return TRUE;
}

//=====================================================================================
//  インターフェイス設定
//=====================================================================================
static
bool_t
NeuralNetworkOptimizerAdam_getInterface(OptimizerFunctionTable* pInterface) {
	//construct optimizer
	pInterface->pGetSizeIn32BitWord			= NeuralNetworkOptimizerAdam_getSizeIn32BitWord;
	pInterface->pConstruct					= NeuralNetworkOptimizerAdam_construct;
	//base class information
	pInterface->pGetType					= NeuralNetworkNeuralNetOptimizer_getType;
	pInterface->pGetParameterSize			= NeuralNetworkNeuralNetOptimizer_getParameterSize;
	pInterface->pGetDeltaParameterBuffer	= NeuralNetworkNeuralNetOptimizer_getDeltaParameterBuffer;
	//independent function
	pInterface->pUpdate						= NeuralNetworkOptimizerAdam_update;
	return TRUE;
}

//=====================================================================================
//  パラメタセット
//=====================================================================================
bool_t
NeuralNetworkOptimizerAdam_setParameters(handle_t hOptimizer, flt32_t beta_1, flt32_t beta_2, flt32_t lr) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	NeuralNetworkOptimizerAdam* pAdam = (NeuralNetworkOptimizerAdam*)hOptimizer;
	if (pAdam == NULL) {
		return FALSE;
	}
	if (This->type == NEURAL_NET_OPTIMIZER_ADAM) {
		pAdam->beta_1	= beta_1;
		pAdam->beta_2	= beta_2;
		pAdam->lr		= lr;
	}
	else {
		return FALSE;
	}
	return TRUE;
}

bool_t
NeuralNetOptimizer_getInterfaceByType(NeuralNetOptimizerType type, OptimizerFunctionTable* pInterface) {
	bool_t fStatus = FALSE;
	switch (type) {
	case NEURAL_NET_OPTIMIZER_SGD:
		NeuralNetworkOptimizerSGC_getInterface(pInterface);
		break;
	case NEURAL_NET_OPTIMIZER_RMSPROP:
		NeuralNetworkOptimizerRMSprop_getInterface(pInterface);
		break;
	case NEURAL_NET_OPTIMIZER_ADAM:
		NeuralNetworkOptimizerAdam_getInterface(pInterface);
		break;
	default:
		return FALSE;
	}
	return TRUE;
}

bool_t
NeuralNetOptimizer_getInterface(handle_t hOptimizer, OptimizerFunctionTable* pInterface) {
	NeuralNetOptimizer* This = (NeuralNetOptimizer*)hOptimizer;
	if (This == NULL) {
		return FALSE;
	}
	*pInterface = This->funcTable;
	return TRUE;
}
