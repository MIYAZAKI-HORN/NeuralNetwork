#include "STDTypeDefinition.h"
#include "SequentialNet.h" 
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDense.h"
#include "NeuralNetLayerConv2D.h"
#include "NeuralNetLayerMaxPooling2D.h"
#include "NeuralNetLayerGlobalAveragePooling2D.h"
#include "NeuralNetLayerDepthwiseConv2D.h"
#include "NeuralNetLayerPointwiseConv2D.h"
#include "NeuralNetLayerActivation.h"
#include "NeuralNetLayerSimpleRNN.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "NeuralNetLayerLayerNormalization.h"
#include "NeuralNetLayerPreDeconv2D.h"
#include "NeuralNetLayerResidualConnection.h"
#include "RandomValueGenerator.h"
#include "NeuralNetOptimizer.h"

//=====================================================================================
//  ファイルヘッダー
//=====================================================================================
typedef struct tagSequentialNetHeader {
	uint32_t	version;
	uint32_t	revision;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	numberOfLayers;
} SequentialNetHeader;


//=====================================================================================
// 各層のインターフェイスを取得する
//=====================================================================================
static
bool_t
NetLayer_getInterface(NeuralNetHeader* pNeuralNetHeader, LayerFuncTable* pInterface) {
	switch (pNeuralNetHeader->layerType) {
	case NET_LAYER_DENSE:
		NeuralNetLayerDense_getInterface(pInterface);
		break;
	case NET_LAYER_SIMPLE_RNN:
		NeuralNetLayerSimpleRNN_getInterface(pInterface);
		break;
	case NET_LAYER_CONV2D:
		NeuralNetLayerConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_DEPTHWISE_CONV2D:
		NeuralNetLayerDepthwiseConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_POINTWISE_CONV2D:
		NeuralNetLayerPointwiseConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_MAX_POOLING2D:
		NeuralNetLayerMaxPooling2D_getInterface(pInterface);
		break;
	case NET_LAYER_GLOBAL_AVERAGE_POOLING2D:
		NeuralNetLayerGlobalAveragePooling2D_getInterface(pInterface);
		break;
	case NET_LAYER_BATCH_NORMALIZATION:
		NeuralNetLayerBatchNormalization_getInterface(pInterface);
		break;
	case NET_LAYER_LAYER_NORMALIZATION:
		NeuralNetLayerLayerNormalization_getInterface(pInterface);
		break;
	case NET_LAYER_ACTIVATION:
		NeuralNetLayerActivation_getInterface(pInterface, (uint32_t*)pNeuralNetHeader);
		break;
	case NET_LAYER_PREDECONV2D:
		NeuralNetLayerPreDeconv2D_getInterface(pInterface);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_SENDER:
		NeuralNetLayerResidualConnectionSender_getInterface(pInterface);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_RECEIVER:
		NeuralNetLayerResidualConnectionReceiver_getInterface(pInterface);
		break;
	default:
		//エラー
		return FALSE;
		break;
	}
	return TRUE;
}

//=====================================================================================
//  シーケンシャルモデル構造体
//=====================================================================================
typedef struct tagSequentialNet {
	uint32_t*	pModelData;
	flt32_t*	ppDataBuffer[2];
	uint32_t*	pTemporaryBuffer;
	uint32_t	dataBufferSize;
	uint32_t	temporaryBufferSize;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	DataShape	inputShape;
	DataShape	outputShape;
	uint32_t	numberOfLayers;
	bool_t		fEnableLearning;
	bool_t		fSkipLastSoftmaxWhenBackpropagation;
	handle_t*	pLayerArray;
	handle_t*	pOptimizerArray;
	uint32_t	backPropagationBatchSize;
	uint32_t	backPropagationCounter;
	uint32_t	backPropagationEndLayerIndex;
	handle_t	hRandomValueGenerator;
} SequentialNet;

//=====================================================================================
//  計算ワークエリアサイズ関連情報の取得
//=====================================================================================
static 
bool_t
SequentialNet_getLayerWorkAreaInfo(
	uint32_t*				pModelData,
	uint32_t*				pLayerObjectSizeIn32BitWord,
	uint32_t*				pMaxDataSize, 
	uint32_t*				pMaxTempWorkAreaSizeIn32BitWord, 
	DataShape*				pFinalOutputShape,
	bool_t					fEnableLearning, 
	NeuralNetOptimizerType	optimizerType,
	uint32_t*				pOptimizerObjectSizeIn32BitWord,
	uint32_t				numberOfBackPropagationLayers) {
	uint32_t				i;
	bool_t					fLearnLayerParameters;			//in:back propagation flag
	uint32_t				layerObjectSizeIn32BitWord;		//out:layer object size
	uint32_t				numberOfLearningParameters;		//out:number of learning prameters for optimizer
	uint32_t				tempWorkAreaSizeIn32BitWord;	//out:temporary work area size for prediction and back propagation
	DataShape				inputShape;						//out:input data shape
	DataShape				outputShape;					//out:output data shape
	uint32_t				outputDim;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	uint32_t				totalOptimizerObjectSizeIn32BitWord;
	uint32_t				totalLayerObjectSizeIn32BitWord;
	SequentialNetHeader*	pHeader;
	NeuralNetHeader*		pNeuralNetHeader;
	DataShape				dataShape;
	uint32_t*				pLayerData;
	bool_t					fStatus;
	uint32_t				bpEndLayerIndex;
	LayerFuncTable			netLayerFuncTable;
	//----------------------------------------------------------------------------------
	//ヘッダーのセットおよびバージョンチェック
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	if (pHeader->version != MODEL_FILE_VERSION) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//パラメタチェック
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//層毎に必要なワークメモリサイズを加算する
	//----------------------------------------------------------------------------------
	dataShape.height	= pHeader->inHeight;
	dataShape.width		= pHeader->inWidth;
	dataShape.channel	= pHeader->inChannel;
	maxDataSize			= DataShape_getSize(&dataShape);
	maxTempWorkAreaSizeIn32BitWord	= 0;
	totalOptimizerObjectSizeIn32BitWord = 0;
	totalLayerObjectSizeIn32BitWord = 0;
	DataShape_initialize(&inputShape);
	DataShape_initialize(&outputShape);
	bpEndLayerIndex = pHeader->numberOfLayers - numberOfBackPropagationLayers;
	pLayerData = pModelData + size_in_type(sizeof(SequentialNetHeader), uint32_t);
	for( i=0; i< pHeader->numberOfLayers; i++ ) {
		pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
		//----------------------------------------------------------------------------------
		//層構築関連関数インターフェース取得
		//----------------------------------------------------------------------------------
		NetLayer_getInterface(pNeuralNetHeader,&netLayerFuncTable);
		//----------------------------------------------------------------------------------
		//back propagation用バッファーサイズ
		//----------------------------------------------------------------------------------
		if (i >= bpEndLayerIndex) {
			fLearnLayerParameters = fEnableLearning;
		}
		else {
			fLearnLayerParameters = FALSE;
		}
		//----------------------------------------------------------------------------------
		//層情報
		//----------------------------------------------------------------------------------
		tempWorkAreaSizeIn32BitWord		= 0;
		fStatus = netLayerFuncTable.pGetLayerInformation(
				pLayerData,						//in:image data
				fLearnLayerParameters,			//in:back propagation flag
				&layerObjectSizeIn32BitWord,	//out:layer object size
				&numberOfLearningParameters,	//out:number of learning prameters for optimizer
				&tempWorkAreaSizeIn32BitWord,	//out:temporary work area size for prediction and back propagation
				&inputShape,					//out:input data shape
				&outputShape					//out:output data shape
			);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//----------------------------------------------------------------------------------
		//層オブジェクトのサイズを積算
		//----------------------------------------------------------------------------------
		totalLayerObjectSizeIn32BitWord += layerObjectSizeIn32BitWord;
		//----------------------------------------------------------------------------------
		//optimizer
		//----------------------------------------------------------------------------------
		if (fLearnLayerParameters == TRUE && numberOfLearningParameters > 0) {
			OptimizerFunctionTable optimizerFunctionTable;
			fStatus = NeuralNetOptimizer_getInterfaceByType(optimizerType, &optimizerFunctionTable);
			if (fStatus == FALSE) {
				return FALSE;
			}
			totalOptimizerObjectSizeIn32BitWord += optimizerFunctionTable.pGetSizeIn32BitWord(numberOfLearningParameters);
		}
		//----------------------------------------------------------------------------------
		//最大入出力データサイズを収集
		//----------------------------------------------------------------------------------
		outputDim = DataShape_getSize(&outputShape);
		maxDataSize = larger_of(maxDataSize, outputDim);
		//----------------------------------------------------------------------------------
		//最大一次ワークエリアサイズを収集
		//----------------------------------------------------------------------------------
		maxTempWorkAreaSizeIn32BitWord = larger_of(maxTempWorkAreaSizeIn32BitWord,tempWorkAreaSizeIn32BitWord);
		//---------------------------------------------------------------------------------
		//次の層情報に移動
		//---------------------------------------------------------------------------------
		pLayerData += pNeuralNetHeader->sizeIn32BitWord;
	}
	//----------------------------------------------------------------------------------
	//最大データ入出力数と一次計算領域サイズ数および合計ワークエリアサイズ
	//----------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = totalLayerObjectSizeIn32BitWord;
	}
	if( pMaxDataSize != NULL ) {
		*pMaxDataSize = maxDataSize;
	}
	if( pMaxTempWorkAreaSizeIn32BitWord != NULL ) {
		*pMaxTempWorkAreaSizeIn32BitWord	= maxTempWorkAreaSizeIn32BitWord;
	}
	if (pOptimizerObjectSizeIn32BitWord != NULL) {
		*pOptimizerObjectSizeIn32BitWord = totalOptimizerObjectSizeIn32BitWord;
	}
	if (pFinalOutputShape != NULL) {
		*pFinalOutputShape = outputShape;
	}
	return TRUE;
}

//=====================================================================================
//  パラメタ微分値によりパラメタを更新する
//=====================================================================================
static
bool_t
SequentialNet_update(handle_t hModel) {
	uint32_t		i;
	SequentialNet*	This = (SequentialNet*)hModel;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//モデルハンドルをチェック
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	if (This->backPropagationBatchSize == 0) {
		return FALSE;
	}
	if (This->numberOfLayers == 0) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//パラメタ更新条件
	//----------------------------------------------------------------------------------
	if (This->backPropagationCounter < This->backPropagationBatchSize) {
		return TRUE;
	}
	//---------------------------------------------------------------------------------
	//各層のパラメタを更新する
	//---------------------------------------------------------------------------------
	for (i = This->backPropagationEndLayerIndex; i < This->numberOfLayers; i++) {
		//---------------------------------------------------------------------------------
		//
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_update(This->pLayerArray[i]);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//逆伝搬カウンタを初期化
	//---------------------------------------------------------------------------------
	This->backPropagationCounter = 0;
	return TRUE;
}

//=====================================================================================
//  バージョン情報取得
//=====================================================================================
#define NEURAL_NETWORK_MODEL_VERSION_MAJOR		(1)
#define NEURAL_NETWORK_MODEL_VERSION_MINOR		(0)
#define NEURAL_NETWORK_MODEL_VERSION_REVISION	(0)

void
SequentialNet_getVersion(uint16_t* pMajorVersion, uint16_t* pMinorVersion, uint16_t* pRevision) {
	if (pMajorVersion != NULL) {
		*pMajorVersion = NEURAL_NETWORK_MODEL_VERSION_MAJOR;
	}
	if (pMinorVersion != NULL) {
		*pMinorVersion = NEURAL_NETWORK_MODEL_VERSION_MINOR;
	}
	if (pRevision != NULL) {
		*pRevision = NEURAL_NETWORK_MODEL_VERSION_REVISION;
	}
}

//=====================================================================================
//  必要ワークエリアサイズ取得
//=====================================================================================
uint32_t	
SequentialNet_getSizeIn32BitWord(uint32_t* pModelData, bool_t fEnableLearning,NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers) {
	bool_t					fStatus;
	uint32_t				sizeOfEngineIn32BitWord = 0;
	SequentialNetHeader*	pHeader;
	uint32_t				layerObjectSizeIn32BitWord;
	uint32_t				optimizerObjectSizeIn32BitWord;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	DataShape				outputShape;
	//----------------------------------------------------------------------------------
	//モデルデータチェック
	//----------------------------------------------------------------------------------
	if (pModelData == NULL) {
		return 0;
	}
	//----------------------------------------------------------------------------------
	//ヘッダーのセット
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	//----------------------------------------------------------------------------------
	//逆伝搬層数が総層数を超える場合は、層総数に置き換えする
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//逆伝搬層数に0を設定した場合は総層数とする
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers == 0) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//作業領域サイズを取得
	//----------------------------------------------------------------------------------
	fStatus = SequentialNet_getLayerWorkAreaInfo(pModelData,&layerObjectSizeIn32BitWord ,&maxDataSize,&maxTempWorkAreaSizeIn32BitWord,&outputShape,fEnableLearning, optimizer,&optimizerObjectSizeIn32BitWord,numberOfBackPropagationLayers);
	if (fStatus == FALSE) {
		return 0;
	}
	//----------------------------------------------------------------------------------
	//モデルサイズ
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord = size_in_type(sizeof(SequentialNet), uint32_t);
	//----------------------------------------------------------------------------------
	//pLayerArray & pOptimizerArray
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += size_in_type(pHeader->numberOfLayers * sizeof(handle_t)*2, uint32_t);
	//----------------------------------------------------------------------------------
	//層オブジェクトサイズを加算  
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += layerObjectSizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//オプティマイザーサイズを加算する
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += optimizerObjectSizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//入出力ダブルバッファと一時ワークエリアサイズ
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += 2 * size_in_type(maxDataSize * sizeof(flt32_t), uint32_t);
	sizeOfEngineIn32BitWord += size_in_type(maxTempWorkAreaSizeIn32BitWord * sizeof(flt32_t), uint32_t);
	//----------------------------------------------------------------------------------
	//乱数発生器オブジェクトサイズを加算する
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += RandomValueGenerator_getSizeIn32BitWord();
	return sizeOfEngineIn32BitWord;
}

//=====================================================================================
//  モデル構築
//=====================================================================================
handle_t
SequentialNet_construct(uint32_t* pModelData,bool_t fEnableLearning,uint32_t batchSize, NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers,uint32_t* pWorkArea,uint32_t sizeOfWorkAreaIn32BitWord) {
	SequentialNet*			This;
	uint32_t				i;
	bool_t					fStatus;
	uint32_t*				pWorkAreaHead; 
	uint32_t				requiredSizeOfWorkAreaIn32BitWord;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	uint32_t				sizeIn32BitWord;
	SequentialNetHeader*	pHeader;
	NeuralNetHeader*		pNeuralNetHeader;
	uint32_t*				pLayerData;
	bool_t					fLearnLayerParameters;
	uint32_t				layerObjectSizeIn32BitWord;
	uint32_t				numberOfLearningParameters;
	uint32_t				tempWorkAreaSizeIn32BitWord;
	DataShape				inputShape;
	DataShape				outputShape;
	handle_t				hOptimizer;
	handle_t				hLayer;
	uint32_t				bpEndLayerIndex;
	LayerFuncTable			netLayerFuncTable;
	//----------------------------------------------------------------------------------
	//モデルデータチェック
	//----------------------------------------------------------------------------------
	if( pModelData == NULL ) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//作業領域チェック
	//----------------------------------------------------------------------------------
	if( pWorkArea == NULL ) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//パラメタチェック
	//----------------------------------------------------------------------------------
	if (fEnableLearning == TRUE && batchSize == 0) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//ヘッダーのセット
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	//----------------------------------------------------------------------------------
	//逆伝搬層数が総層数を超える場合は層総数に置き換え
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//逆伝搬層数に0を設定すると総層数
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers == 0) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//ワークエリアセット
	//----------------------------------------------------------------------------------
	pWorkAreaHead = pWorkArea;
	//----------------------------------------------------------------------------------
	//作業領域サイズをチェック
	//----------------------------------------------------------------------------------
	requiredSizeOfWorkAreaIn32BitWord = SequentialNet_getSizeIn32BitWord(pModelData,fEnableLearning,optimizer,numberOfBackPropagationLayers);
	if(requiredSizeOfWorkAreaIn32BitWord == 0 ) {
		return NULL;
	}
	if( sizeOfWorkAreaIn32BitWord < requiredSizeOfWorkAreaIn32BitWord ) { 
		return NULL;
	}
	//-----------------------------------------------------------------------------------
	//シーケンシャルモデルオブジェクト
	//-----------------------------------------------------------------------------------
	This = (SequentialNet*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(sizeof(SequentialNet), uint32_t);
	This->pLayerArray = (handle_t*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(pHeader->numberOfLayers * sizeof(handle_t), uint32_t);
	This->pOptimizerArray = (handle_t*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(pHeader->numberOfLayers * sizeof(handle_t), uint32_t);
	//----------------------------------------------------------------------------------
	//乱数発生器
	//----------------------------------------------------------------------------------
	sizeIn32BitWord = RandomValueGenerator_getSizeIn32BitWord();
	This->hRandomValueGenerator = RandomValueGenerator_construct(0, pWorkAreaHead, sizeIn32BitWord);
	pWorkAreaHead += sizeIn32BitWord;
	//-----------------------------------------------------------------------------------
	//データおよびパラメタ確保
	//-----------------------------------------------------------------------------------
	This->pModelData							= pModelData;
	This->numberOfLayers						= pHeader->numberOfLayers;
	This->fEnableLearning						= fEnableLearning;
	This->fSkipLastSoftmaxWhenBackpropagation	= TRUE;
	This->backPropagationBatchSize				= batchSize;
	This->backPropagationCounter				= 0;
	This->backPropagationEndLayerIndex			= pHeader->numberOfLayers - numberOfBackPropagationLayers;
	//-----------------------------------------------------------------------------------
	//ワークエリア情報の取得
	//-----------------------------------------------------------------------------------
	fStatus = SequentialNet_getLayerWorkAreaInfo(
		pModelData,
		NULL,
		&maxDataSize, 
		&maxTempWorkAreaSizeIn32BitWord, 
		&outputShape,
		fEnableLearning, 
		optimizer,
		NULL, 
		numberOfBackPropagationLayers);
	if (fStatus == FALSE) {
		return NULL;
	}
	//-----------------------------------------------------------------------------------
	//バッファサイズ
	//-----------------------------------------------------------------------------------
	This->dataBufferSize = maxDataSize;	// double bufferのためこの2倍のサイズが確保される
	This->temporaryBufferSize = maxTempWorkAreaSizeIn32BitWord;
	//-----------------------------------------------------------------------------------
	//入出力形状のセット
	//-----------------------------------------------------------------------------------
	DataShape_construct(&This->inputShape,pHeader->inHeight,pHeader->inWidth,pHeader->inChannel);
	This->outputShape = outputShape;
	//-----------------------------------------------------------------------------------
	//入出力バッファ領域をセット
	//-----------------------------------------------------------------------------------
	sizeIn32BitWord		= size_in_type(sizeof(flt32_t) * maxDataSize, uint32_t);
	for(i=0;i<2;i++) {
		This->ppDataBuffer[i]	= (flt32_t*)pWorkAreaHead;
		pWorkAreaHead			+= sizeIn32BitWord;
	}
	//-----------------------------------------------------------------------------------
	//層内部計算領域をセット
	//-----------------------------------------------------------------------------------
	sizeIn32BitWord		= size_in_type(sizeof(flt32_t) * maxTempWorkAreaSizeIn32BitWord, uint32_t);
	This->pTemporaryBuffer		= pWorkAreaHead;
	pWorkAreaHead				+= sizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//層構築
	//----------------------------------------------------------------------------------
	maxDataSize = pHeader->inHeight * pHeader->inWidth * pHeader->inChannel;
	maxTempWorkAreaSizeIn32BitWord = 0;
	DataShape_initialize(&inputShape);
	DataShape_initialize(&outputShape);
	bpEndLayerIndex = pHeader->numberOfLayers - numberOfBackPropagationLayers;
	pLayerData = pModelData + size_in_type(sizeof(SequentialNetHeader), uint32_t);
	for (i = 0; i < pHeader->numberOfLayers; i++) {
		pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
		//----------------------------------------------------------------------------------
		//層インターフェース取得
		//----------------------------------------------------------------------------------
		NetLayer_getInterface(pNeuralNetHeader, &netLayerFuncTable);
		//----------------------------------------------------------------------------------
		//back propagation用バッファーサイズ
		//----------------------------------------------------------------------------------
		if (i >= bpEndLayerIndex) {
			fLearnLayerParameters = fEnableLearning;
		}
		else {
			fLearnLayerParameters = FALSE;
		}
		//----------------------------------------------------------------------------------
		//層情報
		//----------------------------------------------------------------------------------
		tempWorkAreaSizeIn32BitWord = 0;
		fStatus = netLayerFuncTable.pGetLayerInformation(
			pLayerData,						// in:image data
			fLearnLayerParameters,			// in:back propagation flag
			&layerObjectSizeIn32BitWord,	// out:layer object size
			&numberOfLearningParameters,	// out:number of learning prameters for optimizer
			&tempWorkAreaSizeIn32BitWord,	// out:temporary work area size for prediction and back propagation
			&inputShape,					// out:input data shape
			&outputShape					// out:output data shape
		);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//----------------------------------------------------------------------------------
		//optimizer作成
		//----------------------------------------------------------------------------------
		if (fLearnLayerParameters == TRUE && numberOfLearningParameters > 0) {
			OptimizerFunctionTable optimizerFunctionTable;
			NeuralNetOptimizer_getInterfaceByType(optimizer, &optimizerFunctionTable);
			sizeIn32BitWord = optimizerFunctionTable.pGetSizeIn32BitWord(numberOfLearningParameters);
			hOptimizer = optimizerFunctionTable.pConstruct(numberOfLearningParameters, batchSize, pWorkAreaHead, sizeIn32BitWord);
			pWorkAreaHead += sizeIn32BitWord;
		}
		else {
			hOptimizer = NULL;
		}
		//----------------------------------------------------------------------------------
		//層作成
		//----------------------------------------------------------------------------------
		hLayer = netLayerFuncTable.pConstruct(pLayerData, pWorkAreaHead, layerObjectSizeIn32BitWord, fLearnLayerParameters, hOptimizer);
		if (hLayer == NULL) {
			return FALSE;
		}
		pWorkAreaHead += layerObjectSizeIn32BitWord;
		This->pLayerArray[i] = hLayer;
		This->pOptimizerArray[i] = hOptimizer;
		//---------------------------------------------------------------------------------
		//次の層情報に移動
		//---------------------------------------------------------------------------------
		pLayerData += pNeuralNetHeader->sizeIn32BitWord;
	}
	//---------------------------------------------------------------------------------
	//入出力バッファーポインタ
	//---------------------------------------------------------------------------------
	This->pInputBuffer	= NULL;
	This->pOutputBuffer	= NULL;
	//---------------------------------------------------------------------------------
	//チェック
	//---------------------------------------------------------------------------------
	if( (uint32_t)(pWorkAreaHead - pWorkArea) > sizeOfWorkAreaIn32BitWord ) {
		return NULL;
	}
	return This;
}

//=====================================================================================
//  予測実行
//=====================================================================================
bool_t
SequentialNet_predict(handle_t hModel,flt32_t* pInputData,uint32_t inputDataArraySize) {
	uint32_t				i;
	SequentialNet*			This = (SequentialNet*)hModel;
	SequentialNetHeader*	pHeader;
	uint32_t				inputDim;
	uint32_t				outputDim;
	PropagationInfo			propagationInfo;
	flt32_t*				pLayerParam;
	bool_t					fStatus;
	uint32_t				layerOrder;
	//---------------------------------------------------------------------------------
	//モデルハンドルをチェック
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//入出力データ次元
	//---------------------------------------------------------------------------------
	inputDim	= DataShape_getSize(&This->inputShape);
	outputDim	= DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//パラメータチェック
	//---------------------------------------------------------------------------------
	if (pInputData == NULL) {
		return FALSE;
	}
	if (inputDim != inputDataArraySize) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//ヘッダ
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)This->pModelData;
	//---------------------------------------------------------------------------------
	//入出力バッファーポインタのセット
	//---------------------------------------------------------------------------------
	This->pInputBuffer	= This->ppDataBuffer[0];
	This->pOutputBuffer	= This->ppDataBuffer[1];
	//---------------------------------------------------------------------------------
	//入出力バッファの初期化
	//---------------------------------------------------------------------------------
	for(i=0;i<inputDim;i++) {
		This->pInputBuffer[i] = pInputData[i];
	}
	for(i=0;i<outputDim;i++) {
		This->pOutputBuffer[i] = 0.0f;
	}
	//----------------------------------------------------------------------------------
	//入力データ形状
	//----------------------------------------------------------------------------------
	propagationInfo.dataShape.height = pHeader->inHeight;
	propagationInfo.dataShape.width = pHeader->inWidth;
	propagationInfo.dataShape.channel = pHeader->inChannel;
	propagationInfo.layerOrder = 0;
	propagationInfo.inputBufferSize = This->dataBufferSize;
	propagationInfo.outputBufferSize = This->dataBufferSize;
	propagationInfo.temporaryBufferSize = This->temporaryBufferSize;
	propagationInfo.pInputBuffer = This->pInputBuffer;
	propagationInfo.pOutputBuffer = This->pOutputBuffer;
	propagationInfo.pTemporaryBuffer = This->pTemporaryBuffer;
	//---------------------------------------------------------------------------------
	// 各層をシーケンシャルに計算
	//---------------------------------------------------------------------------------
	for( i=0; i<This->numberOfLayers; i++ ) {
		//---------------------------------------------------------------------------------
		//バッファ設定
		//---------------------------------------------------------------------------------
		propagationInfo.pInputBuffer = This->pInputBuffer;
		propagationInfo.pOutputBuffer = This->pOutputBuffer;
		//---------------------------------------------------------------------------------
		//順伝搬
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_forward(This->pLayerArray[i], &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//層計算順序取得:シーケンシャルなので順番通りであることを確認
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_getOrder(This->pLayerArray[i],&layerOrder);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//入出力バッファの入れ替え
		//---------------------------------------------------------------------------------
		pLayerParam = This->pInputBuffer;
		This->pInputBuffer	= This->pOutputBuffer;
		This->pOutputBuffer	= pLayerParam;
	}
	//---------------------------------------------------------------------------------
	//出力バッファー設定
	//---------------------------------------------------------------------------------
	This->pOutputBuffer	= This->pInputBuffer;
	return TRUE;
}

//=====================================================================================
//  個別出力結果取得
//=====================================================================================
bool_t
SequentialNet_getPrediction(handle_t hModel,uint32_t stateIndex, flt32_t* pValue){
	SequentialNet*	This = (SequentialNet*)hModel;
	uint32_t		outputDim;
	//---------------------------------------------------------------------------------
	//モデルハンドルをチェック
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//出力データ次元
	//---------------------------------------------------------------------------------
	outputDim = DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//状態インデックスをチェック
	//---------------------------------------------------------------------------------
	if( stateIndex >= outputDim ) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//出力バッファーおよび出力パラメタをチェック
	//---------------------------------------------------------------------------------
	if( This->pOutputBuffer == NULL ) {
		return FALSE;
	}
	if (pValue == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//出力値（予測値）を設定
	//---------------------------------------------------------------------------------
	*pValue = This->pOutputBuffer[stateIndex];
	return TRUE;
}

//=====================================================================================
// 入力形状取得
//=====================================================================================
bool_t
SequentialNet_getInputShape(handle_t hModel, uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel) {
	SequentialNet*	This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//形状セット
	//---------------------------------------------------------------------------------
	if (pHeight != NULL) {
		*pHeight = This->inputShape.height;
	}
	if (pWidth != NULL) {
		*pWidth = This->inputShape.width;
	}
	if (pChannel != NULL) {
		*pChannel = This->inputShape.channel;
	}
	return TRUE;
}

//=====================================================================================
//出力次数取得
//=====================================================================================
bool_t
SequentialNet_getOutputShape(handle_t hModel,uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//形状セット
	//---------------------------------------------------------------------------------
	if (pHeight != NULL) {
		*pHeight = This->outputShape.height;
	}
	if (pWidth != NULL) {
		*pWidth = This->outputShape.width;
	}
	if (pChannel != NULL) {
		*pChannel = This->outputShape.channel;
	}
	return TRUE;
}

//=====================================================================================
//  層の数を取得
//=====================================================================================
bool_t
SequentialNet_getNumberOfLayers(handle_t hModel, uint32_t* pNumOfLayers) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	if (pNumOfLayers != NULL) {
		*pNumOfLayers = This->numberOfLayers;
	}
	return TRUE;
}

//=====================================================================================
//  層のタイプを取得
//=====================================================================================
bool_t		
SequentialNet_getLayerType(handle_t hModel,uint32_t layerIndex, NetLayerType* pNetLayerType) {
	SequentialNet*		This = (SequentialNet*)hModel;
	NeuralNetHeader*	pNeuralNetHeader = NULL;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (layerIndex >= This->numberOfLayers) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層際プを取得
	//---------------------------------------------------------------------------------
	return NeuralNetLayer_getType(This->pLayerArray[layerIndex], pNetLayerType);
}

//=====================================================================================
//  層ハンドルを取得
//=====================================================================================
bool_t
SequentialNet_getLayerHandle(handle_t hModel, uint32_t layerIndex, handle_t* phLayer) {
	SequentialNet*		This = (SequentialNet*)hModel;
	NeuralNetHeader*	pNeuralNetHeader = NULL;
	//---------------------------------------------------------------------------------
	//エラーハンドリング
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (layerIndex >= This->numberOfLayers) {
		return FALSE;
	}
	if (phLayer == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層ハンドルをセット
	//---------------------------------------------------------------------------------
	*phLayer = This->pLayerArray[layerIndex];
	return TRUE;
}

//=====================================================================================
//  モデルパラメタ初期化
//=====================================================================================
bool_t
SequentialNet_initializeParameter(handle_t hModel) {
	SequentialNet*	This = (SequentialNet*)hModel;
	uint32_t		i;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//モデルハンドルをチェック
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ初期化
	//---------------------------------------------------------------------------------
	for (i = 0; i < This->numberOfLayers; i++) {
		//---------------------------------------------------------------------------------
		//
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_initializeParameters(This->pLayerArray[i], This->hRandomValueGenerator);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//逆伝搬カウンタを初期化
	//---------------------------------------------------------------------------------
	This->backPropagationCounter = 0;
	return TRUE;
}

//=====================================================================================
//  オプティマイザーを取得する
//=====================================================================================
handle_t
SequentialNet_getOptimizer(handle_t hModel, uint32_t layerIndex) {
	SequentialNet* This = (SequentialNet*)hModel;
	NeuralNetHeader* pNeuralNetHeader = NULL;
	if (This == NULL) {
		return NULL;
	}
	if (layerIndex >= This->numberOfLayers) {
		return NULL;
	}
	return This->pOptimizerArray[layerIndex];
}

//=====================================================================================
//  誤差逆伝搬時に最終層のsoftmaxをスキップするか否かの設定を行う（デフォルトはスキップ）
//	分類問題のロス計算を、（softmax＋クロスエントロピー）とした場合、（正解値-予測値）と簡単になるため、
//	最終層のsoftmaxはスキップすると与えるロス計算が簡単になる。
//=====================================================================================
bool_t
SequentialNet_skipLastSoftmaxWhenBackpropagation(handle_t hModel, bool_t fSkip) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	This->fSkipLastSoftmaxWhenBackpropagation = fSkip;
	return TRUE;
}

//=====================================================================================
//  学習（誤差逆伝搬）
//=====================================================================================
bool_t
SequentialNet_fit(handle_t hModel, flt32_t* pLoss, uint32_t arraySize) {
	uint32_t				i,j;
	SequentialNet*			This = (SequentialNet*)hModel;
	uint32_t				inputDim;
	uint32_t				outputDim;
	PropagationInfo			propagationInfo;
	flt32_t*				pNextLayerOutput;
	handle_t				hLastLayer;
	NetLayerType			layerType;
	NeuralNetActivationType activationType;
	int32_t					targetLayerOrder;
	bool_t					fStatus;
	//---------------------------------------------------------------------------------
	//モデルハンドルをチェック
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習モードで無ければエラー
	//---------------------------------------------------------------------------------
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//入出力データ次元
	//---------------------------------------------------------------------------------
	inputDim	= DataShape_getSize(&This->inputShape);
	outputDim	= DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//パラメータチェック
	//---------------------------------------------------------------------------------
	if (pLoss == NULL) {
		return FALSE;
	}
	if (outputDim != arraySize) {
		return FALSE;
	}
	if (This->numberOfLayers == 0) {
		return FALSE;
	}
	if (This->backPropagationBatchSize == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//バッチサイズを超えている場合は無視
	//---------------------------------------------------------------------------------
	if (This->backPropagationCounter > This->backPropagationBatchSize) {
		return TRUE;
	}
	//----------------------------------------------------------------------------------
	//パラメタを更新したのちに誤差逆伝搬を実施した回数
	//----------------------------------------------------------------------------------
	This->backPropagationCounter++;
	//---------------------------------------------------------------------------------
	//入出力バッファーポインタのセット
	//---------------------------------------------------------------------------------
	This->pInputBuffer = This->ppDataBuffer[0];
	This->pOutputBuffer = This->ppDataBuffer[1];
	//---------------------------------------------------------------------------------
	//入出力バッファの初期化
	//---------------------------------------------------------------------------------
	for (i = 0; i < inputDim; i++) {
		This->pInputBuffer[i] = 0.0f;
	}
	for (i = 0; i < outputDim; i++) {
		This->pOutputBuffer[i] = pLoss[i];
	}
	//----------------------------------------------------------------------------------
	//入力データ形状
	//----------------------------------------------------------------------------------
	propagationInfo.dataShape = This->outputShape;
	propagationInfo.layerOrder = This->numberOfLayers;
	propagationInfo.inputBufferSize = This->dataBufferSize;
	propagationInfo.outputBufferSize = This->dataBufferSize;
	propagationInfo.temporaryBufferSize = This->temporaryBufferSize;
	propagationInfo.pInputBuffer = This->pInputBuffer;
	propagationInfo.pOutputBuffer = This->pOutputBuffer;
	propagationInfo.pTemporaryBuffer = This->pTemporaryBuffer;
	//---------------------------------------------------------------------------------
	//逆伝搬の開始層を選択
	//layerOrder : 1　～　This->numberOfLayers
	//最終層がactivation層で、softmaxの場合は、t-yがこの層を込みとしたロスとして与えられることを前提として、スキップする（簡単のため）
	//---------------------------------------------------------------------------------
	if (This->fSkipLastSoftmaxWhenBackpropagation == TRUE) {
		hLastLayer = This->pLayerArray[This->numberOfLayers - 1];	//最終層
		NeuralNetLayer_getType(hLastLayer, &layerType);	//層のタイプを取得
		switch (layerType) {
		case NET_LAYER_ACTIVATION:
			//activation層の場合
			fStatus = NeuralNetLayerActivation_getType(hLastLayer, &activationType);	//activationのタイプを取得
			if (activationType == NEURAL_NET_ACTIVATION_SOFTMAX) {
				//softmaxの場合は、スキップする
				//形状は変わらない
				if (This->numberOfLayers == 1) {
					return FALSE;
				}
				targetLayerOrder = This->numberOfLayers - 1;	// skip softmax
			}
			else {
				//それ以外の活性化関数は逆伝搬の対象となる
				targetLayerOrder = This->numberOfLayers;
			}
			break;
		default:
			//activation層以外の層
			//最終層から逆伝搬
			targetLayerOrder = This->numberOfLayers;
			break;
		}
	}
	else {
		//最終層から逆伝搬
		targetLayerOrder = This->numberOfLayers;
	}
	//---------------------------------------------------------------------------------
	//逆伝搬
	//---------------------------------------------------------------------------------
#if 1
	do {
		uint32_t layerIndex = targetLayerOrder - 1;
		//---------------------------------------------------------------------------------
		//バッファ設定
		//---------------------------------------------------------------------------------
		propagationInfo.pInputBuffer = This->pInputBuffer;
		propagationInfo.pOutputBuffer = This->pOutputBuffer;
		//---------------------------------------------------------------------------------
		//各層の逆伝搬
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_backward(This->pLayerArray[layerIndex], &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//入出力バッファの入れ替え
		//---------------------------------------------------------------------------------
		pNextLayerOutput = This->pInputBuffer;
		This->pInputBuffer = This->pOutputBuffer;
		This->pOutputBuffer = pNextLayerOutput;
	} while (--targetLayerOrder > 0); 
#else
	do {
		for (j = 0; j < This->numberOfLayers; j++) {
			uint32_t layerOrder;
			uint32_t layerIndex;
			//---------------------------------------------------------------------------------
			//層計算順序取得:シーケンシャルなので順番通り
			//layerOrder : 1　～　This->numberOfLayers
			//layerIndex : 0　～　This->numberOfLayers-1
			//---------------------------------------------------------------------------------
			fStatus = NeuralNetLayer_getOrder(This->pLayerArray[j], &layerOrder);
			if (fStatus == FALSE) {
				return FALSE;
			}
			if (layerOrder != targetLayerOrder) {
				continue;
			}
			layerIndex = targetLayerOrder - 1;
			//---------------------------------------------------------------------------------
			//バッファ設定
			//---------------------------------------------------------------------------------
			propagationInfo.pInputBuffer = This->pInputBuffer;
			propagationInfo.pOutputBuffer = This->pOutputBuffer;
			//---------------------------------------------------------------------------------
			//各層の逆伝搬
			//---------------------------------------------------------------------------------
			fStatus = NeuralNetLayer_backward(This->pLayerArray[layerIndex], &propagationInfo);
			if (fStatus == FALSE) {
				return FALSE;
			}
			//---------------------------------------------------------------------------------
			//入出力バッファの入れ替え
			//---------------------------------------------------------------------------------
			pNextLayerOutput = This->pInputBuffer;
			This->pInputBuffer = This->pOutputBuffer;
			This->pOutputBuffer = pNextLayerOutput;
		}
	} while (--targetLayerOrder > 0);
#endif
	//---------------------------------------------------------------------------------
	//誤差逆伝播回数がバッチサイズに達した場合にネットワークパラメタを更新する
	//---------------------------------------------------------------------------------
	SequentialNet_update(hModel);
	return TRUE;
}

//=====================================================================================
//  モデルヘッダ作成
//=====================================================================================
bool_t
SequentialNet_createHeader(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
	uint32_t	numberOfLayers,
	uint32_t*	pSizeOfHeaderIn32BitWord) {
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	SequentialNetHeader* pSequentialNetHeader;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(SequentialNetHeader), uint32_t);
	sizeLayer = sizeHeader;
	if (pSizeOfHeaderIn32BitWord != NULL) {
		*pSizeOfHeaderIn32BitWord = sizeLayer;
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
		// header
		pSequentialNetHeader = (SequentialNetHeader*)pLayer;
		pSequentialNetHeader->version			= MODEL_FILE_VERSION;
		pSequentialNetHeader->revision			= 0;
		pSequentialNetHeader->inHeight		= inHeight;
		pSequentialNetHeader->inWidth		= inWidth;
		pSequentialNetHeader->inChannel		= inChannel;
		pSequentialNetHeader->numberOfLayers	= numberOfLayers;
		pLayer += sizeHeader;
	}
	return TRUE;
}

//=====================================================================================
//  Dense層作成
//=====================================================================================
bool_t
SequentialNet_appendDense(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel,uint32_t unit,uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerDense_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, unit, pSizeOfLayerIn32BitWord);
}
	
//=====================================================================================
//  SimpleRNN層作成
//=====================================================================================
bool_t
SequentialNet_appendSimpleRNN(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t unit, NeuralNetActivationType activation, bool_t returnSequence, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerSimpleRNN_constructLayerData(pBuffer,sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel,unit, activation,returnSequence,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  Conv2D層作成
//=====================================================================================
bool_t
SequentialNet_appendConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel,
	uint32_t nFilter, uint32_t	kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t	strideWidth, bool_t fPadding,
	uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerConv2D_constructLayerData(pBuffer,sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel,nFilter,kernelHeight,kernelWidth,strideHeight,strideWidth, fPadding,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  DepthwiseConv2D層作成
//=====================================================================================
bool_t
SequentialNet_appendDepthwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t nFilter, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t strideWidth, bool_t fPadding, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerDepthwiseConv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, nFilter, kernelHeight, kernelWidth, strideHeight, strideWidth, fPadding, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  PointwiseConv2D層作成
//=====================================================================================
bool_t
SequentialNet_appendPointwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t pw_nFilter, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerPointwiseConv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pw_nFilter, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  MaxPooling2D層作成
//=====================================================================================
bool_t
SequentialNet_appendMaxPooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t poolinghHeight, uint32_t poolingWidth, uint32_t strideHeight, uint32_t strideWidth, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerMaxPooling2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, poolinghHeight, poolingWidth, strideHeight, strideWidth, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  GlobalAveragePooling2D層作成
//=====================================================================================
bool_t
SequentialNet_appendGlobalAveragePooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerGlobalAveragePooling2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  BatchNormalization層作成
//=====================================================================================
bool_t
SequentialNet_appendBatchNormalization(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerBatchNormalization_constructLayerData(pBuffer,sizeOfBufferIn32BitWord,pInputHeight,pInputWidth,pInputChannel,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  LayerNormalization層作成
//=====================================================================================
bool_t
SequentialNet_appendLayerNormalization(uint32_t * pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t * pInputHeight, uint32_t * pInputWidth, uint32_t * pInputChannel, uint32_t * pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerLayerNormalization_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  Activation層作成
//=====================================================================================
bool_t
SequentialNet_appendActivation(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, NeuralNetActivationType activation, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, activation, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  PreDeconv2D層作成
//=====================================================================================
bool_t
SequentialNet_appendPreDeconv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t strideHeight, uint32_t strideWidth, uint32_t outHeight, uint32_t outWidth, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerPreDeconv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, strideHeight, strideWidth, outHeight, outWidth, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  NeuralNetLayerResidualConnectionSender層作成
//=====================================================================================
bool_t
SequentialNet_appendResidualConnectionSender(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerResidualConnectionSender_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  NeuralNetLayerResidualConnectionReceiver層作成
//=====================================================================================
bool_t
SequentialNet_appendResidualConnectionReceiver(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerResidualConnectionReceiver_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

