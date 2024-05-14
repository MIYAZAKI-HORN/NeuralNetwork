#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerResidualConnection.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON				(0.001f)
#define DEFAULT_MOMENTUM	(0.99f)

//=====================================================================================
//  ResidualConnectionSender層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagResidualConnectionSenderNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} ResidualConnectionSenderNeuralNetHeader;

//=====================================================================================
//  ResidualConnectionSender層構造体
//=====================================================================================
typedef struct tagResidualConnectionSenderNeuralNetLayer {
	NeuralNetLayer						super;			//base layer class
	flt32_t*							pReceiverLoss;	//データバッファ
	handle_t							hReceiver;		//受信先層ハンドル
	ResidualConnectionSenderInterface	funcTable;		//sender funcTable
} ResidualConnectionSenderNeuralNetLayer;

//=====================================================================================
//  ResidualConnectionReceiver層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagResidualConnectionReceiverNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} ResidualConnectionReceiverNeuralNetHeader;

//=====================================================================================
//  ResidualConnectionReceiver層構造体
//=====================================================================================
typedef struct tagResidualConnectionReceiverNeuralNetLayer {
	NeuralNetLayer						super;		//base layer class
	flt32_t*							pX;			//データバッファ
	handle_t							hSender;	//送信先層ハンドル
	ResidualConnectionReceiverInterface	funcTable;	//receiver funcTable
} ResidualConnectionReceiverNeuralNetLayer;

//=====================================================================================
//=====================================================================================
// 
//  ResidualConnectionSender層
// 
//=====================================================================================
//=====================================================================================

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
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
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	flt32_t*	pLoss;
	flt32_t*	pInputData;
	flt32_t*	pOutputData;
	uint32_t	size;
	PropagationInfo	 propagationInfo;
	ResidualConnectionReceiverInterface	receiverInterface;
	bool_t	fStatus;
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//入力次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//逆伝搬用:初期化しておく
		//-------------------------------------------------------
		pLoss	= pResidualConnectionSenderLayer->pReceiverLoss;
		size = inputDataDim;
		while (size--) {
			*pLoss++ = 0.0f;
		}
	}
	//---------------------------------------------------------------------------------
	//受信層にデータを送信
	//---------------------------------------------------------------------------------
	if (pResidualConnectionSenderLayer->hReceiver != NULL) {
		propagationInfo = *pPropagationInfo;
		propagationInfo.outputBufferSize	= 0;	//出力バッファ配列サイズ
		propagationInfo.pOutputBuffer		= NULL;	//出力バッファ]
		fStatus = NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(pResidualConnectionSenderLayer->hReceiver,&receiverInterface);
		if (fStatus == FALSE) {
			return FALSE;
		}
		fStatus = receiverInterface.pSetForwardData(pResidualConnectionSenderLayer->hReceiver, hLayer, &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//出力バッファにデータをコピー
	//---------------------------------------------------------------------------------
	pInputData	= pPropagationInfo->pInputBuffer;
	pOutputData = pPropagationInfo->pOutputBuffer;
	size	= inputDataDim;
	while (size--) {
		*pOutputData++ = *pInputData++;
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
NeuralNetLayerResidualConnectionSender_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	flt32_t*	pInputData;
	flt32_t*	pLoss;
	flt32_t*	pReceiverLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	if (pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//誤差伝搬先バッファー
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInputData = pPropagationInfo->pInputBuffer;
	pLoss = pPropagationInfo->pOutputBuffer;
	pReceiverLoss = pResidualConnectionSenderLayer->pReceiverLoss;
	while (size--) {
		*pInputData++ = (*pLoss++) + (*pReceiverLoss++);
	}
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
NeuralNetLayerResidualConnectionSender_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  更新パラメタ設定
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_setBackwardData(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	flt32_t*	pLoss;
	flt32_t*	pReceiverLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	//---------------------------------------------------------------------------------
	//学習対象でない場合
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE ) {
		return TRUE;
	}
	//---------------------------------------------------------------------------------
	//バッファチェック
	//---------------------------------------------------------------------------------
	if( pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Lossをバッファに格納する
	//---------------------------------------------------------------------------------
	size = inputDataDim;
	pLoss = pPropagationInfo->pOutputBuffer;
	pReceiverLoss = pResidualConnectionSenderLayer->pReceiverLoss;
	while (size--) {
		*pReceiverLoss++ = *pLoss++;
	}
	return TRUE;
}

//=====================================================================================
//  更新パラメタ設定
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_setReceiver(handle_t hLayer, handle_t hReceiver)
{
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NetLayerType	senderLayerType;
	NetLayerType	receiverLayerType;
	DataShape		senderInputShape;
	DataShape		receiverInputShape;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//層タイプ取得
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getType(hLayer, &senderLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus	= NeuralNetLayer_getType(hReceiver, &receiverLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (senderLayerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		return FALSE;
	}
	if (receiverLayerType != NET_LAYER_RESIDUAL_CONNECTION_RECEIVER) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層入力形状取得
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getShape(hLayer, &senderInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus = NeuralNetLayer_getShape(hReceiver, &receiverInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (DataShape_equal(&senderInputShape,&receiverInputShape) == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//受信層セット
	//---------------------------------------------------------------------------------
	pResidualConnectionSenderLayer->hReceiver = hReceiver;
	return TRUE;
}

//=====================================================================================
//  sender インターフェース取得
//=====================================================================================
static
void
NeuralNetLayerResidualConnectionSender_getExtraInterface(ResidualConnectionSenderInterface* pInterface) {
	pInterface->pSetReceiver		= NeuralNetLayerResidualConnectionSender_setReceiver;
	pInterface->pSetBackwardData	= NeuralNetLayerResidualConnectionSender_setBackwardData;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	uint32_t inputDataDim;
	if (pResidualConnectionSenderNeuralNetHeader == NULL) {
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(ResidualConnectionSenderNeuralNetLayer), uint32_t);
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
NeuralNetLayerResidualConnectionSender_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerResidualConnectionSender_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	NeuralNetLayerResidualConnectionSender_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerResidualConnectionSender_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//追加インターフェイスセット
		NeuralNetLayerResidualConnectionSender_getExtraInterface(&pResidualConnectionSenderLayer->funcTable);
		//receiver handle
		pResidualConnectionSenderLayer->hReceiver	= NULL;
		if (fEnableLearning == TRUE) {
			//layer
			pObjectWork += size_in_type(sizeof(ResidualConnectionSenderNeuralNetLayer), uint32_t);
			//バッファ
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			//pReceiverLoss
			pResidualConnectionSenderLayer->pReceiverLoss = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//バッファ初期化
			i = inputDataDim;
			while (i--) {
				pResidualConnectionSenderLayer->pReceiverLoss[i] = 0.0f;
			}
		}
		else {
			//バッファ
			pResidualConnectionSenderLayer->pReceiverLoss = NULL;
		}
		return (handle_t)pResidualConnectionSenderLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerResidualConnectionSender_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerResidualConnectionSender_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerResidualConnectionSender_construct;
	pInterface->pGetShape = NeuralNetLayerResidualConnectionSender_getShape;
	pInterface->pForward = NeuralNetLayerResidualConnectionSender_forward;
	pInterface->pBackward = NeuralNetLayerResidualConnectionSender_backward;
	pInterface->pUpdate = NeuralNetLayerResidualConnectionSender_update;
	pInterface->pInitializeParameters = NeuralNetLayerResidualConnectionSender_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerResidualConnectionSender_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_constructLayerData(
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
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(ResidualConnectionSenderNeuralNetHeader), uint32_t);
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
		pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pResidualConnectionSenderNeuralNetHeader->super, NET_LAYER_RESIDUAL_CONNECTION_SENDER, inHeight, inWidth, inChannel, sizeLayer);
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}

//=====================================================================================
//  追加インターフェイス
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_getSenderInterface(handle_t hLayer, ResidualConnectionSenderInterface* pInterface) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pNeuralNetHeader->layerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		pInterface->pSetReceiver		= NULL;
		pInterface->pSetBackwardData	= NULL;
		return FALSE;
	}
	else {
		*pInterface = pResidualConnectionSenderLayer->funcTable;
	}
	return TRUE;
}

//=====================================================================================
//=====================================================================================
// 
//  ResidualConnectionReceiver層
// 
//=====================================================================================
//=====================================================================================

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
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
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	flt32_t*	pInputData;
	flt32_t*	pInputDataReceived;
	flt32_t*	pOutputData;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//入力次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//出力バッファに、残渣接続からのデータを加えて出力する
	//---------------------------------------------------------------------------------
	pInputData = pPropagationInfo->pInputBuffer;
	pOutputData = pPropagationInfo->pOutputBuffer;
	pInputDataReceived = pResidualConnectionReceiverLayer->pX;
	size = inputDataDim;
	while (size--) {
		*pOutputData++ = (*pInputData++) + (*pInputDataReceived++);
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
NeuralNetLayerResidualConnectionReceiver_backward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	flt32_t*	pInputData;
	flt32_t*	pLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	ResidualConnectionSenderInterface	senderInterface;
	PropagationInfo	 propagationInfo;
	bool_t	fStatus;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//送信層にデータを送信
	//---------------------------------------------------------------------------------
	if (pResidualConnectionReceiverLayer->hSender != NULL) {
		propagationInfo = *pPropagationInfo;
		propagationInfo.inputBufferSize = 0;	//入力バッファ配列サイズ
		propagationInfo.pInputBuffer = NULL;	//入力バッファ
		fStatus = NeuralNetLayerResidualConnectionSender_getSenderInterface(pResidualConnectionReceiverLayer->hSender, &senderInterface);
		if (fStatus == FALSE) {
			return FALSE;
		}
		fStatus = senderInterface.pSetBackwardData(pResidualConnectionReceiverLayer->hSender, &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//lossをコピー
	//---------------------------------------------------------------------------------
	pInputData	= pPropagationInfo->pInputBuffer;
	pLoss	= pPropagationInfo->pOutputBuffer;
	size = inputDataDim;
	while (size--) {
		*pInputData++ = *pLoss++;
	}
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
NeuralNetLayerResidualConnectionReceiver_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  ResidualConnectionReceiver層
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_setForwardData(handle_t hLayer, handle_t hSender, PropagationInfo* pPropagationInfo)
{
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer*	pLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;			//base layer class
	NetLayerType	senderLayerType;
	DataShape		senderInputShape;
	DataShape		receiverInputShape;
	uint32_t		size;
	uint32_t		inHeight;
	uint32_t		inWidth;
	uint32_t		inChannel;
	uint32_t		inputDataDim;
	flt32_t*		pInputData;
	flt32_t*		pX;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//計算順序
	//---------------------------------------------------------------------------------
	pPropagationInfo->layerOrder++;
	if (pLayer->layerOrder < pPropagationInfo->layerOrder) {
		pLayer->layerOrder = pPropagationInfo->layerOrder;
	}
	//---------------------------------------------------------------------------------
	//層タイプ取得
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getType(hSender, &senderLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (senderLayerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層入力形状取得
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getShape(hSender, &senderInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus = NeuralNetLayer_getShape(hLayer, &receiverInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (DataShape_equal(&senderInputShape, &receiverInputShape) == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	pResidualConnectionReceiverLayer->hSender = hSender;
	//---------------------------------------------------------------------------------
	//入力次元
	//---------------------------------------------------------------------------------
	inHeight = receiverInputShape.height;
	inWidth = receiverInputShape.width;
	inChannel = receiverInputShape.channel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//次元チェック
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//誤差伝搬先バッファー
	//---------------------------------------------------------------------------------
	size = inputDataDim;
	pInputData = pPropagationInfo->pInputBuffer;
	pX = pResidualConnectionReceiverLayer->pX;
	while (size--) {
		*pX++ = *pInputData++;
	}
	return TRUE;
}

//=====================================================================================
//  receiver インターフェース取得
//=====================================================================================
static
void
NeuralNetLayerResidualConnectionReceiver_getExtraInterface(ResidualConnectionReceiverInterface* pInterface) {
	pInterface->pSetForwardData = NeuralNetLayerResidualConnectionReceiver_setForwardData;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_getLayerInformation(
	uint32_t* pLayerData,
	bool_t		fEnableLearning,
	uint32_t* pLayerObjectSizeIn32BitWord,
	uint32_t* pNumberOfLearningParameters,
	uint32_t* pTempWorkAreaSizeIn32BitWord,
	DataShape* pInputShape,
	DataShape* pOutputShape) {
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	uint32_t inputDataDim;
	if (pResidualConnectionReceiverNeuralNetHeader == NULL) {
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(ResidualConnectionReceiverNeuralNetLayer), uint32_t);
		//X
		inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
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
NeuralNetLayerResidualConnectionReceiver_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerResidualConnectionReceiver_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	NeuralNetLayerResidualConnectionReceiver_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerResidualConnectionReceiver_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//追加インターフェイスセット
		NeuralNetLayerResidualConnectionReceiver_getExtraInterface(&pResidualConnectionReceiverLayer->funcTable);
		//layer
		pObjectWork += size_in_type(sizeof(ResidualConnectionReceiverNeuralNetLayer), uint32_t);
		//バッファ
		inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		//pX
		pResidualConnectionReceiverLayer->pX = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
		//sender
		pResidualConnectionReceiverLayer->hSender = NULL;
		//バッファ初期化
		i = inputDataDim;
		while (i--) {
			pResidualConnectionReceiverLayer->pX[i] = 0.0f;
		}
		return (handle_t)pResidualConnectionReceiverLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerResidualConnectionReceiver_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerResidualConnectionReceiver_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerResidualConnectionReceiver_construct;
	pInterface->pGetShape = NeuralNetLayerResidualConnectionReceiver_getShape;
	pInterface->pForward = NeuralNetLayerResidualConnectionReceiver_forward;
	pInterface->pBackward = NeuralNetLayerResidualConnectionReceiver_backward;
	pInterface->pUpdate = NeuralNetLayerResidualConnectionReceiver_update;
	pInterface->pInitializeParameters = NeuralNetLayerResidualConnectionReceiver_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerResidualConnectionReceiver_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_constructLayerData(
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
	uint32_t	inputDataDim;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader;
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
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(ResidualConnectionReceiverNeuralNetHeader), uint32_t);
	//sizeOfBuffer = size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
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
		pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pResidualConnectionReceiverNeuralNetHeader->super, NET_LAYER_RESIDUAL_CONNECTION_RECEIVER, inHeight, inWidth, inChannel, sizeLayer);
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

//=====================================================================================
//  追加インターフェイス
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(handle_t hLayer, ResidualConnectionReceiverInterface* pInterface) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pNeuralNetHeader->layerType != NET_LAYER_RESIDUAL_CONNECTION_RECEIVER) {
		pInterface->pSetForwardData = NULL;
		return FALSE;
	}
	else {
		*pInterface = pResidualConnectionReceiverLayer->funcTable;
	}
	return TRUE;
}
