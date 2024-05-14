#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NeuralNetLayerType.h"
#include "NeuralNetModelConstructor.h"
#include "LOG_Function.h"

//--------------------------------------------------------------------
// レイヤーイメージデータを構築する
//--------------------------------------------------------------------
bool_t
constructLayer(uint32_t* pBuffer, uint32_t bufferSizeIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, LayerInformation* pLayerInformation, uint32_t* pLayerImageSizeIn32BitWord, char* pLayerName) {
	uint32_t	unit;
	NeuralNetActivationType activation;
	bool_t		returnSequence;
	uint32_t	nFilter;
	uint32_t	kernelHeight;
	uint32_t	kernelWidth;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	poolingHeight;
	uint32_t	poolinhWidth;
	bool_t		fPadding;
	bool_t		fStatus;
	//-----------------------------------------------------
	//　Layer
	//-----------------------------------------------------
	fStatus = FALSE;
	switch (pLayerInformation->layerType) {
	case NET_LAYER_DENSE:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "dense               ");
		}
		unit	= pLayerInformation->parameterArray[0];
		fStatus = SequentialNet_appendDense(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, unit, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_SIMPLE_RNN:
		unit			= pLayerInformation->parameterArray[0];
		activation		= (NeuralNetActivationType)pLayerInformation->parameterArray[1];
		returnSequence	= (bool_t)pLayerInformation->parameterArray[2];
		fStatus			= SequentialNet_appendSimpleRNN(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, unit, activation, returnSequence, pLayerImageSizeIn32BitWord);
		if (pLayerName != NULL) {
			switch (activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				strcpy(pLayerName, "simple rnn relu     ");
				break;
			case NEURAL_NET_ACTIVATION_TANH:
				strcpy(pLayerName, "simple rnn tanh     ");
				break;
			case NEURAL_NET_ACTIVATION_SIGMOID:
				strcpy(pLayerName, "simple rnn sigmoid  ");
				break;
			default:
				strcpy(pLayerName, "simple rnn          ");
				break;
			}
		}
		break;
	case NET_LAYER_CONV2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "conv2d              ");
		}
		nFilter			= pLayerInformation->parameterArray[0];
		kernelHeight	= pLayerInformation->parameterArray[1];
		kernelWidth		= pLayerInformation->parameterArray[2];
		strideHeight	= pLayerInformation->parameterArray[3];
		strideWidth		= pLayerInformation->parameterArray[4];
		fPadding		= pLayerInformation->parameterArray[5];
		fStatus			= SequentialNet_appendConv2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, nFilter, kernelHeight, kernelWidth, strideHeight, strideWidth, fPadding, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_DEPTHWISE_CONV2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "depthwise conv2d    ");
		}
		nFilter			= pLayerInformation->parameterArray[0];
		kernelHeight	= pLayerInformation->parameterArray[1];
		kernelWidth		= pLayerInformation->parameterArray[2];
		strideHeight	= pLayerInformation->parameterArray[3];
		strideWidth		= pLayerInformation->parameterArray[4];
		fPadding		= pLayerInformation->parameterArray[5];
		fStatus			= SequentialNet_appendDepthwiseConv2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, nFilter, kernelHeight, kernelWidth, strideHeight, strideWidth, fPadding,pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_POINTWISE_CONV2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "pointwise conv2d    ");
		}
		nFilter		= pLayerInformation->parameterArray[0];
		fStatus		= SequentialNet_appendPointwiseConv2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, nFilter, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_MAX_POOLING2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "max pooling2d       ");
		}
		poolingHeight	= pLayerInformation->parameterArray[0];
		poolinhWidth	= pLayerInformation->parameterArray[1];
		strideHeight	= pLayerInformation->parameterArray[2];
		strideWidth		= pLayerInformation->parameterArray[3];
		fStatus			= SequentialNet_appendMaxPooling2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, poolingHeight, poolinhWidth, strideHeight, strideWidth, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_GLOBAL_AVERAGE_POOLING2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "global ave pooling2d");
		}
		fStatus = SequentialNet_appendGlobalAveragePooling2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_BATCH_NORMALIZATION:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "batch normalization ");
		}
		fStatus = SequentialNet_appendBatchNormalization(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_LAYER_NORMALIZATION:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "layer normalization ");
		}
		fStatus = SequentialNet_appendLayerNormalization(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_ACTIVATION:
		activation	= (NeuralNetActivationType)pLayerInformation->parameterArray[0];
		fStatus		= SequentialNet_appendActivation(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, activation, pLayerImageSizeIn32BitWord);
		if (pLayerName != NULL) {
			switch (activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				strcpy(pLayerName, "activation relu     ");
				break;
			case NEURAL_NET_ACTIVATION_TANH:
				strcpy(pLayerName, "activation tanh     ");
				break;
			case NEURAL_NET_ACTIVATION_SIGMOID:
				strcpy(pLayerName, "activation sigmoid  ");
				break;
			case NEURAL_NET_ACTIVATION_SOFTMAX:
				strcpy(pLayerName, "activation softmax  ");
				break;
			default:
				strcpy(pLayerName, "activation          ");
				break;
			}
		}
		break;
	case NET_LAYER_PREDECONV2D:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "predeconv2d         ");
		}
		strideHeight = pLayerInformation->parameterArray[0];
		strideWidth = pLayerInformation->parameterArray[1];
		outHeight = pLayerInformation->parameterArray[2];
		outWidth = pLayerInformation->parameterArray[3];
		fStatus = SequentialNet_appendPreDeconv2D(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, strideHeight, strideWidth, outHeight, outWidth, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_SENDER:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "residual sender     ");
		}
		fStatus = SequentialNet_appendResidualConnectionSender(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pLayerImageSizeIn32BitWord);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_RECEIVER:
		if (pLayerName != NULL) {
			strcpy(pLayerName, "residual receiver   ");
		}
		fStatus = SequentialNet_appendResidualConnectionReceiver(pBuffer, bufferSizeIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pLayerImageSizeIn32BitWord);
		break;
	default:
		fStatus = FALSE;
		break;
	}
	return fStatus;
}

//--------------------------------------------------------------------
// シーケンシャルモデルイメージを作成する内部関数
//--------------------------------------------------------------------
static
bool_t
constructNeuralNetModelImage(	uint32_t			inHeight,
								uint32_t			inWidth,
								uint32_t			inChannel,
								uint32_t			numberOfLayers,
								LayerInformation*	pLayerInformationArray,
								uint32_t**			ppNeuralNetworkImage, 
								uint32_t*			pSizeOfImageIn32BitWord)
{
	uint32_t	j;
	uint32_t	height;
	uint32_t	width;
	uint32_t	channel;
	uint32_t	sizeOfTotalImageIn32BitWord;
	uint32_t	sizeOfImageIn32BitWord;
	uint32_t*	pDynamicNetworkImage;
	uint32_t*	pBuffer;
	uint32_t	bufferRemainSize;
	char		strLayer[50];
	char		strInformation[200];
	bool_t		fStatus;
	//-------------------------------------------------------------------------
	//　パラメタ初期化
	//-------------------------------------------------------------------------
	*ppNeuralNetworkImage = NULL;
	*pSizeOfImageIn32BitWord = 0;
	//=========================================================================
	//　シーケンシャルモデルサイズ計算
	//=========================================================================
	sizeOfTotalImageIn32BitWord = 0;
	sizeOfImageIn32BitWord = 0;
	height = inHeight;
	width = inWidth;
	channel = inChannel;
	//-------------------------------------------------------------------------
	//　シーケンシャルモデルイメージデータサイズの計算：ヘッダーサイズ
	//-------------------------------------------------------------------------
	fStatus = SequentialNet_createHeader(NULL, 0, height, width, channel, numberOfLayers, &sizeOfImageIn32BitWord);
	if (fStatus == FALSE) {
		printf("error in SequentialNet_createHeader\n");
		return FALSE;
	}
	sizeOfTotalImageIn32BitWord += sizeOfImageIn32BitWord;
	//-----------------------------------------------------
	//　title
	//-----------------------------------------------------
	sprintf(strInformation, "type                 \toutput\tsize\n", height, width, channel);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	sprintf(strInformation, "--------------------------------------\n", height, width, channel);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//-----------------------------------------------------
	//　入力次元
	//-----------------------------------------------------
	sprintf(strInformation, "input               :\t%dx%dx%d\n", height, width, channel);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//-------------------------------------------------------------------------
	//　シーケンシャルモデルイメージデータサイズの計算：レイヤーイメージ計算
	//-------------------------------------------------------------------------
	for (j = 0; j < numberOfLayers; j++) {
		//-----------------------------------------------------
		// レイヤー情報取得
		//-----------------------------------------------------
		fStatus = constructLayer(NULL, 0, &height, &width, &channel, &pLayerInformationArray[j], &sizeOfImageIn32BitWord, strLayer);
		if (fStatus == FALSE) {
			printf("error in building layerArray\n");
			return FALSE;
		}
		//-----------------------------------------------------
		// レイヤータイプ
		//-----------------------------------------------------
		sprintf(strInformation, "%3d : %s:\t",j, strLayer);
		//-----------------------------------------------------
		// レイヤーから出力されるデータの次元数
		//-----------------------------------------------------
		sprintf(strInformation + strlen(strInformation), "%dx%dx%d\t", height, width, channel);
		//-----------------------------------------------------
		// レイヤーイメージデータサイズ（バイト）
		//-----------------------------------------------------
		sprintf(strInformation + strlen(strInformation), "%d\n", sizeOfImageIn32BitWord * sizeof(uint32_t));
		//-----------------------------------------------------
		// レイヤー情報出力
		//-----------------------------------------------------
		SAVE_LOG_WITHOUT_RETURN(strInformation);
		//-----------------------------------------------------
		// 合計シーケンシャルモデルサイズ
		//-----------------------------------------------------
		sizeOfTotalImageIn32BitWord += sizeOfImageIn32BitWord;
	}
	//-----------------------------------------------------
	//　区切り
	//-----------------------------------------------------
	sprintf(strInformation, "--------------------------------------\n");
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//=========================================================================
	//　シーケンシャルモデルイメージ作成
	//=========================================================================
	//-------------------------------------------------------------------------
	//　シーケンシャルモデルイメージバッファ確保
	//-------------------------------------------------------------------------
	pDynamicNetworkImage = (uint32_t*)malloc(sizeof(uint32_t) * sizeOfTotalImageIn32BitWord);
	if (pDynamicNetworkImage == NULL) {
		printf("memory allocation failed for neural network image, size:%d byte\n", sizeof(uint32_t) * sizeOfTotalImageIn32BitWord);
		return FALSE;
	}
	//-------------------------------------------------------------------------
	//　シーケンシャルモデルイメージ領域情報
	//-------------------------------------------------------------------------
	pBuffer = pDynamicNetworkImage;
	bufferRemainSize = sizeOfTotalImageIn32BitWord;
	//-------------------------------------------------------------------------
	//　ヘッダー作成
	//-------------------------------------------------------------------------
	height = inHeight;
	width = inWidth;
	channel = inChannel;
	fStatus = SequentialNet_createHeader(pBuffer, bufferRemainSize, height, width, channel, numberOfLayers, &sizeOfImageIn32BitWord);
	if (fStatus == FALSE) {
		printf("error in SequentialNet_createHeader\n");
		return FALSE;
	}
	pBuffer += sizeOfImageIn32BitWord;
	bufferRemainSize -= sizeOfImageIn32BitWord;
	//-------------------------------------------------------------------------
	//　レイヤーイメージ作成
	//-------------------------------------------------------------------------
	for (j = 0; j < numberOfLayers; j++) {
		char strLayer[100];
		fStatus = constructLayer(pBuffer, bufferRemainSize, &height, &width, &channel, &pLayerInformationArray[j], &sizeOfImageIn32BitWord, strLayer);
		if (fStatus == FALSE) {
			printf("error in bulding layerArray image:%d\n", j);
			return FALSE;
		}
		pBuffer += sizeOfImageIn32BitWord;
		bufferRemainSize -= sizeOfImageIn32BitWord;
	}
	//-------------------------------------------------------------------------
	//　シーケンシャルモデルイメージデータをセットする
	//-------------------------------------------------------------------------
	*ppNeuralNetworkImage = pDynamicNetworkImage;
	*pSizeOfImageIn32BitWord = sizeOfTotalImageIn32BitWord;
	return TRUE;
}

//--------------------------------------------------------------------
// シーケンシャルモデルイメージを作成する関数
//--------------------------------------------------------------------
bool_t	
constructNeuralNetModel(ModelInformation* pModelInformation,uint32_t** ppNeuralNetworkImage,uint32_t* pSizeOfImageIn32BitWord) {
	return constructNeuralNetModelImage(pModelInformation->inHeight, pModelInformation->inWidth,pModelInformation->inChannel,pModelInformation->numberOfLayers,pModelInformation->layerArray,ppNeuralNetworkImage,pSizeOfImageIn32BitWord);
}

//--------------------------------------------------------------------
// シーケンシャルモデルのレイヤー情報をセットするヘルパー関数
//--------------------------------------------------------------------
void
sequential_model_header(ModelInformation* pModelInformation, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel) {
	pModelInformation->inHeight = inHeight;
	pModelInformation->inWidth = inWidth;
	pModelInformation->inChannel = inChannel;
	pModelInformation->numberOfLayers = 0;
}

bool_t
dense(ModelInformation* pModelInformation, uint32_t units) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_DENSE;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = units;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
simple_rnn(ModelInformation* pModelInformation,uint32_t units,uint32_t activation, uint32_t retuenSequence) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_SIMPLE_RNN;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = units;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[1] = activation;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[2] = retuenSequence;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
activation(ModelInformation* pModelInformation, uint32_t activation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_ACTIVATION;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = activation;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
conv2d(ModelInformation* pModelInformation,
	uint32_t	filters,
	uint32_t	kernel_height,
	uint32_t	kernel_width,
	uint32_t	stride_height,
	uint32_t	stride_width,
	bool_t		fPadding) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_CONV2D;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = filters;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[1] = kernel_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[2] = kernel_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[3] = stride_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[4] = stride_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[5] = fPadding;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
depthwise_conv2d(ModelInformation* pModelInformation,
	uint32_t	nfilters,
	uint32_t	kernel_height,
	uint32_t	kernel_width,
	uint32_t	stride_height,
	uint32_t	stride_width,
	bool_t		fPadding) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_DEPTHWISE_CONV2D;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = nfilters;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[1] = kernel_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[2] = kernel_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[3] = stride_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[4] = stride_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[5] = fPadding;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
pointwise_conv2d(ModelInformation* pModelInformation,
	uint32_t	pw_filters) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_POINTWISE_CONV2D;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = pw_filters;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
max_pooling2d(ModelInformation* pModelInformation,
	uint32_t	pool_height,
	uint32_t	pool_width,
	uint32_t	stride_height,
	uint32_t	stride_width) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_MAX_POOLING2D;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = pool_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[1] = pool_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[2] = stride_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[3] = stride_width;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
global_average_pooling2d(ModelInformation* pModelInformation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_GLOBAL_AVERAGE_POOLING2D;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
batch_normalization(ModelInformation* pModelInformation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_BATCH_NORMALIZATION;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
layer_normalization(ModelInformation* pModelInformation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_LAYER_NORMALIZATION;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
predeconv2d(ModelInformation* pModelInformation,
	uint32_t	stride_height,
	uint32_t	stride_width,
	uint32_t	out_height,
	uint32_t	out_width) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_PREDECONV2D;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[0] = stride_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[1] = stride_width;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[2] = out_height;
	pModelInformation->layerArray[pModelInformation->numberOfLayers].parameterArray[3] = out_width;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
residual_connection_sender(ModelInformation* pModelInformation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_RESIDUAL_CONNECTION_SENDER;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

bool_t
residual_connection_receiver(ModelInformation* pModelInformation) {
	if (pModelInformation->numberOfLayers == MAX_LAYERS) {
		printf("number of layers exceeded the limit\n");
		return FALSE;
	}
	pModelInformation->layerArray[pModelInformation->numberOfLayers].layerType = NET_LAYER_RESIDUAL_CONNECTION_RECEIVER;
	pModelInformation->numberOfLayers++;
	return TRUE;
}

//--------------------------------------------------------------------
// レイヤーパラメタの取得
//--------------------------------------------------------------------
static
void
getLayerParameters(char* pLayerInformationText, uint32_t* pParameters, uint32_t* pNumberOfParameters) {
	char* pLast;
	char* pHead = pLayerInformationText;
	char* pParam;
	int	length = 0;
	while (*pHead != '\0' && *pHead != '\n') {
		length++;
		pHead++;
	}
	pLast = pLayerInformationText + length;
	pHead = pLayerInformationText;
	pParam = pHead;
	*pNumberOfParameters = 0;
	while (pParam <= pLast) {
		if (*pParam == ',' || *pParam == '\n' || *pParam == '\0') {
			*pParam = '\0';
			if (pHead != pParam) {
				pParameters[*pNumberOfParameters] = atoi(pHead);
				(*pNumberOfParameters)++;
			}
			pHead = pParam + 1;
			pParam = pHead;
		}
		else {
			pParam++;
		}
	}
}

//--------------------------------------------------------------------
// 文字列比較
//--------------------------------------------------------------------
static
bool_t
compareString(char* pString, char* pStringCompared, char** ppParameters) {
	int i;
	int length = 0;
	int lengthCompred = 0;
	char* pHead = pString;
	while (*pHead++ != '\0') {
		length++;
	}
	pHead = pStringCompared;
	while (*pHead++ != '\0') {
		lengthCompred++;
	}
	if (length < lengthCompred) {
		return FALSE;
	}
	for (i = 0; i < lengthCompred; i++) {
		if (pString[i] != pStringCompared[i]) {
			return FALSE;
		}
	}
	if (length > (lengthCompred + 1)) {
		*ppParameters = pString + lengthCompred + 1;
	}
	else {
		//パラメタがない
		*ppParameters = pString + length;	//'\0'
	}
	return TRUE;
}

//--------------------------------------------------------------------
// シーケンシャルモデル定義テキストファイルからレイヤー情報を取得する
//--------------------------------------------------------------------
void
getLayerInformation(char* pLayerInformationText, uint32_t* pInputDim, LayerInformation* pLayerInformationArray, uint32_t* pNumberOfLayers) {
	char* pLast;
	char* pHead = pLayerInformationText;
	char* pParam;
	int		length = 0;
	int		lineLength;
	char* pLayerParameters;
	LayerInformation* pLayerInformation = pLayerInformationArray;
	uint32_t numberOfParams = 0;
	while (*pHead++ != '\0') {
		length++;
	}
	pLast = pLayerInformationText + length;
	pHead = pLayerInformationText;
	pParam = pHead;
	lineLength = 0;
	*pNumberOfLayers = 0;
	while (pParam <= pLast) {
		if (*pParam == '\n') {
			*pParam = '\0';
			if (pHead != pParam) {
				if (pHead[0] != '/' && pHead[1] != '/') {
					lineLength = pParam - pHead;
					if (compareString(pHead, "input\0", &pLayerParameters) == TRUE) {
						getLayerParameters(pLayerParameters, pInputDim, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
					}
					else if (compareString(pHead, "dense\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_DENSE;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "simple_rnn\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_SIMPLE_RNN;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "conv2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_CONV2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "depthwise_conv2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_DEPTHWISE_CONV2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "pointwise_conv2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_POINTWISE_CONV2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "max_pooling2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_MAX_POOLING2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "global_average_pooling2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_GLOBAL_AVERAGE_POOLING2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "batch_normalization\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_BATCH_NORMALIZATION;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "layer_normalization\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_LAYER_NORMALIZATION;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "activation\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_ACTIVATION;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "predeconv2d\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_PREDECONV2D;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "residual_connection_sender\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_RESIDUAL_CONNECTION_SENDER;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
					else if (compareString(pHead, "residual_connection_receiver\0", &pLayerParameters) == TRUE) {
						pLayerInformation->layerType = NET_LAYER_RESIDUAL_CONNECTION_RECEIVER;
						getLayerParameters(pLayerParameters, pLayerInformation->parameterArray, &numberOfParams);
						pLayerInformation->numberOfParams = numberOfParams;
						(*pNumberOfLayers)++;
						pLayerInformation++;
					}
				}
			}
			pHead = pParam + 1;
			pParam = pHead;
		}
		else {
			pParam++;
		}
	}
}

//--------------------------------------------------------------------
// 定義ファイルを読取り、シーケンシャルモデルを作成する
//--------------------------------------------------------------------
bool_t
constructNeuralNetModelByFile(const char* pNetworkDefinitionFileName, uint32_t** ppNeuralNetworkImage, uint32_t* pSizeOfNeuralNetworkImageIn32BitWord)
{
	FILE* fp;
	int					fileSizeInByte;
	char*				pNeuralNetworkParameterText;
	LayerInformation	layerInformationArray[100];
	uint32_t			inputDimension[10];
	uint32_t			numberOfLayers = 0;
	uint32_t			inputHeight;
	uint32_t			inputWidth;
	uint32_t			inputChannel;
	bool_t				fStatus;
	*ppNeuralNetworkImage = NULL;
	*pSizeOfNeuralNetworkImageIn32BitWord = 0;
	memset(inputDimension, 0, sizeof(inputDimension));
	//-------------------------------------------------
	//	シーケンシャルモデルファイルオープン
	//-------------------------------------------------
	fp = fopen(pNetworkDefinitionFileName, "r");
	if (fp == NULL) {
		printf("neural network definition file could not found : %s\n", pNetworkDefinitionFileName);
		return FALSE;
	}
	//-------------------------------------------------
	//	データサイズ取得
	//-------------------------------------------------
	fseek(fp, 0, SEEK_END);
	fileSizeInByte = ftell(fp);
	//-------------------------------------------------
	//	シーケンシャルモデルデータ読み込み
	//-------------------------------------------------
	fseek(fp, 0, SEEK_SET);
	pNeuralNetworkParameterText = (char*)malloc(sizeof(char) * (fileSizeInByte + 1));
	fread(pNeuralNetworkParameterText, sizeof(char), fileSizeInByte, fp);
	pNeuralNetworkParameterText[fileSizeInByte] = '\0';
	//-------------------------------------------------
	//	ファイルクローズ
	//-------------------------------------------------
	fclose(fp);
	//-----------------------------------------------
	//　レイヤー情報を取得する
	//-----------------------------------------------
	numberOfLayers = 0;
	getLayerInformation(pNeuralNetworkParameterText, inputDimension, layerInformationArray, &numberOfLayers);
	free(pNeuralNetworkParameterText);
	//-----------------------------------------------
	//　シーケンシャルモデルイメージ作成
	//-----------------------------------------------
	inputHeight = inputDimension[0];
	inputWidth = inputDimension[1];
	inputChannel = inputDimension[2];
	fStatus = constructNeuralNetModelImage(inputHeight, inputWidth, inputChannel, numberOfLayers, layerInformationArray, ppNeuralNetworkImage, pSizeOfNeuralNetworkImageIn32BitWord);
	return TRUE;
}