#ifndef NetworkNetModelConstructor_H
#define NetworkNetModelConstructor_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "SequentialNet.h"

//--------------------------------------------------------------------
// レイヤー情報構造体
//--------------------------------------------------------------------
#define LAYER_INFORMATION_MAX_PARAMS	(10)

typedef struct tagLayerInformation {
	NetLayerType	layerType;
	uint32_t		parameterArray[LAYER_INFORMATION_MAX_PARAMS];
	uint32_t		numberOfParams;
} LayerInformation;

//--------------------------------------------------------------------
// シーケンシャルモデル情報構造体
//--------------------------------------------------------------------
#define MAX_LAYERS	(200)
typedef struct tagModelInformation {
	uint32_t			inHeight;
	uint32_t			inWidth;
	uint32_t			inChannel;
	uint32_t			numberOfLayers;
	LayerInformation	layerArray[MAX_LAYERS];
} ModelInformation;

//--------------------------------------------------------------------
// シーケンシャルモデルイメージを作成する関数
//--------------------------------------------------------------------
bool_t	
constructNeuralNetModel(	ModelInformation*	pModelInformation,	//ネットワーク情報
							uint32_t**			ppNeuralNetworkImage,	//生成するニューラルネットワークイメージのポインタのポインタ
							uint32_t*			pSizeOfImageIn32BitWord	//生成するニューラルネットワークイメージのサイズを取得するパラメタのポインタ
);

//--------------------------------------------------------------------
// シーケンシャルモデルのレイヤー情報をセットするヘルパー関数
//--------------------------------------------------------------------
void	sequential_model_header(ModelInformation* pModelInformation, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);
bool_t	dense(ModelInformation* pModelInformation, uint32_t units);
bool_t	simple_rnn(ModelInformation* pModelInformation, uint32_t units, uint32_t activation, uint32_t retuenSequence);
bool_t	activation(ModelInformation* pModelInformation, uint32_t activation);
bool_t	conv2d(ModelInformation* pModelInformation, uint32_t filters,uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,bool_t	fPadding);
bool_t	depthwise_conv2d(ModelInformation* pModelInformation, uint32_t nfilters, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width, bool_t fPadding);
bool_t	pointwise_conv2d(ModelInformation* pModelInformation, uint32_t pw_filters);
bool_t	max_pooling2d(ModelInformation* pModelInformation, uint32_t pool_height, uint32_t pool_width, uint32_t stride_height, uint32_t stride_width);
bool_t	global_average_pooling2d(ModelInformation* pModelInformation);
bool_t	batch_normalization(ModelInformation* pModelInformation);
bool_t	layer_normalization(ModelInformation* pModelInformation);
bool_t	predeconv2d(ModelInformation* pModelInformation,uint32_t stride_height,uint32_t stride_width,uint32_t out_height,uint32_t out_width);
bool_t	residual_connection_sender(ModelInformation* pModelInformation);
bool_t	residual_connection_receiver(ModelInformation* pModelInformation);

//--------------------------------------------------------------------
// シーケンシャルモデル定義テキストファイルからレイヤー情報を取得する
//--------------------------------------------------------------------
void	getLayerInformation(char* pLayerInformationText, uint32_t* pInputDim, LayerInformation* pLayerInformationArray, uint32_t* pNumberOfLayers);

//--------------------------------------------------------------------
// 層情報から層イメージを作成する
//--------------------------------------------------------------------
bool_t	constructLayer(uint32_t* pBuffer, uint32_t bufferSizeIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, LayerInformation* pLayerInformation, uint32_t* pLayerImageSizeIn32BitWord, char* pLayerName);

//--------------------------------------------------------------------
// 定義ファイルを読取モデルを作成する
//--------------------------------------------------------------------
bool_t	constructNeuralNetModelByFile(const char* pNetworkDefinitionFileName, uint32_t** ppNeuralNetworkImage, uint32_t* pSizeOfNeuralNetworkImageIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
