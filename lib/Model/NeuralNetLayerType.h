
#ifndef NEURAL_NET_LAYER_TYPE_H
#define NEURAL_NET_LAYER_TYPE_H

#include "STDTypeDefinition.h"

//=====================================================================================
//  層のタイプ
//=====================================================================================
typedef enum tagNetLayerType {
	NET_LAYER_UNKNOWN						= 0,
	NET_LAYER_DENSE							= 1,
	NET_LAYER_SIMPLE_RNN					= 2,
	NET_LAYER_CONV2D						= 3,
	NET_LAYER_DEPTHWISE_CONV2D				= 4,
	NET_LAYER_POINTWISE_CONV2D				= 5,
	NET_LAYER_MAX_POOLING2D					= 6,
	NET_LAYER_BATCH_NORMALIZATION			= 7,
	NET_LAYER_ACTIVATION					= 8,
	NET_LAYER_LAYER_NORMALIZATION			= 9,
	NET_LAYER_PREDECONV2D					= 10,
	NET_LAYER_RESIDUAL_CONNECTION_SENDER	= 11,
	NET_LAYER_RESIDUAL_CONNECTION_RECEIVER	= 12,
	NET_LAYER_GLOBAL_AVERAGE_POOLING2D		= 13
} NetLayerType;

//=====================================================================================
//  データ形状
//=====================================================================================
typedef struct tagDataShape {
	uint32_t	height;
	uint32_t	width;
	uint32_t	channel;
} DataShape;

void		DataShape_construct(DataShape* pDataShape, uint32_t	height, uint32_t width, uint32_t channel);
void		DataShape_initialize(DataShape* pDataShape);
uint32_t	DataShape_getSize(const DataShape* pDataShape);
bool_t		DataShape_equal(const DataShape* pDataShape, const DataShape* pDataShapeToCompare);

//=====================================================================================
//  伝搬情報
//=====================================================================================
typedef struct tagPropagationInfo {
	DataShape	dataShape;				//入出力データ形状（伝搬前後で変化）
	uint32_t	layerOrder;				//計算順序
	uint32_t	inputBufferSize;		//入力バッファ配列サイズ
	uint32_t	outputBufferSize;		//出力バッファ配列サイズ
	uint32_t	temporaryBufferSize;	//一時バッファ配列サイズ
	flt32_t*	pInputBuffer;			//入力バッファ
	flt32_t*	pOutputBuffer;			//出力バッファ
	uint32_t*	pTemporaryBuffer;		//一時計算バッファ
} PropagationInfo;

//=====================================================================================
//  funcTable定義
//=====================================================================================
typedef bool_t(*NetLayer_getLayerInformation)		(
	uint32_t*	pLayerData,						// in:image data
	bool_t		fEnableLearning,				// in:back propagation flag
	uint32_t*	pLayerObjectSizeIn32BitWord,	// out:layer object size
	uint32_t*	pNumberOfLearningParameters,	// out:number of learning prameters for optimizer
	uint32_t*	pTempWorkAreaSizeIn32BitWord,	// out:temporary work area size for prediction and back propagation
	DataShape*	pInputShape,					// out:input data shape
	DataShape*	pOutputShape					// out:output data shape
	);

typedef handle_t(*NetLayer_construct)		(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer
	);

//=====================================================================================
//  インターフェイス定義
//=====================================================================================
typedef bool_t(*NetLayer_getShape)				(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape);
typedef bool_t(*NetLayer_forward)				(handle_t hLayer, PropagationInfo* pPropagationInfo);
typedef bool_t(*NetLayer_backward)				(handle_t hLayer, PropagationInfo* pPropagationInfo);
typedef bool_t(*NetLayer_update)				(handle_t hLayer);
typedef bool_t(*NetLayer_initializeParameters)	(handle_t hLayer, handle_t hRandomValueGenerator);
typedef bool_t(*NetLayer_getParameters)			(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters);

typedef struct tagLayerFuncTable {
	NetLayer_getLayerInformation	pGetLayerInformation;
	NetLayer_construct				pConstruct;
	NetLayer_getShape				pGetShape;
	NetLayer_forward				pForward;
	NetLayer_backward				pBackward;
	NetLayer_update					pUpdate;
	NetLayer_initializeParameters	pInitializeParameters;
	NetLayer_getParameters			pGetParameters;
} LayerFuncTable;

#endif
