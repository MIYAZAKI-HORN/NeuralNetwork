#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerSimpleRNN.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  SimpleRNN層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagSimpleRNNNeuralNetHeader {
	NeuralNetHeader			super;			//base layer header
	uint32_t				unit;			//ユニット数
	NeuralNetActivationType	activation;		//活性化関数
	bool_t					returnSequence;	//リターンシーケンス　TRUE：全時間　FALSE：最終時間
} SimpleRNNNeuralNetHeader;

//=====================================================================================
//  SimpleRNN層構造体
//=====================================================================================
typedef struct tagSimpleRNNNeuralNetLayer {
	NeuralNetLayer	super;					//base layer class
	flt32_t*		pX;						//誤差逆伝搬用データバッファ：Wへの最終入力　timeStepsの入力値
	flt32_t*		pH;						//誤差逆伝搬用データバッファ：出力値　timeSteps分の出力値
	flt32_t*		pBPData;				//誤差逆伝搬用データバッファ：内部活性化関数への最終入力または内部活性化関数からの最終出力　timeSteps分の出力値
	flt32_t*		pTimeLossBuffer;		//出力サイズの逆伝播作業用バッファー：1時間分の出力値
	uint32_t		maxPropagationTime;		//最大誤差伝播時間
	handle_t		hOptimizer;				//オプティマイザーハンドル
} SimpleRNNNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	pInputShape->height = pNeuralNetHeader->inHeight;
	pInputShape->width = pNeuralNetHeader->inWidth;
	pInputShape->channel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		pOutputShape->height = timeSteps;
	}
	else {
		pOutputShape->height = 1;
	}
	pOutputShape->width = pSimpleRNNNeuralNetHeader->unit;
	pOutputShape->channel = 1;
	return TRUE;
}

//=====================================================================================
//  順伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_forward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps	= pNeuralNetHeader->inHeight;
	uint32_t	nInputDim	= pNeuralNetHeader->inWidth;
	uint32_t	i;
	uint32_t	t;
	flt32_t*	pW;
	flt32_t*	pU;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	flt32_t*	pTimeIn;
	flt32_t*	pTimeOut;
	uint32_t	outputBufferSize;
	flt32_t*	pInternalOutputBuffer;
	flt32_t*	pUnitOut;
	flt32_t*	pInput;
	flt32_t*	pX;
	flt32_t*	pH;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//出力バッファーを初期化
	//---------------------------------------------------------------------------------
	outputBufferSize = timeSteps * pSimpleRNNNeuralNetHeader->unit;
	pInternalOutputBuffer = (flt32_t*)pPropagationInfo->pTemporaryBuffer;
	pUnitOut = pInternalOutputBuffer;
	i = outputBufferSize;
	while (i--) {
		*pUnitOut++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//timeSteps分計算
	//---------------------------------------------------------------------------------
	pTimeIn = pPropagationInfo->pInputBuffer;
	for (t = 0; t < timeSteps; t++) {
		//------------------------------------------------------
		// 誤差逆伝搬：Wブロックへの入力
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE) {
			pInput = pTimeIn;
			pX = pSimpleRNNLayer->pX + t * nInputDim;
			size = nInputDim;
			while (size--) {
				*pX++ = *pInput++;
			}
		}
		//------------------------------------------------------
		//tフレームに出力する
		//------------------------------------------------------
		pTimeOut = pInternalOutputBuffer + t * pSimpleRNNNeuralNetHeader->unit;
		//------------------------------------------------------
		//W・in[t]＋B
		//------------------------------------------------------
		weight_matrix_with_bias_forward(pTimeIn, nInputDim, pW, pB, pTimeOut, pSimpleRNNNeuralNetHeader->unit,FALSE);
		//------------------------------------------------------
		//U・h[t-1]
		//------------------------------------------------------
		if (t > 0) {
			//------------------------------------------------------
			//前の時刻の出力値 h(t-1) を入力とする
			//------------------------------------------------------
			pUnitOut = pInternalOutputBuffer + (t - 1) * pSimpleRNNNeuralNetHeader->unit;
			//------------------------------------------------------
			//pTimeOutの値に（t-1）出力にUに対する重み計算値が加算される
			//------------------------------------------------------
			weight_matrix_with_bias_forward(pUnitOut, pSimpleRNNNeuralNetHeader->unit, pU,NULL, pTimeOut, pSimpleRNNNeuralNetHeader->unit,TRUE);
		}
		//------------------------------------------------------
		//誤差逆伝搬計算で活性化関数への入力を利用する場合:relu
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE ) {
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				pInput	= pTimeOut;
				pX		= pSimpleRNNLayer->pBPData + t * pSimpleRNNNeuralNetHeader->unit;
				size	= pSimpleRNNNeuralNetHeader->unit;
				while (size--) {
					*pX++ = *pInput++;
				}
				break;
			default:
				break;
			}
		}
		//------------------------------------------------------
		//活性化関数
		//------------------------------------------------------
		switch (pSimpleRNNNeuralNetHeader->activation) {
		case NEURAL_NET_ACTIVATION_RELU:
			relu_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit,0.0f);
			break;
		case NEURAL_NET_ACTIVATION_TANH:
			tanh_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		case NEURAL_NET_ACTIVATION_SIGMOID:
			sigmoid_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		case NEURAL_NET_ACTIVATION_SOFTMAX:
			softmax_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		default:
			return FALSE;
		}
		//------------------------------------------------------
		//誤差逆伝搬計算で活性化関数への出力値を利用する場合：hyperbolic_tangent,sigmoid
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE ) {
			//出力値を利用する場合
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_TANH:
			case NEURAL_NET_ACTIVATION_SIGMOID:
			case NEURAL_NET_ACTIVATION_SOFTMAX:
				pInput	= pTimeOut;
				pX		= pSimpleRNNLayer->pBPData + t * pSimpleRNNNeuralNetHeader->unit;
				size	= pSimpleRNNNeuralNetHeader->unit;
				while (size--) {
					*pX++ = *pInput++;
				}
				break;
			default:
				break;
			}
		}
		//------------------------------------------------------
		//誤差逆伝搬：Uへの入力　h(t)
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE) {
			pInput	= pTimeOut;
			pH = pSimpleRNNLayer->pH + t * pSimpleRNNNeuralNetHeader->unit;
			size = pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pH++ = *pInput++;
			}
		}
		//------------------------------------------------------
		//次のフレームに移動
		//------------------------------------------------------
		pTimeIn += nInputDim;
	}
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		//---------------------------------------------------------------------------------
		//全フレームのデータを出力バッファにコピー
		//---------------------------------------------------------------------------------
		pTimeOut = pPropagationInfo->pOutputBuffer;
		pUnitOut = pInternalOutputBuffer;
		i = pSimpleRNNNeuralNetHeader->unit * timeSteps;
		while (i--) {
			*pTimeOut++ = *pUnitOut++;
		}
		//---------------------------------------------------------------------------------
		//timeSteps分のデータを出力
		//---------------------------------------------------------------------------------
		DataShape_construct(&pPropagationInfo->dataShape, timeSteps, pSimpleRNNNeuralNetHeader->unit, 1);
	}
	else {
		//---------------------------------------------------------------------------------
		//最終（最新）フレームのデータを出力バッファの先頭にコピー
		//---------------------------------------------------------------------------------
		if (timeSteps > 0) {
			pTimeOut = pPropagationInfo->pOutputBuffer;
			pUnitOut = pInternalOutputBuffer + (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			i = pSimpleRNNNeuralNetHeader->unit;
			while (i--) {
				*pTimeOut++ = *pUnitOut++;
			}
		}
		//---------------------------------------------------------------------------------
		//出力データサイズ形状
		//---------------------------------------------------------------------------------
		DataShape_construct(&pPropagationInfo->dataShape, 1, pSimpleRNNNeuralNetHeader->unit, 1);
	}
	return TRUE;
}

//=====================================================================================
//  逆伝搬計算
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	flt32_t*	pW;
	flt32_t*	pU;
	uint32_t*	pLayerParam;
	flt32_t*	pDW;
	flt32_t*	pDB;
	flt32_t*	pDU;
	uint32_t	size;
	flt32_t*	pInternalOutputBuffer;
	flt32_t*	pX;
	flt32_t*	pOutput;
	flt32_t*	pBPDataT;
	flt32_t*	pH;
	flt32_t*	pInputWX;
	flt32_t*	pTimeLossBuffer;
	int32_t		t;
	int32_t		frameTime;
	flt32_t*	pInputTime;
	flt32_t*	pOutputTime;
	int32_t		lastTime;
	OptimizerFunctionTable optimizerInterface;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//誤差伝搬バッファー初期化
	//---------------------------------------------------------------------------------
	size = nInputDim * timeSteps;
	pX = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pX++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//微分値バッファポインタ
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pSimpleRNNLayer->hOptimizer, &optimizerInterface);
	pDW = optimizerInterface.pGetDeltaParameterBuffer(pSimpleRNNLayer->hOptimizer);
	pDU = pDW + nInputDim * timeSteps;
	pDB = pDU + pSimpleRNNNeuralNetHeader->unit * timeSteps;
	//---------------------------------------------------------------------------------
	//returnSequenceがFALSEの場合は、先頭フレームのデータ（微分値）をもとの最終フレームに戻しておく
	//---------------------------------------------------------------------------------
	pInternalOutputBuffer = (flt32_t*)pPropagationInfo->pTemporaryBuffer;
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		//出力バッファの誤差データを一時バッファにすべてコピーする
		pOutput = pInternalOutputBuffer;
		pX = pPropagationInfo->pOutputBuffer;
		size = pSimpleRNNNeuralNetHeader->unit * timeSteps;
		while (size--) {
			*pOutput++ = *pX++;
		}
	}
	else{
		if (timeSteps > 0) {
			//先頭フレームだけに誤差が伝搬しているので、この誤差を最終フレームにコピーする
			pOutput = pInternalOutputBuffer + (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			pX = pPropagationInfo->pOutputBuffer;
			size = pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pOutput++ = *pX++;
			}
			//その他の時間伝播は利用されていないのでは0にする
			pOutput = pInternalOutputBuffer;
			size = (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pOutput++ = 0.0f;
			}
		}
	}
	//---------------------------------------------------------------------------------
	//逆伝播
	//---------------------------------------------------------------------------------
	for (t = timeSteps - 1; t >= 0; t--) {
		//---------------------------------------------------------------------------------
		//  誤差伝播最大時間数
		//---------------------------------------------------------------------------------
		if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
			lastTime = t - pSimpleRNNLayer->maxPropagationTime;
			if (lastTime < 0) {
				lastTime = 0;
			}
		}
		else {
			lastTime = 0;
		}
		//---------------------------------------------------------------------------------
		//誤差伝搬バッファー
		//この時間のデータは今後利用しないため、時間伝搬用のバッファとして一時バッファとして再利用されることに注意
		//---------------------------------------------------------------------------------
		pOutputTime = pInternalOutputBuffer + t * pSimpleRNNNeuralNetHeader->unit;
		//---------------------------------------------------------------------------------
		// 指定された時間数だけ誤差伝播を実施する
		//---------------------------------------------------------------------------------
		for (frameTime = t; frameTime >= lastTime; frameTime--) {
			//---------------------------------------------------------------------------------
			//内部活性化関数に対する逆伝播
			//---------------------------------------------------------------------------------
			pBPDataT = pSimpleRNNLayer->pBPData + frameTime * pSimpleRNNNeuralNetHeader->unit;
			size = pSimpleRNNNeuralNetHeader->unit;
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				relu_backword(pBPDataT, pOutputTime, pOutputTime, size,0.0f);
				break;
			case NEURAL_NET_ACTIVATION_TANH:
				tanh_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			case NEURAL_NET_ACTIVATION_SIGMOID:
				sigmoid_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			case NEURAL_NET_ACTIVATION_SOFTMAX:
				softmax_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			default:
				break;
			}
			//---------------------------------------------------------------------------------
			//Wブロックに対する逆伝播：Dense層と同じ
			//誤差を保持しているバッファが、内部活性化関数からの出力である点と出力先のバッファが時間で異なる
			//---------------------------------------------------------------------------------
			pInputTime = pPropagationInfo->pInputBuffer + frameTime * nInputDim;
			pInputWX = pSimpleRNNLayer->pX + frameTime * nInputDim;
			weight_matrix_with_bias_backward(
				pInputTime,							//誤差出力（順伝搬時入力）
				nInputDim,							//誤差出力次元（順伝搬時入力次元）
				pW,									//W重み
				pOutputTime,						//内部活性化関数から出力される誤差を保持するバッファ領域
				pSimpleRNNNeuralNetHeader->unit,	//出力次元
				pInputWX,							//Wに入力されるデータのバッファ
				pDW,								//W微分値
				pDB									//B微分値
			);
			//---------------------------------------------------------------------------------
			//内部活性化関数からの誤差出力をUブロックに誤差逆伝播
			//---------------------------------------------------------------------------------
			if (frameTime > 0) {
				//---------------------------------------------------------------------------------
				//出力作業バッファを初期化
				//---------------------------------------------------------------------------------
				size = pSimpleRNNNeuralNetHeader->unit;
				pTimeLossBuffer = pSimpleRNNLayer->pTimeLossBuffer;
				while (size--) {
					*pTimeLossBuffer++ = 0.0f;
				}
				//---------------------------------------------------------------------------------
				//Uブロック逆伝播 Uブロック入力：h(t-1)　逆伝播
				//---------------------------------------------------------------------------------
				pH = pSimpleRNNLayer->pH + (frameTime - 1) * pSimpleRNNNeuralNetHeader->unit;
				weight_matrix_with_bias_backward(
					pSimpleRNNLayer->pTimeLossBuffer,	//U誤差出力（順伝搬時U入力）
					pSimpleRNNNeuralNetHeader->unit,		//誤差出力次元（順伝搬時U入力次元）
					pU,									//U重み
					pOutputTime,						//内部活性化関数から出力される誤差を保持するバッファ領域
					pSimpleRNNNeuralNetHeader->unit,		//U出力次元
					pH,									//Uに入力されるデータのバッファ（t-1）
					pDU,								//U部分値
					NULL								//バイアスは無し
				);
				//---------------------------------------------------------------------------------
				//Uブロックから出力される誤差で誤差伝播バッファを置き換える
				//pOutputTimeはt時刻のデータバッファであるが、この後利用されないため、
				//次の時刻（前の時刻）の逆伝搬用のデータバッファとして再利用されている
				//---------------------------------------------------------------------------------
				size = pSimpleRNNNeuralNetHeader->unit;
				pOutput = pOutputTime;
				pTimeLossBuffer = pSimpleRNNLayer->pTimeLossBuffer;
				while (size--) {
					*pOutput++ = *pTimeLossBuffer++;
				}
			}
		}
		//---------------------------------------------------------------------------------
		//順伝搬でこれ以上の出力がない
		//---------------------------------------------------------------------------------
		if (pSimpleRNNNeuralNetHeader->returnSequence == FALSE) {
			break;
		}
	}
	//---------------------------------------------------------------------------------
	//入力方向データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, timeSteps, nInputDim, 1);
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_update(handle_t hLayer) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	uint32_t*	pLayerParam;
	OptimizerFunctionTable	optimizerInterface;
	NeuralNetOptimizer_getInterface(pSimpleRNNLayer->hOptimizer, &optimizerInterface);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pParameter = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	//　W&U&B
	optimizerInterface.pUpdate(pSimpleRNNLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	flt32_t*	pW;
	flt32_t*	pU;
	flt32_t*	pB;
	uint32_t*	pLayerParam;
	uint32_t	paramSize;
	uint32_t	normSize;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//層パラメタ初期化
	//---------------------------------------------------------------------------------
	//W
	paramSize = pSimpleRNNNeuralNetHeader->unit * nInputDim;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pW, paramSize, normSize);
	//U
	paramSize = pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pU, paramSize, normSize);
	//B
	paramSize = pSimpleRNNNeuralNetHeader->unit;
	set_constant_initial_values(pB, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  最大誤差伝播時間の設定
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_setMaxPropagationTime(handle_t hLayer,uint32_t maxPropagationTime)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	layerType	= pNeuralNetHeader->layerType;
	uint32_t	timeSteps	= pNeuralNetHeader->inHeight;
	uint32_t	nInputDim	= pNeuralNetHeader->inWidth;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (layerType != NET_LAYER_SIMPLE_RNN) {
		return FALSE;
	}
	if (timeSteps == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//最大誤差伝播時間 0～timeSteps-1
	//---------------------------------------------------------------------------------
	if (maxPropagationTime < timeSteps) {	//最大誤差伝播時間t=t-1時刻まで
		pSimpleRNNLayer->maxPropagationTime = maxPropagationTime;
	}
	else {
		pSimpleRNNLayer->maxPropagationTime = timeSteps - 1;
	}
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_getLayerInformation(
	uint32_t*	pLayerData,						// in:image data
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	if (pSimpleRNNNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pSimpleRNNNeuralNetHeader->unit * nInputDim;	// W
		*pNumberOfLearningParameters += pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit;	// U
		*pNumberOfLearningParameters += pSimpleRNNNeuralNetHeader->unit;	// B
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(SimpleRNNNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInputDim * timeSteps, uint32_t);						//　Wへの入力
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);	//　Uへの入力
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);	//　内部活性化関数への入力または出力：内部活性化関数によって利用される値が異なる
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);				//	出力サイズのの逆伝播バッファー
		}
	}
	//---------------------------------------------------------------------------------
	//層内部の計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		//内部時間伝搬処理用
		*pTempWorkAreaSizeIn32BitWord = pSimpleRNNNeuralNetHeader->unit * timeSteps;
	}
	//---------------------------------------------------------------------------------
	//出力形状
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
			pOutputShape->height = timeSteps;
		}
		else {
			pOutputShape->height = 1;
		}
		pOutputShape->width = pSimpleRNNNeuralNetHeader->unit;
		pOutputShape->channel = 1;
	}
	//---------------------------------------------------------------------------------
	//入力形状
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerSimpleRNN_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerSimpleRNN_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	LayerFuncTable	funcTable;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerSimpleRNN_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerSimpleRNN_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			// 学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerInterface;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerInterface);
			parameterSize = optimizerInterface.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(SimpleRNNNeuralNetLayer), uint32_t);
			//data
			pSimpleRNNLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * nInputDim * timeSteps, uint32_t);								//　Wへの入力;
			pSimpleRNNLayer->pH = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);		//　Uへの入力
			pSimpleRNNLayer->pBPData = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);		//　内部活性化関数への入力または出力：内部活性化関数によって利用される値が異なる
			pSimpleRNNLayer->pTimeLossBuffer = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);					//	出力サイズのの逆伝播バッファー;
			pSimpleRNNLayer->hOptimizer = hOptimizer;
		}
		else {
			pSimpleRNNLayer->pX = NULL;
			pSimpleRNNLayer->hOptimizer = NULL;
		}
		return (handle_t)pSimpleRNNLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerSimpleRNN_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerSimpleRNN_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerSimpleRNN_construct;
	pInterface->pGetShape = NeuralNetLayerSimpleRNN_getShape;
	pInterface->pForward = NeuralNetLayerSimpleRNN_forward;
	pInterface->pBackward = NeuralNetLayerSimpleRNN_backward;
	pInterface->pUpdate = NeuralNetLayerSimpleRNN_update;
	pInterface->pInitializeParameters = NeuralNetLayerSimpleRNN_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerSimpleRNN_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_constructLayerData(
	uint32_t*	pBuffer, 
	uint32_t	sizeOfBufferIn32BitWord, 
	uint32_t*	pInputHeight,	// time steps
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	unit,
	NeuralNetActivationType activation,
	bool_t		returnSequence,
	uint32_t*	pSizeOfLayerIn32BitWord) 
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamW;
	uint32_t	sizeOfParamU;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	timeSteps;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	if (*pInputChannel != 1) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
	timeSteps	= inHeight;
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	sizeOfParamW = size_in_type(sizeof(flt32_t) * unit * inWidth, uint32_t);
	sizeOfParamU = size_in_type(sizeof(flt32_t) * unit * unit, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamW + sizeOfParamU + sizeOfParamB;
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
		pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pSimpleRNNNeuralNetHeader->super, NET_LAYER_SIMPLE_RNN, inHeight, inWidth, inChannel, sizeLayer);
		pSimpleRNNNeuralNetHeader->returnSequence = returnSequence;
		pSimpleRNNNeuralNetHeader->activation = activation;
		pSimpleRNNNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//W
		pLayer += sizeOfParamW;
		//U
		pLayer += sizeOfParamU;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	if (returnSequence == TRUE) {
		*pInputHeight = timeSteps;
	}
	else {
		*pInputHeight = 1;
	}
	*pInputWidth	= unit;
	*pInputChannel	= 1;
	return TRUE;
}
