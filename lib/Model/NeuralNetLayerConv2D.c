#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerConv2D.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  Conv2D layer header
//=====================================================================================
typedef struct tagConv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		nFilter;		//フィル多数
	uint32_t		kernelHeight;	//フィルター高（kernel size in height direction）
	uint32_t		kernelWidth;	//フィルター幅（kernel size in width direction）
	uint32_t		kernelChannel;	//フィルター深（kernel size in channel direction）
	uint32_t		strideHeight;	//ストライド高
	uint32_t		strideWidth;	//ストライド幅
	bool_t			fPadding;		//パディングフラグ
} Conv2DNeuralNetHeader;

//=====================================================================================
//  Conv2D層構造体
//=====================================================================================
typedef struct tagConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ
	handle_t		hOptimizer;		//オプティマイザーハンドル
} Conv2DNeuralNetLayer;

//=====================================================================================
//  形状関連情報計算
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_getShapeInformation(	bool_t		fPadding,
											uint32_t	inHeight,
											uint32_t	inWidth,
											uint32_t	nFilter,
											uint32_t	kernelHeight,
											uint32_t	kernelWidth,
											uint32_t	strideHeight,
											uint32_t	strideWidth,
											int32_t*	pPaddingHeight,
											int32_t*	pPaddingWidth,
											DataShape*	pOutputShape)
{
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	uint32_t	outHeight;
	uint32_t	outWidth;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Paddingサイズ
	//---------------------------------------------------------------------------------
	if (fPadding == TRUE) {
		//strideでダウンサンプルする
		outHeight = (inHeight + strideHeight - 1) / strideHeight;
		outWidth = (inWidth + strideWidth - 1) / strideWidth;
		//paddingサイズ
		paddingHeight = (outHeight - 1) * strideHeight + kernelHeight - inHeight;
		paddingWidth = (outWidth - 1) * strideWidth + kernelWidth - inWidth;
	}
	else {
		paddingHeight = 0;
		paddingWidth = 0;
	}
	if (pPaddingHeight != NULL) {
		*pPaddingHeight = paddingHeight;
	}
	if (pPaddingWidth != NULL) {
		*pPaddingWidth = paddingWidth;
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		pOutputShape->height = 1 + ((inHeight + paddingHeight) - kernelHeight) / strideHeight;
		pOutputShape->width = 1 + ((inWidth + paddingWidth) - kernelWidth) / strideWidth;
		pOutputShape->channel = nFilter;
	}
	return TRUE;
}

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	NeuralNetLayerConv2D_getShapeInformation(
		pConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pConv2DNeuralNetHeader->nFilter,
		pConv2DNeuralNetHeader->kernelHeight,
		pConv2DNeuralNetHeader->kernelWidth,
		pConv2DNeuralNetHeader->strideHeight,
		pConv2DNeuralNetHeader->strideWidth,
		NULL,
		NULL,
		pOutputShape);
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  層パラメタ
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_getLayerParameter(Conv2DNeuralNetHeader* pConv2DNeuralNetHeader, flt32_t** ppFilter, flt32_t** ppBias)
{
	uint32_t*	pLayerParam;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(Conv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel, uint32_t);
	pBias = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->nFilter, uint32_t); //biasはフィルター数分ある
	//---------------------------------------------------------------------------------
	//パラメタ配列ポインタ
	//---------------------------------------------------------------------------------
	if (ppFilter != NULL) {
		*ppFilter = pFilter;
	}
	if (ppBias != NULL) {
		*ppBias = pBias;
	}
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
#define OUT_OF_REGION_INDICATION_VALUE	(0xFFFFFFFF)

static
bool_t
NeuralNetLayerConv2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pConv2DNeuralNetHeader;
	uint32_t	i,j,k;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	uint32_t	kernelHeight;
	uint32_t	kernelWidth;
	//一時変数
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//データ位置
	uint32_t	iH;
	uint32_t	iW;
	int32_t		iCornerInHeight;
	int32_t		iCornerInWidth;
	uint32_t	nFilterSize;
	int32_t		iPosInHeight;
	int32_t		iPosInWidth;
	uint32_t	dataIndex;
	uint32_t	nKernelWidthChanSize;
	uint32_t	nImageWidthChannel;
	flt32_t*	pExtractedData;
	flt32_t*	pOutput;
	flt32_t*	pFilter;
	flt32_t*	pFilterHead;
	flt32_t*	pBias;
	flt32_t*	pBiasHead;
	flt32_t		filterdData;
	DataShape	outputShape;
	flt32_t*	pInput;
	flt32_t*	pX;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getShapeInformation(
		pConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pConv2DNeuralNetHeader->nFilter,
		pConv2DNeuralNetHeader->kernelHeight,
		pConv2DNeuralNetHeader->kernelWidth,
		pConv2DNeuralNetHeader->strideHeight,
		pConv2DNeuralNetHeader->strideWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	if (fStatus == FALSE) {
		return FALSE;
	}
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//一時変数利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	nFilter			= pConv2DNeuralNetHeader->nFilter;
	kernelHeight	= pConv2DNeuralNetHeader->kernelHeight;
	kernelWidth		= pConv2DNeuralNetHeader->kernelWidth;
	strideHeight	= pConv2DNeuralNetHeader->strideHeight;
	strideWidth		= pConv2DNeuralNetHeader->strideWidth;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer= pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
	//---------------------------------------------------------------------------------
	//フィルターサイズ分のデータをストライドしながら1次元のデータに変換する
	//---------------------------------------------------------------------------------
	nKernelWidthChanSize	= kernelWidth * inChannel;
	nImageWidthChannel		= inWidth * inChannel;
	nFilterSize				= kernelHeight * kernelWidth * inChannel;
	pOutput = pOutputBuffer;
	iCornerInHeight = -paddingHeight / 2;	//paddingを考慮してシフトする;
	iH = outHeight;
	while(iH--) {
		iCornerInWidth = -paddingWidth / 2;	//paddingを考慮してシフトする
		iW = outWidth;
		while(iW--) {
			pExtractedData = (flt32_t*)pTemporaryBuffer;
			//-----------------------------------------------------------------
			//fiterHeight×kernelWidth×Channelのデータを収集
			//-----------------------------------------------------------------
			iPosInHeight = iCornerInHeight;
			k = kernelHeight;
			while (k--) {
				iPosInWidth = iCornerInWidth;
				if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
					//範囲外データは0とする
					i = kernelWidth * inChannel;
					while (i--) {
						*pExtractedData++ = 0.0f;
					}
				}
				else {
					dataIndex = (iPosInHeight * inWidth + iPosInWidth) * inChannel;
					j = kernelWidth;
					while (j--) {
						if (iPosInWidth < 0 || iPosInWidth >= (int32_t)inWidth) {
							//範囲外データは0とする
							i = inChannel;
							while (i--) {
								*pExtractedData++ = 0.0f;
							}
							dataIndex += inChannel;
						}
						else {
							i = inChannel;
							while (i--) {
								*pExtractedData++ = pInputBuffer[dataIndex++];
							}
						}
						iPosInWidth++;
					}
				}
				iPosInHeight++;
			}
			//-----------------------------------------------------------------
			//nFilter個のフィルターをかける
			//-----------------------------------------------------------------
			pFilterHead = pFilter;
			pBiasHead	= pBias;
			i			= nFilter;
			while (i--) {
				filterdData = *pBiasHead++; //biasはフィルター数分だけ存在する
				pExtractedData = (flt32_t*)pTemporaryBuffer;
				//フィルター計算：フィルターデータはkernelHeight*kernelWidth*inChannelで並んでいる
				j = nFilterSize;
				while (j--) {
					filterdData += *pExtractedData++ * *pFilterHead++;
				}
				*pOutput++ = filterdData;
			}
			iCornerInWidth += strideWidth;	//ストライド分移動  
		}
		iCornerInHeight += strideHeight;	//ストライド分移動 
	}
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pConv2DLayer->pX == NULL) {
			return FALSE;
		}
		//入力データコピー
		pInput = pInputBuffer;
		pX = pConv2DLayer->pX;
		i = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		while (i--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//出力データサイズ形状
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape = outputShape;
	return TRUE;
}

//=====================================================================================
//  逆伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pConv2DNeuralNetHeader;
	uint32_t	i,j,k;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	uint32_t	outHeight;
	uint32_t	outWidth;
	//一時変数
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	uint32_t	kernelHeight;
	uint32_t	kernelWidth;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//データ位置
	uint32_t	iH;
	uint32_t	iW;
	int32_t		iCornerInHeight;
	int32_t		iCornerInWidth;
	uint32_t	nFilterSize;
	int32_t		iPosInHeight;
	int32_t		iPosInWidth;
	uint32_t	dataIndex;
	uint32_t	nKernelWidthChanSize;
	uint32_t	nImageWidthChannel;
	uint32_t*	pExtractedDataIndex;
	flt32_t*	pInputData;
	flt32_t*	pXDataArray;
	flt32_t*	pXData;	
	flt32_t*	pDLossArray;
	flt32_t*	pFilter;
	flt32_t*	pFilterHead;
	flt32_t*	pBias;
	//パラメタ微分ポインタ
	flt32_t*	pDFilter;
	flt32_t*	pDBias;
	uint32_t	size;
	flt32_t*	pDFilterHead;
	flt32_t*	pDBiasHead;
	flt32_t*	pInputHead;
	flt32_t		deltaLoss;
	//層出力マトリクス
	DataShape	outputShape;
	bool_t		fStatus;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//逆伝搬対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getShapeInformation(
		pConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pConv2DNeuralNetHeader->nFilter,
		pConv2DNeuralNetHeader->kernelHeight,
		pConv2DNeuralNetHeader->kernelWidth,
		pConv2DNeuralNetHeader->strideHeight,
		pConv2DNeuralNetHeader->strideWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pConv2DNeuralNetHeader->nFilter;
	kernelHeight = pConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pConv2DNeuralNetHeader->kernelWidth;
	strideHeight = pConv2DNeuralNetHeader->strideHeight;
	strideWidth = pConv2DNeuralNetHeader->strideWidth;
	pInputBuffer = pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer = pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pConv2DLayer->hOptimizer);
	pDBias = pDFilter + pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel;
	//---------------------------------------------------------------------------------
	//誤差出力バッファー
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInputHead = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInputHead++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//kernelのピクセルをストライドしながら1次元のデータに変換する
	//inHeight*inWidthからfiterHeight×kernelWidth分のデータをstridして取得することをChannel分実施する。
	//---------------------------------------------------------------------------------
	nKernelWidthChanSize	= kernelWidth * inChannel;
	nImageWidthChannel		= inWidth * inChannel;
	nFilterSize				= kernelHeight * kernelWidth * inChannel;
	//入力マトリクスから切り出された3次元データに対してフィルターを適用する
	pDLossArray = pPropagationInfo->pOutputBuffer;
	deltaLoss = (*pDLossArray);
	pXDataArray = pConv2DLayer->pX;
	iCornerInHeight = -paddingHeight / 2;	//paddingを考慮してシフトする;
	iH = outHeight;
	while (iH--) {
		iCornerInWidth = -paddingWidth / 2;	//paddingを考慮してシフトする
		iW = outWidth;
		while (iW--) {
			pExtractedDataIndex = (uint32_t*)pPropagationInfo->pTemporaryBuffer;	//データインデックスバッファー　H×W×Chan(深さ)
			//-----------------------------------------------------------------
			//入力イメージ中のフィルターfiterHeight×kernelWidth×Channelのピクセルを収集
			//-----------------------------------------------------------------
			iPosInHeight = iCornerInHeight;
			k = kernelHeight;
			while (k--) {
				if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight ) {
					//範囲外を示すデータインデックスをセットする
					i = kernelWidth * inChannel;
					while (i--) {
						*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;	//利用されたデータのインデックス
					}
				}
				else {
					iPosInWidth = iCornerInWidth;
					dataIndex = (iPosInHeight * inWidth + iPosInWidth) * inChannel;
					j = kernelWidth;
					while (j--) {
						if (iPosInWidth < 0 || iPosInWidth >= (int32_t)inWidth) {
							//範囲外を示すデータインデックスをセットする
							i = inChannel;
							while (i--) {
								*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;	//利用されたデータのインデックス
							}
							dataIndex += inChannel;
						}
						else {
							i = inChannel;
							while (i--) {
								*pExtractedDataIndex++ = dataIndex++;	//利用されたデータのインデックス
							}
						}
						iPosInWidth++;
					}
				}
				iPosInHeight++;
			}
			//-----------------------------------------------------------------
			//収集されたフィルターfiterHeight×kernelWidth×Channelに対しnFilter個のフィルターをかける
			//-----------------------------------------------------------------
			pFilterHead = pFilter;
			pDFilterHead = pDFilter;
			pDBiasHead = pDBias;
			i = nFilter;
			while (i--) {
				//----------------------------------------------------------
				//逆伝搬入力の伝搬誤差微分値
				//----------------------------------------------------------
				deltaLoss = (*pDLossArray++);
				//----------------------------------------------------------
				//bias微分値積算
				//----------------------------------------------------------
				*pDBiasHead++ += deltaLoss;
				//----------------------------------------------------------
				//対象入力データ位置インデックス
				//----------------------------------------------------------
				pExtractedDataIndex = (uint32_t*)pPropagationInfo->pTemporaryBuffer;
				//----------------------------------------------------------
				//演算量削減記述：フィルター計算にしたがってデータは並べてある kernelHeight*kernelWidth*inChannel
				//----------------------------------------------------------
				j = nFilterSize;
				while (j--) {
					if (*pExtractedDataIndex != OUT_OF_REGION_INDICATION_VALUE) {
						pInputData = pPropagationInfo->pInputBuffer + *pExtractedDataIndex;	//微分値を順伝搬の入力方向へ渡すバッファ位置
						pXData = pXDataArray + *pExtractedDataIndex;		//順伝搬時に保持した入力値X
						//----------------------------------------------------------
						//フィルター係数微分値へ積算
						//----------------------------------------------------------
						*pDFilterHead += (*pXData) * deltaLoss;
						//----------------------------------------------------------
						//逆伝搬出力（inputBuffer）へ積算
						//----------------------------------------------------------
						*pInputData += (*pFilterHead) * deltaLoss;
					}
					//----------------------------------------------------------
					//ポインタ更新
					//----------------------------------------------------------
					pExtractedDataIndex++;
					pFilterHead++;
					pDFilterHead++;
				}
			}
			iCornerInWidth += strideWidth;	//ストライド分移動  
		}
		iCornerInHeight += strideHeight;	//ストライド分移動 
	}
	//---------------------------------------------------------------------------------
	//逆伝搬出力データサイズ形状(順伝搬の入力データ形状)
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, inChannel);
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_update(handle_t hLayer) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	bool_t		fStatus;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pConv2DLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pParameter, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	uint32_t	paramSize;
	uint32_t	normSize;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	//Filter
	paramSize = pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pFilter, paramSize, normSize);
	//Bias
	paramSize = pConv2DNeuralNetHeader->nFilter;
	set_random_initial_values(hRandomValueGenerator,pBias, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape* pInputShape,
	DataShape* pOutputShape) {
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pConv2DNeuralNetHeader;
	if (pConv2DNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel;	//filter
		*pNumberOfLearningParameters += pConv2DNeuralNetHeader->nFilter;	//Beta
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(Conv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//層内部の計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->kernelChannel * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//出力形状
	//---------------------------------------------------------------------------------
	NeuralNetLayerConv2D_getShapeInformation(
		pConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pConv2DNeuralNetHeader->nFilter,
		pConv2DNeuralNetHeader->kernelHeight,
		pConv2DNeuralNetHeader->kernelWidth,
		pConv2DNeuralNetHeader->strideHeight,
		pConv2DNeuralNetHeader->strideWidth,
		NULL,
		NULL,
		pOutputShape);
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
NeuralNetLayerConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(Conv2DNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerConv2D_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerConv2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t parameterSize;
	NeuralNetLayerConv2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerConv2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			//学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer,&optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(Conv2DNeuralNetLayer), uint32_t);
			pConv2DLayer->pX = (flt32_t*)pObjectWork;
			pConv2DLayer->hOptimizer = hOptimizer;
		}
		else {
			pConv2DLayer->pX = NULL;
			pConv2DLayer->hOptimizer = NULL;
		}
		return (handle_t)pConv2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerConv2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerConv2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerConv2D_construct;
	pInterface->pGetShape = NeuralNetLayerConv2D_getShape;
	pInterface->pForward = NeuralNetLayerConv2D_forward;
	pInterface->pBackward = NeuralNetLayerConv2D_backward;
	pInterface->pUpdate = NeuralNetLayerConv2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerConv2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerConv2D_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerConv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	nFilter,
	uint32_t	kernelHeight,
	uint32_t	kernelWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	bool_t		fPadding,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamFilter;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	kernelChannel;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	DataShape	outputShape;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	inHeight = *pInputHeight;
	inWidth = *pInputWidth;
	inChannel = *pInputChannel;
	kernelChannel = *pInputChannel;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (inHeight < kernelHeight) {
		return FALSE;
	}
	if (inWidth < kernelWidth) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(Conv2DNeuralNetHeader), uint32_t);
	sizeOfParamFilter = size_in_type(sizeof(flt32_t) * nFilter * kernelHeight * kernelWidth * kernelChannel, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * nFilter, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamFilter + sizeOfParamB;
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
		pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pConv2DNeuralNetHeader->super, NET_LAYER_CONV2D, inHeight, inWidth, inChannel, sizeLayer);
		pConv2DNeuralNetHeader->nFilter = nFilter;
		pConv2DNeuralNetHeader->kernelHeight = kernelHeight;
		pConv2DNeuralNetHeader->kernelWidth = kernelWidth;
		pConv2DNeuralNetHeader->kernelChannel = kernelChannel;
		pConv2DNeuralNetHeader->strideHeight = strideHeight;
		pConv2DNeuralNetHeader->strideWidth = strideWidth;
		//Padding
		pConv2DNeuralNetHeader->fPadding = fPadding;
		//header
		pLayer += sizeHeader;
		//Filter
		pLayer += sizeOfParamFilter;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//Paddingサイズ
	//---------------------------------------------------------------------------------
	NeuralNetLayerConv2D_getShapeInformation(fPadding, inHeight, inWidth, nFilter,kernelHeight, kernelWidth, strideHeight, strideWidth, &paddingHeight, &paddingWidth,&outputShape);
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = outputShape.height;
	*pInputWidth = outputShape.width;
	*pInputChannel = outputShape.channel;
	return TRUE;
}
