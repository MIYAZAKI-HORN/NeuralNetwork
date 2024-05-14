#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDepthwiseConv2D.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  DepthwiseConv2D層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagDepthwiseConv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		nFilter;		//フィル多数
	uint32_t		kernelHeight;	//フィルター高（kernel size in height direction）
	uint32_t		kernelWidth;	//フィルター幅（kernel size in width direction）
	uint32_t		strideHeight;	//ストライド高
	uint32_t		strideWidth;	//ストライド幅
	bool_t			fPadding;		//パディングフラグ
} DepthwiseConv2DNeuralNetHeader;

//=====================================================================================
//  DepthwiseConv2D層構造体
//=====================================================================================
typedef struct tagDepthwiseConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝搬用データバッファ
	handle_t		hOptimizer;		//オプティマイザーハンドル
} DepthwiseConv2DNeuralNetLayer;

//=====================================================================================
//  形状関連情報計算
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_getShapeInformation(
	bool_t		fPadding,
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
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
		//kerasに合わせた出力サイズ：strideでダウンサンプルされる
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
		pOutputShape->channel = inChannel * nFilter;
	}
	return TRUE;
}

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	NeuralNetLayerDepthwiseConv2D_getShapeInformation(
		pDepthwiseConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pDepthwiseConv2DNeuralNetHeader->nFilter,
		pDepthwiseConv2DNeuralNetHeader->kernelHeight,
		pDepthwiseConv2DNeuralNetHeader->kernelWidth,
		pDepthwiseConv2DNeuralNetHeader->strideHeight,
		pDepthwiseConv2DNeuralNetHeader->strideWidth,
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
NeuralNetLayerDepthwiseConv2D_getLayerParameter(
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader, 
	flt32_t** ppFilter)
{
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pFilter;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pDepthwiseConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(DepthwiseConv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pNeuralNetHeader->inChannel * pDepthwiseConv2DNeuralNetHeader->nFilter * pDepthwiseConv2DNeuralNetHeader->kernelHeight * pDepthwiseConv2DNeuralNetHeader->kernelWidth, uint32_t);
	//---------------------------------------------------------------------------------
	//パラメタ配列ポインタ
	//---------------------------------------------------------------------------------
	if (ppFilter != NULL) {
		*ppFilter = pFilter;
	}
	return TRUE;
}

//=====================================================================================
//  順伝搬
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	uint32_t	i,j;
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
	uint32_t	kernelDim;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//データ位置
	uint32_t	iChan;
	uint32_t	iH;
	uint32_t	iW;
	int32_t		iCornerInHeight;
	int32_t		iCornerInWidth;
	int32_t		iPosInHeight;
	int32_t		iPosInWidth;
	int32_t		dataIndex;
	flt32_t*	pFilter;
	flt32_t*	pOutFiltered;
	flt32_t*	pExtractedData;
	flt32_t*	pOutputData;
	flt32_t*	pFilterHead;
	uint32_t	dataSize;
	flt32_t		filterdData;
	flt32_t*	pInput;
	flt32_t*	pX;
	DataShape	outputShape;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//サイズ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getShapeInformation(
		pDepthwiseConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pDepthwiseConv2DNeuralNetHeader->nFilter,
		pDepthwiseConv2DNeuralNetHeader->kernelHeight,
		pDepthwiseConv2DNeuralNetHeader->kernelWidth,
		pDepthwiseConv2DNeuralNetHeader->strideHeight,
		pDepthwiseConv2DNeuralNetHeader->strideWidth,
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
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//高速化と見やすさのため一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	nFilter			= pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight	= pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth		= pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	strideHeight	= pDepthwiseConv2DNeuralNetHeader->strideHeight;
	strideWidth		= pDepthwiseConv2DNeuralNetHeader->strideWidth;
	kernelDim		= kernelHeight * kernelWidth;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
	//---------------------------------------------------------------------------------
	//出力データの並びは kernelHeight * kernelWidth * (nFilter×iChan)
	//---------------------------------------------------------------------------------
	for (iChan = 0; iChan < inChannel; iChan++) {
		pOutputData = pOutputBuffer + iChan * nFilter; //1 pixelあたりdwされてできる深さは入力チャンネル数×depth_multifierで与えられるフィルター数分だけある。
		//チャンネルの先頭
		iCornerInHeight = -paddingHeight / 2;	//paddingを考慮してシフトする
		iH = outHeight;
		while(iH--) {
			iCornerInWidth = -paddingWidth / 2;	//paddingを考慮してシフトする
			iW = outWidth;
			while(iW--) {
				pExtractedData = (flt32_t*)pTemporaryBuffer;	//データバッファー
				//-----------------------------------------------------------------
				//カーネルサイズデータを収集
				//-----------------------------------------------------------------
				iPosInHeight = iCornerInHeight;
				j = kernelHeight;
				while(j--){
					if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
						//範囲外データ
						i = kernelWidth;
						while (i--) {
							*pExtractedData++ = 0.0f;
						}
					}
					else {
						iPosInWidth = iCornerInWidth;
						dataIndex = (iPosInHeight * inWidth + iPosInWidth) * inChannel + iChan;
						i = kernelWidth;
						while(i--){
							if (iPosInWidth < 0 || iPosInWidth >= (int32_t)inWidth) {
								//範囲外データ
								*pExtractedData++ = 0.0f;
							}
							else {
								*pExtractedData++ = pInputBuffer[dataIndex];
							}
							iPosInWidth++;
							dataIndex += inChannel;
						}
					}
					iPosInHeight++;
				}
				//-----------------------------------------------------------------
				//上記切り出したフィルターサイズデータに対しnFilter個のフィルターをかける
				//Filter係数並び：（nChan,nFilter,Height,Width）
				//-----------------------------------------------------------------
				pFilterHead = pFilter + iChan * nFilter * kernelDim;
				i = nFilter;
				pOutFiltered = pOutputData;
				while (i--) {
					//-----------------------------------------------------------------
					//上記切り出したフィルターサイズのデータに対し、Σ(F(i,j)×D(i,j,k))する
					//-----------------------------------------------------------------
					filterdData = 0;
					pExtractedData = (flt32_t*)pTemporaryBuffer;
					j = kernelDim;
					while (j--) {
						filterdData += *pExtractedData++ * *pFilterHead++;
					}
					*pOutFiltered++ = filterdData;
				}
				pOutputData += inChannel * nFilter; //1 pixelあたりdwされてできる深さは入力チャンネル数×depth_multifierで与えられるフィルター数分だけある。
				//
				iCornerInWidth += strideWidth;	//2次元平面上のX（ストライド考慮） 
			}
			iCornerInHeight += strideHeight;
		}
	}
	//---------------------------------------------------------------------------------
	//back propagation用入力データ保持:X
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//エラーハンドリング
		if (pDepthwiseConv2DLayer->pX == NULL) {
			return FALSE;
		}
		dataSize = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		pInput = pInputBuffer;
		pX = pDepthwiseConv2DLayer->pX;
		while (dataSize--) {
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
#define OUT_OF_REGION_INDICATION_VALUE	(0xFFFFFFFF)
static
bool_t
NeuralNetLayerDepthwiseConv2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	uint32_t	i,j;
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
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
	uint32_t	kernelDim;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//データ位置
	uint32_t	iChan;
	int32_t		iH;
	int32_t		iW;
	int32_t		iCornerInHeight;
	int32_t		iCornerInWidth;
	int32_t		iPosInHeight;
	int32_t		iPosInWidth;
	int32_t		dataIndex;
	flt32_t*	pFilter;
	flt32_t*	pOutputData;
	flt32_t*	pFilterHead;
	uint32_t	dataSize;
	flt32_t*	pInputData;
	DataShape	outputShape;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	bool_t		fStatus;
	//伝搬誤差微分値
	flt32_t*	pDLossArray;
	flt32_t		deltaLoss;
	//畳み込み
	uint32_t*	pExtractedDataIndex;
	flt32_t*	pDDWFilter;
	flt32_t*	pXDataArray;
	flt32_t*	pXData;
	//パラメタ微分値
	flt32_t*	pDFilterHead;
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
	fStatus = NeuralNetLayerDepthwiseConv2D_getShapeInformation(
		pDepthwiseConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pDepthwiseConv2DNeuralNetHeader->nFilter,
		pDepthwiseConv2DNeuralNetHeader->kernelHeight,
		pDepthwiseConv2DNeuralNetHeader->kernelWidth,
		pDepthwiseConv2DNeuralNetHeader->strideHeight,
		pDepthwiseConv2DNeuralNetHeader->strideWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight = pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	strideHeight = pDepthwiseConv2DNeuralNetHeader->strideHeight;
	strideWidth = pDepthwiseConv2DNeuralNetHeader->strideWidth;
	kernelDim = kernelHeight * kernelWidth;
	pInputBuffer = pPropagationInfo->pInputBuffer;		//入力バッファ
	pOutputBuffer = pPropagationInfo->pOutputBuffer;		//出力バッファ
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//一時計算バッファ
	//---------------------------------------------------------------------------------
	//微分値を保持するバッファポインタ取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pDepthwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDDWFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pDepthwiseConv2DLayer->hOptimizer);
	//---------------------------------------------------------------------------------
	//back propagation用ネットワーク微分パラメタバッファ
	//---------------------------------------------------------------------------------
	pDFilterHead	= pDDWFilter;
	//---------------------------------------------------------------------------------
	//逆伝搬誤差バッファーを初期化する
	//---------------------------------------------------------------------------------
	dataSize = inHeight * inWidth * inChannel;
	pInputData = pInputBuffer;
	while (dataSize--) {
		*pInputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//出力データ(誤差保持)の並びは kernelHeight * kernelWidth * (nFilter×iChan)
	//---------------------------------------------------------------------------------
	pXDataArray = pDepthwiseConv2DLayer->pX;
	pDLossArray = pOutputBuffer;
	for (iChan = 0; iChan < inChannel; iChan++) {
		pOutputData = pOutputBuffer + iChan * nFilter; //1 pixelあたりdwされてできる深さは入力チャンネル数×depth_multifierで与えられるフィルター数分だけある。
		iCornerInHeight = -paddingHeight / 2;	//paddingを考慮してシフトする
		iH = outHeight;
		while(iH--) {
			iCornerInWidth = -paddingWidth / 2;	//paddingを考慮してシフトする
			iW = outWidth;
			while(iW--) {
				pExtractedDataIndex = pTemporaryBuffer;	//データバッファー
				//-----------------------------------------------------------------
				//カーネルサイズデータを収集
				//-----------------------------------------------------------------
				iPosInHeight = iCornerInHeight;
				j = kernelHeight;
				while (j--) {
						if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
						//範囲外データ
						i = kernelWidth;
						while (i--) {
							*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;
						}
					}
					else {
						iPosInWidth = iCornerInWidth;
						i = kernelWidth;
						dataIndex = (iPosInHeight * inWidth + iPosInWidth) * inChannel + iChan;
						while (i--) {
							if (iPosInWidth < 0 || iPosInWidth >= (int32_t)inWidth) {
								//範囲外データ
								*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;
							}
							else {
								*pExtractedDataIndex++ = dataIndex;
							}
							iPosInWidth++;
							dataIndex += inChannel;
						}
					}
					iPosInHeight++;
				}
				//-----------------------------------------------------------------
				//上記切り出したフィルターサイズデータに対しnFilter個のフィルターをかける
				//Filter係数並び：（nChan,nFilter,Height,Width）
				//-----------------------------------------------------------------
				pFilterHead = pFilter + iChan * nFilter * kernelDim;
				pDFilterHead = pDDWFilter + iChan * nFilter * kernelDim;
				i = nFilter;
				pDLossArray = pOutputData;
				while (i--) {
					//----------------------------------------------------------
					//伝搬誤差微分値
					//----------------------------------------------------------
					deltaLoss = *pDLossArray++;
					//-----------------------------------------------------------------
					//上記切り出したフィルターサイズのデータに対し、Σ(F(i,j)×D(i,j,k))する
					//-----------------------------------------------------------------
					//対象入力データXの位置インデックス
					pExtractedDataIndex = (uint32_t*)pTemporaryBuffer;
					j = kernelDim;
					while (j--) {
						//----------------------------------------------------------
						//データ位置
						//----------------------------------------------------------
						if (*pExtractedDataIndex != OUT_OF_REGION_INDICATION_VALUE) {
							pInputData = pInputBuffer + *pExtractedDataIndex;	//逆伝搬出力：順伝搬の入力方向へ渡す微分値
							pXData = pXDataArray + *pExtractedDataIndex;		//順伝搬時の入力値X
							//----------------------------------------------------------
							//フィルター係数(学習パラメタ)微分値へ積算
							//----------------------------------------------------------
							*pDFilterHead += (*pXData) * deltaLoss;
							//----------------------------------------------------------
							//逆伝搬出力へ積算
							//----------------------------------------------------------
							*pInputData += (*pFilterHead) * deltaLoss;
						}
						//----------------------------------------------------------
						//ポインタ更新
						//----------------------------------------------------------
						pExtractedDataIndex++;
						pDFilterHead++;
						pFilterHead++;
					}
				}
				pOutputData += inChannel * nFilter;
				iCornerInWidth += strideWidth;
			}
			iCornerInHeight += strideHeight;
		}
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
NeuralNetLayerDepthwiseConv2D_update(handle_t hLayer) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t* pParameter;
	bool_t		fStatus;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pDepthwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pParameter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pDepthwiseConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	uint32_t	kernelHeight;
	uint32_t	kernelWidth;
	flt32_t*	pFilter;
	uint32_t	paramSize;
	uint32_t	normSize;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight = pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	paramSize = nFilter * kernelHeight * kernelWidth * inChannel;
	normSize = paramSize;;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pFilter, paramSize, normSize);
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	uint32_t	kernelHeight;
	uint32_t	kernelWidth;
	//---------------------------------------------------------------------------------
	//パラメタは一時変数で利用
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight = pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = inChannel * nFilter * kernelHeight * kernelWidth;	//dw filter
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(DepthwiseConv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//層内部の計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = size_in_type(sizeof(flt32_t) * pDepthwiseConv2DNeuralNetHeader->kernelHeight * pDepthwiseConv2DNeuralNetHeader->kernelWidth, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//出力形状
	//---------------------------------------------------------------------------------
	NeuralNetLayerDepthwiseConv2D_getShapeInformation(
		pDepthwiseConv2DNeuralNetHeader->fPadding,
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pDepthwiseConv2DNeuralNetHeader->nFilter,
		pDepthwiseConv2DNeuralNetHeader->kernelHeight,
		pDepthwiseConv2DNeuralNetHeader->kernelWidth,
		pDepthwiseConv2DNeuralNetHeader->strideHeight,
		pDepthwiseConv2DNeuralNetHeader->strideWidth,
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
NeuralNetLayerDepthwiseConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DepthwiseConv2DNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerDepthwiseConv2D_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerDepthwiseConv2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	LayerFuncTable	funcTable;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerDepthwiseConv2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//層インターフェイス取得
		NeuralNetLayerDepthwiseConv2D_getInterface(&funcTable);
		//層構造体構築
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//学習用データおよび最適化アルゴリズムオブジェクトハンドル
		if (fEnableLearning == TRUE) {
			//学習パラメタサイズチェック
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(DepthwiseConv2DNeuralNetLayer), uint32_t);
			pDepthwiseConv2DLayer->pX = (flt32_t*)pObjectWork;
			pDepthwiseConv2DLayer->hOptimizer = hOptimizer;
		}
		else {
			pDepthwiseConv2DLayer->pX = NULL;
			pDepthwiseConv2DLayer->hOptimizer = NULL;
		}
		return (handle_t)pDepthwiseConv2DLayer;
	}
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerDepthwiseConv2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerDepthwiseConv2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerDepthwiseConv2D_construct;
	pInterface->pGetShape = NeuralNetLayerDepthwiseConv2D_getShape;
	pInterface->pForward = NeuralNetLayerDepthwiseConv2D_forward;
	pInterface->pBackward = NeuralNetLayerDepthwiseConv2D_backward;
	pInterface->pUpdate = NeuralNetLayerDepthwiseConv2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerDepthwiseConv2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerDepthwiseConv2D_getParameters;
}

//=====================================================================================
//  層作成
//=====================================================================================
bool_t
NeuralNetLayerDepthwiseConv2D_constructLayerData(
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
	uint32_t	sizeOfParamDWFilter;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	int32_t		paddingHeight;
	int32_t		paddingWidth;
	DataShape	outputShape;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader;
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
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
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
	sizeHeader = size_in_type(sizeof(DepthwiseConv2DNeuralNetHeader), uint32_t);
	sizeOfParamDWFilter = size_in_type(sizeof(flt32_t) * inChannel * nFilter * kernelHeight * kernelWidth, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamDWFilter;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
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
		//header
		pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pDepthwiseConv2DNeuralNetHeader->super, NET_LAYER_DEPTHWISE_CONV2D, inHeight, inWidth, inChannel, sizeLayer);
		pDepthwiseConv2DNeuralNetHeader->nFilter		= nFilter;
		pDepthwiseConv2DNeuralNetHeader->kernelHeight	= kernelHeight;
		pDepthwiseConv2DNeuralNetHeader->kernelWidth	= kernelWidth;
		pDepthwiseConv2DNeuralNetHeader->strideHeight	= strideHeight;
		pDepthwiseConv2DNeuralNetHeader->strideWidth	= strideWidth;
		pDepthwiseConv2DNeuralNetHeader->fPadding		= fPadding;
		pLayer += sizeHeader;
		//DWFilter
		pLayer += sizeOfParamDWFilter;
	}
	//---------------------------------------------------------------------------------
	//Paddingサイズ
	//---------------------------------------------------------------------------------
	NeuralNetLayerDepthwiseConv2D_getShapeInformation(	
		fPadding, 
		inHeight, 
		inWidth, 
		inChannel,
		nFilter, 
		kernelHeight,
		kernelWidth, 
		strideHeight, 
		strideWidth,
		&paddingHeight,
		&paddingWidth,
		&outputShape);
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight = outputShape.height;
	*pInputWidth = outputShape.width;
	*pInputChannel = outputShape.channel;
	return TRUE;
}
