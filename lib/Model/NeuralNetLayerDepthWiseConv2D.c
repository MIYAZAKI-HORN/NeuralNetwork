#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDepthwiseConv2D.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  DepthwiseConv2D�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagDepthwiseConv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		nFilter;		//�t�B������
	uint32_t		kernelHeight;	//�t�B���^�[���ikernel size in height direction�j
	uint32_t		kernelWidth;	//�t�B���^�[���ikernel size in width direction�j
	uint32_t		strideHeight;	//�X�g���C�h��
	uint32_t		strideWidth;	//�X�g���C�h��
	bool_t			fPadding;		//�p�f�B���O�t���O
} DepthwiseConv2DNeuralNetHeader;

//=====================================================================================
//  DepthwiseConv2D�w�\����
//=====================================================================================
typedef struct tagDepthwiseConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} DepthwiseConv2DNeuralNetLayer;

//=====================================================================================
//  �`��֘A���v�Z
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
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Padding�T�C�Y
	//---------------------------------------------------------------------------------
	if (fPadding == TRUE) {
		//keras�ɍ��킹���o�̓T�C�Y�Fstride�Ń_�E���T���v�������
		outHeight = (inHeight + strideHeight - 1) / strideHeight;
		outWidth = (inWidth + strideWidth - 1) / strideWidth;
		//padding�T�C�Y
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
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		pOutputShape->height = 1 + ((inHeight + paddingHeight) - kernelHeight) / strideHeight;
		pOutputShape->width = 1 + ((inWidth + paddingWidth) - kernelWidth) / strideWidth;
		pOutputShape->channel = inChannel * nFilter;
	}
	return TRUE;
}

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	DepthwiseConv2DNeuralNetLayer* pDepthwiseConv2DLayer = (DepthwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDepthwiseConv2DLayer;
	DepthwiseConv2DNeuralNetHeader* pDepthwiseConv2DNeuralNetHeader = (DepthwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDepthwiseConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
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
	//���̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  �w�p�����^
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pDepthwiseConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(DepthwiseConv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pNeuralNetHeader->inChannel * pDepthwiseConv2DNeuralNetHeader->nFilter * pDepthwiseConv2DNeuralNetHeader->kernelHeight * pDepthwiseConv2DNeuralNetHeader->kernelWidth, uint32_t);
	//---------------------------------------------------------------------------------
	//�p�����^�z��|�C���^
	//---------------------------------------------------------------------------------
	if (ppFilter != NULL) {
		*ppFilter = pFilter;
	}
	return TRUE;
}

//=====================================================================================
//  ���`��
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
	//�ꎞ�ϐ�
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
	//�f�[�^�ʒu
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
	//�T�C�Y
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�������ƌ��₷���̂��߈ꎞ�ϐ��ŗ��p
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
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�̕��т� kernelHeight * kernelWidth * (nFilter�~iChan)
	//---------------------------------------------------------------------------------
	for (iChan = 0; iChan < inChannel; iChan++) {
		pOutputData = pOutputBuffer + iChan * nFilter; //1 pixel������dw����Ăł���[���͓��̓`�����l�����~depth_multifier�ŗ^������t�B���^�[������������B
		//�`�����l���̐擪
		iCornerInHeight = -paddingHeight / 2;	//padding���l�����ăV�t�g����
		iH = outHeight;
		while(iH--) {
			iCornerInWidth = -paddingWidth / 2;	//padding���l�����ăV�t�g����
			iW = outWidth;
			while(iW--) {
				pExtractedData = (flt32_t*)pTemporaryBuffer;	//�f�[�^�o�b�t�@�[
				//-----------------------------------------------------------------
				//�J�[�l���T�C�Y�f�[�^�����W
				//-----------------------------------------------------------------
				iPosInHeight = iCornerInHeight;
				j = kernelHeight;
				while(j--){
					if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
						//�͈͊O�f�[�^
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
								//�͈͊O�f�[�^
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
				//��L�؂�o�����t�B���^�[�T�C�Y�f�[�^�ɑ΂�nFilter�̃t�B���^�[��������
				//Filter�W�����сF�inChan,nFilter,Height,Width�j
				//-----------------------------------------------------------------
				pFilterHead = pFilter + iChan * nFilter * kernelDim;
				i = nFilter;
				pOutFiltered = pOutputData;
				while (i--) {
					//-----------------------------------------------------------------
					//��L�؂�o�����t�B���^�[�T�C�Y�̃f�[�^�ɑ΂��A��(F(i,j)�~D(i,j,k))����
					//-----------------------------------------------------------------
					filterdData = 0;
					pExtractedData = (flt32_t*)pTemporaryBuffer;
					j = kernelDim;
					while (j--) {
						filterdData += *pExtractedData++ * *pFilterHead++;
					}
					*pOutFiltered++ = filterdData;
				}
				pOutputData += inChannel * nFilter; //1 pixel������dw����Ăł���[���͓��̓`�����l�����~depth_multifier�ŗ^������t�B���^�[������������B
				//
				iCornerInWidth += strideWidth;	//2�������ʏ��X�i�X�g���C�h�l���j 
			}
			iCornerInHeight += strideHeight;
		}
	}
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�:X
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
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
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape = outputShape;
	return TRUE;
}

//=====================================================================================
//  �t�`��
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
	//�ꎞ�ϐ�
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
	//�f�[�^�ʒu
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
	//�`���덷�����l
	flt32_t*	pDLossArray;
	flt32_t		deltaLoss;
	//��ݍ���
	uint32_t*	pExtractedDataIndex;
	flt32_t*	pDDWFilter;
	flt32_t*	pXDataArray;
	flt32_t*	pXData;
	//�p�����^�����l
	flt32_t*	pDFilterHead;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�T�C�Y
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
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
	pInputBuffer = pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer = pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pDepthwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDDWFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pDepthwiseConv2DLayer->hOptimizer);
	//---------------------------------------------------------------------------------
	//back propagation�p�l�b�g���[�N�����p�����^�o�b�t�@
	//---------------------------------------------------------------------------------
	pDFilterHead	= pDDWFilter;
	//---------------------------------------------------------------------------------
	//�t�`���덷�o�b�t�@�[������������
	//---------------------------------------------------------------------------------
	dataSize = inHeight * inWidth * inChannel;
	pInputData = pInputBuffer;
	while (dataSize--) {
		*pInputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^(�덷�ێ�)�̕��т� kernelHeight * kernelWidth * (nFilter�~iChan)
	//---------------------------------------------------------------------------------
	pXDataArray = pDepthwiseConv2DLayer->pX;
	pDLossArray = pOutputBuffer;
	for (iChan = 0; iChan < inChannel; iChan++) {
		pOutputData = pOutputBuffer + iChan * nFilter; //1 pixel������dw����Ăł���[���͓��̓`�����l�����~depth_multifier�ŗ^������t�B���^�[������������B
		iCornerInHeight = -paddingHeight / 2;	//padding���l�����ăV�t�g����
		iH = outHeight;
		while(iH--) {
			iCornerInWidth = -paddingWidth / 2;	//padding���l�����ăV�t�g����
			iW = outWidth;
			while(iW--) {
				pExtractedDataIndex = pTemporaryBuffer;	//�f�[�^�o�b�t�@�[
				//-----------------------------------------------------------------
				//�J�[�l���T�C�Y�f�[�^�����W
				//-----------------------------------------------------------------
				iPosInHeight = iCornerInHeight;
				j = kernelHeight;
				while (j--) {
						if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
						//�͈͊O�f�[�^
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
								//�͈͊O�f�[�^
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
				//��L�؂�o�����t�B���^�[�T�C�Y�f�[�^�ɑ΂�nFilter�̃t�B���^�[��������
				//Filter�W�����сF�inChan,nFilter,Height,Width�j
				//-----------------------------------------------------------------
				pFilterHead = pFilter + iChan * nFilter * kernelDim;
				pDFilterHead = pDDWFilter + iChan * nFilter * kernelDim;
				i = nFilter;
				pDLossArray = pOutputData;
				while (i--) {
					//----------------------------------------------------------
					//�`���덷�����l
					//----------------------------------------------------------
					deltaLoss = *pDLossArray++;
					//-----------------------------------------------------------------
					//��L�؂�o�����t�B���^�[�T�C�Y�̃f�[�^�ɑ΂��A��(F(i,j)�~D(i,j,k))����
					//-----------------------------------------------------------------
					//�Ώۓ��̓f�[�^X�̈ʒu�C���f�b�N�X
					pExtractedDataIndex = (uint32_t*)pTemporaryBuffer;
					j = kernelDim;
					while (j--) {
						//----------------------------------------------------------
						//�f�[�^�ʒu
						//----------------------------------------------------------
						if (*pExtractedDataIndex != OUT_OF_REGION_INDICATION_VALUE) {
							pInputData = pInputBuffer + *pExtractedDataIndex;	//�t�`���o�́F���`���̓��͕����֓n�������l
							pXData = pXDataArray + *pExtractedDataIndex;		//���`�����̓��͒lX
							//----------------------------------------------------------
							//�t�B���^�[�W��(�w�K�p�����^)�����l�֐ώZ
							//----------------------------------------------------------
							*pDFilterHead += (*pXData) * deltaLoss;
							//----------------------------------------------------------
							//�t�`���o�͂֐ώZ
							//----------------------------------------------------------
							*pInputData += (*pFilterHead) * deltaLoss;
						}
						//----------------------------------------------------------
						//�|�C���^�X�V
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
	//�t�`���o�̓f�[�^�T�C�Y�`��(���`���̓��̓f�[�^�`��)
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, inChannel);
	return TRUE;
}

//=====================================================================================
//  �p�����^�X�V
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pParameter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pDepthwiseConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerDepthwiseConv2D_getLayerParameter(pDepthwiseConv2DNeuralNetHeader, &pFilter);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight = pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	paramSize = nFilter * kernelHeight * kernelWidth * inChannel;
	normSize = paramSize;;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pFilter, paramSize, normSize);
	return TRUE;
}

//=====================================================================================
//  �w���擾
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
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pDepthwiseConv2DNeuralNetHeader->nFilter;
	kernelHeight = pDepthwiseConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pDepthwiseConv2DNeuralNetHeader->kernelWidth;
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = inChannel * nFilter * kernelHeight * kernelWidth;	//dw filter
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(DepthwiseConv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����̌v�Z�o�b�t�@�[�T�C�Y
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = size_in_type(sizeof(flt32_t) * pDepthwiseConv2DNeuralNetHeader->kernelHeight * pDepthwiseConv2DNeuralNetHeader->kernelWidth, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//�o�͌`��
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
	//���͌`��
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  �w�K�p�����^���擾
//=====================================================================================
static
bool_t
NeuralNetLayerDepthwiseConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
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
//  �w�\�z
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
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerDepthwiseConv2D_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�w�K�p�f�[�^����эœK���A���S���Y���I�u�W�F�N�g�n���h��
		if (fEnableLearning == TRUE) {
			//�w�K�p�����^�T�C�Y�`�F�b�N
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
//  �C���^�[�t�F�[�X�擾
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
//  �w�쐬
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
	//�p�����^�`�F�b�N
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
	//�p�����^
	//---------------------------------------------------------------------------------
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (inHeight < kernelHeight) {
		return FALSE;
	}
	if (inWidth < kernelWidth) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(DepthwiseConv2DNeuralNetHeader), uint32_t);
	sizeOfParamDWFilter = size_in_type(sizeof(flt32_t) * inChannel * nFilter * kernelHeight * kernelWidth, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamDWFilter;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
	}
	//---------------------------------------------------------------------------------
	//�w�f�[�^�\�z
	//---------------------------------------------------------------------------------
	if (pBuffer != NULL) {
		if (sizeOfBufferIn32BitWord < sizeLayer) {
			return FALSE;
		}
		//�o�b�t�@�[�̐擪���Z�b�g
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
	//Padding�T�C�Y
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
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight = outputShape.height;
	*pInputWidth = outputShape.width;
	*pInputChannel = outputShape.channel;
	return TRUE;
}
