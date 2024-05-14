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
	uint32_t		nFilter;		//�t�B������
	uint32_t		kernelHeight;	//�t�B���^�[���ikernel size in height direction�j
	uint32_t		kernelWidth;	//�t�B���^�[���ikernel size in width direction�j
	uint32_t		kernelChannel;	//�t�B���^�[�[�ikernel size in channel direction�j
	uint32_t		strideHeight;	//�X�g���C�h��
	uint32_t		strideWidth;	//�X�g���C�h��
	bool_t			fPadding;		//�p�f�B���O�t���O
} Conv2DNeuralNetHeader;

//=====================================================================================
//  Conv2D�w�\����
//=====================================================================================
typedef struct tagConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} Conv2DNeuralNetLayer;

//=====================================================================================
//  �`��֘A���v�Z
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
		//stride�Ń_�E���T���v������
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
		pOutputShape->channel = nFilter;
	}
	return TRUE;
}

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
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
NeuralNetLayerConv2D_getLayerParameter(Conv2DNeuralNetHeader* pConv2DNeuralNetHeader, flt32_t** ppFilter, flt32_t** ppBias)
{
	uint32_t*	pLayerParam;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(Conv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel, uint32_t);
	pBias = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->nFilter, uint32_t); //bias�̓t�B���^�[��������
	//---------------------------------------------------------------------------------
	//�p�����^�z��|�C���^
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
//  ���`��
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
	//�ꎞ�ϐ�
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//�f�[�^�ʒu
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
	//�T�C�Y
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�ꎞ�ϐ����p
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	nFilter			= pConv2DNeuralNetHeader->nFilter;
	kernelHeight	= pConv2DNeuralNetHeader->kernelHeight;
	kernelWidth		= pConv2DNeuralNetHeader->kernelWidth;
	strideHeight	= pConv2DNeuralNetHeader->strideHeight;
	strideWidth		= pConv2DNeuralNetHeader->strideWidth;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer= pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//---------------------------------------------------------------------------------
	//�t�B���^�[�T�C�Y���̃f�[�^���X�g���C�h���Ȃ���1�����̃f�[�^�ɕϊ�����
	//---------------------------------------------------------------------------------
	nKernelWidthChanSize	= kernelWidth * inChannel;
	nImageWidthChannel		= inWidth * inChannel;
	nFilterSize				= kernelHeight * kernelWidth * inChannel;
	pOutput = pOutputBuffer;
	iCornerInHeight = -paddingHeight / 2;	//padding���l�����ăV�t�g����;
	iH = outHeight;
	while(iH--) {
		iCornerInWidth = -paddingWidth / 2;	//padding���l�����ăV�t�g����
		iW = outWidth;
		while(iW--) {
			pExtractedData = (flt32_t*)pTemporaryBuffer;
			//-----------------------------------------------------------------
			//fiterHeight�~kernelWidth�~Channel�̃f�[�^�����W
			//-----------------------------------------------------------------
			iPosInHeight = iCornerInHeight;
			k = kernelHeight;
			while (k--) {
				iPosInWidth = iCornerInWidth;
				if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight) {
					//�͈͊O�f�[�^��0�Ƃ���
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
							//�͈͊O�f�[�^��0�Ƃ���
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
			//nFilter�̃t�B���^�[��������
			//-----------------------------------------------------------------
			pFilterHead = pFilter;
			pBiasHead	= pBias;
			i			= nFilter;
			while (i--) {
				filterdData = *pBiasHead++; //bias�̓t�B���^�[�����������݂���
				pExtractedData = (flt32_t*)pTemporaryBuffer;
				//�t�B���^�[�v�Z�F�t�B���^�[�f�[�^��kernelHeight*kernelWidth*inChannel�ŕ���ł���
				j = nFilterSize;
				while (j--) {
					filterdData += *pExtractedData++ * *pFilterHead++;
				}
				*pOutput++ = filterdData;
			}
			iCornerInWidth += strideWidth;	//�X�g���C�h���ړ�  
		}
		iCornerInHeight += strideHeight;	//�X�g���C�h���ړ� 
	}
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
		if (pConv2DLayer->pX == NULL) {
			return FALSE;
		}
		//���̓f�[�^�R�s�[
		pInput = pInputBuffer;
		pX = pConv2DLayer->pX;
		i = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		while (i--) {
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
	//�ꎞ�ϐ�
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
	//�f�[�^�ʒu
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
	//�p�����^�����|�C���^
	flt32_t*	pDFilter;
	flt32_t*	pDBias;
	uint32_t	size;
	flt32_t*	pDFilterHead;
	flt32_t*	pDBiasHead;
	flt32_t*	pInputHead;
	flt32_t		deltaLoss;
	//�w�o�̓}�g���N�X
	DataShape	outputShape;
	bool_t		fStatus;
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pConv2DNeuralNetHeader->nFilter;
	kernelHeight = pConv2DNeuralNetHeader->kernelHeight;
	kernelWidth = pConv2DNeuralNetHeader->kernelWidth;
	strideHeight = pConv2DNeuralNetHeader->strideHeight;
	strideWidth = pConv2DNeuralNetHeader->strideWidth;
	pInputBuffer = pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer = pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pConv2DLayer->hOptimizer);
	pDBias = pDFilter + pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel;
	//---------------------------------------------------------------------------------
	//�덷�o�̓o�b�t�@�[
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInputHead = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInputHead++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//kernel�̃s�N�Z�����X�g���C�h���Ȃ���1�����̃f�[�^�ɕϊ�����
	//inHeight*inWidth����fiterHeight�~kernelWidth���̃f�[�^��strid���Ď擾���邱�Ƃ�Channel�����{����B
	//---------------------------------------------------------------------------------
	nKernelWidthChanSize	= kernelWidth * inChannel;
	nImageWidthChannel		= inWidth * inChannel;
	nFilterSize				= kernelHeight * kernelWidth * inChannel;
	//���̓}�g���N�X����؂�o���ꂽ3�����f�[�^�ɑ΂��ăt�B���^�[��K�p����
	pDLossArray = pPropagationInfo->pOutputBuffer;
	deltaLoss = (*pDLossArray);
	pXDataArray = pConv2DLayer->pX;
	iCornerInHeight = -paddingHeight / 2;	//padding���l�����ăV�t�g����;
	iH = outHeight;
	while (iH--) {
		iCornerInWidth = -paddingWidth / 2;	//padding���l�����ăV�t�g����
		iW = outWidth;
		while (iW--) {
			pExtractedDataIndex = (uint32_t*)pPropagationInfo->pTemporaryBuffer;	//�f�[�^�C���f�b�N�X�o�b�t�@�[�@H�~W�~Chan(�[��)
			//-----------------------------------------------------------------
			//���̓C���[�W���̃t�B���^�[fiterHeight�~kernelWidth�~Channel�̃s�N�Z�������W
			//-----------------------------------------------------------------
			iPosInHeight = iCornerInHeight;
			k = kernelHeight;
			while (k--) {
				if (iPosInHeight < 0 || iPosInHeight >= (int32_t)inHeight ) {
					//�͈͊O�������f�[�^�C���f�b�N�X���Z�b�g����
					i = kernelWidth * inChannel;
					while (i--) {
						*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;	//���p���ꂽ�f�[�^�̃C���f�b�N�X
					}
				}
				else {
					iPosInWidth = iCornerInWidth;
					dataIndex = (iPosInHeight * inWidth + iPosInWidth) * inChannel;
					j = kernelWidth;
					while (j--) {
						if (iPosInWidth < 0 || iPosInWidth >= (int32_t)inWidth) {
							//�͈͊O�������f�[�^�C���f�b�N�X���Z�b�g����
							i = inChannel;
							while (i--) {
								*pExtractedDataIndex++ = OUT_OF_REGION_INDICATION_VALUE;	//���p���ꂽ�f�[�^�̃C���f�b�N�X
							}
							dataIndex += inChannel;
						}
						else {
							i = inChannel;
							while (i--) {
								*pExtractedDataIndex++ = dataIndex++;	//���p���ꂽ�f�[�^�̃C���f�b�N�X
							}
						}
						iPosInWidth++;
					}
				}
				iPosInHeight++;
			}
			//-----------------------------------------------------------------
			//���W���ꂽ�t�B���^�[fiterHeight�~kernelWidth�~Channel�ɑ΂�nFilter�̃t�B���^�[��������
			//-----------------------------------------------------------------
			pFilterHead = pFilter;
			pDFilterHead = pDFilter;
			pDBiasHead = pDBias;
			i = nFilter;
			while (i--) {
				//----------------------------------------------------------
				//�t�`�����͂̓`���덷�����l
				//----------------------------------------------------------
				deltaLoss = (*pDLossArray++);
				//----------------------------------------------------------
				//bias�����l�ώZ
				//----------------------------------------------------------
				*pDBiasHead++ += deltaLoss;
				//----------------------------------------------------------
				//�Ώۓ��̓f�[�^�ʒu�C���f�b�N�X
				//----------------------------------------------------------
				pExtractedDataIndex = (uint32_t*)pPropagationInfo->pTemporaryBuffer;
				//----------------------------------------------------------
				//���Z�ʍ팸�L�q�F�t�B���^�[�v�Z�ɂ��������ăf�[�^�͕��ׂĂ��� kernelHeight*kernelWidth*inChannel
				//----------------------------------------------------------
				j = nFilterSize;
				while (j--) {
					if (*pExtractedDataIndex != OUT_OF_REGION_INDICATION_VALUE) {
						pInputData = pPropagationInfo->pInputBuffer + *pExtractedDataIndex;	//�����l�����`���̓��͕����֓n���o�b�t�@�ʒu
						pXData = pXDataArray + *pExtractedDataIndex;		//���`�����ɕێ��������͒lX
						//----------------------------------------------------------
						//�t�B���^�[�W�������l�֐ώZ
						//----------------------------------------------------------
						*pDFilterHead += (*pXData) * deltaLoss;
						//----------------------------------------------------------
						//�t�`���o�́iinputBuffer�j�֐ώZ
						//----------------------------------------------------------
						*pInputData += (*pFilterHead) * deltaLoss;
					}
					//----------------------------------------------------------
					//�|�C���^�X�V
					//----------------------------------------------------------
					pExtractedDataIndex++;
					pFilterHead++;
					pDFilterHead++;
				}
			}
			iCornerInWidth += strideWidth;	//�X�g���C�h���ړ�  
		}
		iCornerInHeight += strideHeight;	//�X�g���C�h���ړ� 
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
NeuralNetLayerConv2D_update(handle_t hLayer) {
	Conv2DNeuralNetLayer* pConv2DLayer = (Conv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pConv2DLayer;
	Conv2DNeuralNetHeader* pConv2DNeuralNetHeader = (Conv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	bool_t		fStatus;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pConv2DLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pParameter, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
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
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerConv2D_getLayerParameter(pConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
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
//  �w���擾
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
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pConv2DNeuralNetHeader->nFilter * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth * pConv2DNeuralNetHeader->kernelChannel;	//filter
		*pNumberOfLearningParameters += pConv2DNeuralNetHeader->nFilter;	//Beta
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(Conv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����̌v�Z�o�b�t�@�[�T�C�Y
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = size_in_type(sizeof(flt32_t) * pConv2DNeuralNetHeader->kernelChannel * pConv2DNeuralNetHeader->kernelHeight * pConv2DNeuralNetHeader->kernelWidth, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//�o�͌`��
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
NeuralNetLayerConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
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
//  �w�\�z
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
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerConv2D_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�w�K�p�f�[�^����эœK���A���S���Y���I�u�W�F�N�g�n���h��
		if (fEnableLearning == TRUE) {
			//�w�K�p�����^�T�C�Y�`�F�b�N
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
//  �C���^�[�t�F�[�X�擾
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
//  �w�쐬
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
	inHeight = *pInputHeight;
	inWidth = *pInputWidth;
	inChannel = *pInputChannel;
	kernelChannel = *pInputChannel;
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
	sizeHeader = size_in_type(sizeof(Conv2DNeuralNetHeader), uint32_t);
	sizeOfParamFilter = size_in_type(sizeof(flt32_t) * nFilter * kernelHeight * kernelWidth * kernelChannel, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * nFilter, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamFilter + sizeOfParamB;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
	}
	//---------------------------------------------------------------------------------
	//�w�f�[�^�\�z
	//---------------------------------------------------------------------------------
	if (pBuffer != NULL) {
		//�T�C�Y�`�F�b�N
		if (sizeOfBufferIn32BitWord < sizeLayer) {
			return FALSE;
		}
		//�o�b�t�@�[�̐擪���Z�b�g
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
	//Padding�T�C�Y
	//---------------------------------------------------------------------------------
	NeuralNetLayerConv2D_getShapeInformation(fPadding, inHeight, inWidth, nFilter,kernelHeight, kernelWidth, strideHeight, strideWidth, &paddingHeight, &paddingWidth,&outputShape);
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight = outputShape.height;
	*pInputWidth = outputShape.width;
	*pInputChannel = outputShape.channel;
	return TRUE;
}
