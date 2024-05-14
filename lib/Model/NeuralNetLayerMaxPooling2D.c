#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerMaxPooling2D.h"

//=====================================================================================
//  MaxPooling2D�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagMaxPooling2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		poolingHeight;	//�v�[�����O��
	uint32_t		poolingWidth;	//�v�[�����O��
	uint32_t		strideHeight;	//�X�g���C�h��
	uint32_t		strideWidth;	//�X�g���C�h��
} MaxPooling2DNeuralNetHeader;

//=====================================================================================
//  MaxPooling2D�w�\����
//=====================================================================================
typedef struct tagMaxPooling2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	uint32_t*		pMaxValueIndex;	//�덷�t�`���p�f�[�^�o�b�t�@
} MaxPooling2DNeuralNetLayer;

//=====================================================================================
//  �`��֘A���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getShapeInformation(
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
	uint32_t	poolingHeight,
	uint32_t	poolingWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	DataShape*	pOutputShape)
{
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
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		pOutputShape->height = 1 + (inHeight - poolingHeight) / strideHeight;
		pOutputShape->width = 1 + (inWidth - poolingWidth) / strideWidth;
		pOutputShape->channel = inChannel;
	}
	return TRUE;
}

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		pOutputShape);
	//---------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  MaxPooling�w�@���`��
//=====================================================================================
bool_t
NeuralNetLayerMaxPooling2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	uint32_t	i, j;
	DataShape	outputShape;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inWidth;
	uint32_t	inChannel;
	//�ꎞ�ϐ�
	uint32_t	poolingHeight;
	uint32_t	poolingWidth;
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	bool_t		fEnableLearning;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	flt32_t*	pOutput;
	flt32_t		pixelValue;
	flt32_t		maxValue;
	uint32_t*	pMaxValueIndexHead;
	//�f�[�^�ʒu
	uint32_t	iH;
	uint32_t	iW;
	uint32_t	iC;
	uint32_t	iCornerInHeight;
	uint32_t	iCornerInWidth;
	flt32_t*	pInputBufferCorner;
	flt32_t*	pInputBufferPoolY;
	flt32_t*	pInputBufferPool;
	flt32_t*	pXwithMaxValue;
	uint32_t	indexOfXwithMaxValue;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		if (pMaxPooling2DLayer->pMaxValueIndex == NULL) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//�T�C�Y
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		&outputShape);
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	poolingHeight	= pMaxPooling2DNeuralNetHeader->poolingHeight;
	poolingWidth	= pMaxPooling2DNeuralNetHeader->poolingWidth;
	strideHeight	= pMaxPooling2DNeuralNetHeader->strideHeight;
	strideWidth		= pMaxPooling2DNeuralNetHeader->strideWidth;
	fEnableLearning	= pNeuralNetLayer->fEnableLearning;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//------------------------------------------------------------------------------------------
	//�ő�l�v�[�����O
	//------------------------------------------------------------------------------------------
	pOutput = pOutputBuffer;
	pMaxValueIndexHead = pMaxPooling2DLayer->pMaxValueIndex;
	iH = outHeight;
	iCornerInHeight = 0;
	while(iH--) {
		iW = outWidth;
		iCornerInWidth = 0;
		while(iW--) {
			pInputBufferCorner = pInputBuffer + (iCornerInHeight * inWidth + iCornerInWidth) * inChannel;
			iC = inChannel;
			while(iC--) {
				maxValue = *pInputBufferCorner;
				pInputBufferPoolY = pInputBufferCorner;
				pXwithMaxValue = pInputBufferPoolY;	//�ő�l��ێ�����X�̃|�C���^
				i = poolingHeight;
				while (i--) {
					pInputBufferPool = pInputBufferPoolY;
					j = poolingWidth;
					while (j--) {
						pixelValue = *pInputBufferPool;
						if (maxValue < pixelValue) {
							maxValue = pixelValue;
							pXwithMaxValue = pInputBufferPool;
						}
						pInputBufferPool += inChannel;
					}
					pInputBufferPoolY += inWidth * inChannel;
				}
				if (pMaxValueIndexHead != NULL) {
					//�ő�l�̃C���f�b�N�X
					indexOfXwithMaxValue = pXwithMaxValue - pInputBuffer;
					*pMaxValueIndexHead++ = indexOfXwithMaxValue;
				}
				pInputBufferCorner++;
				*pOutput++ = maxValue;
			}
			iCornerInWidth += strideWidth;
		}
		iCornerInHeight += strideHeight;
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
bool_t
NeuralNetLayerMaxPooling2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	uint32_t	i;
	DataShape	outputShape;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	poolingHeight;
	uint32_t	poolingWidth;
	//�ꎞ�ϐ�
	uint32_t	strideHeight;
	uint32_t	strideWidth;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	uint32_t*	pTemporaryBuffer;
	//�f�[�^�ʒu
	uint32_t	iH;
	uint32_t	iW;
	uint32_t	iChan;
	uint32_t	iCornerInHeight;
	uint32_t	iCornerInWidth;
	bool_t		fStatus;
	flt32_t*	pDLossArray;
	uint32_t	indexOfXwithMaxValue;
	uint32_t	outDataCounter;
	flt32_t*	pInput;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�T�C�Y
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
		&outputShape);
	if (fStatus == FALSE) {
		return FALSE;
	}
	outHeight = outputShape.height;
	outWidth = outputShape.width;
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	poolingHeight	= pMaxPooling2DNeuralNetHeader->poolingHeight;
	poolingWidth	= pMaxPooling2DNeuralNetHeader->poolingWidth;
	strideHeight	= pMaxPooling2DNeuralNetHeader->strideHeight;
	strideWidth		= pMaxPooling2DNeuralNetHeader->strideWidth;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;	//�ꎞ�v�Z�o�b�t�@
	//---------------------------------------------------------------------------------
	//�o�b�t�@�[�̏�����
	//---------------------------------------------------------------------------------
	//�덷�o�̓o�b�t�@�[
	pInput = pInputBuffer;
	i = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	while (i--) {
		*pInput++ = 0.0f;
	}
	//------------------------------------------------------------------------------------------
	//�ő�l�v�[�����O
	//------------------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	outDataCounter = 0;
	iH = outHeight;
	iCornerInHeight = 0;
	while (iH--) {
		iW = outWidth;
		iCornerInWidth = 0;
		for (iW = 0; iW < outWidth; iW++) {
			for (iChan = 0; iChan < inChannel; iChan++) {
				//------------------------------------------------------------------------------------------
				//�ő�l��ێ������X�ɑ΂��Č덷��`��
				//------------------------------------------------------------------------------------------
				//�ő�l�̃C���f�b�N�X
				indexOfXwithMaxValue = pMaxPooling2DLayer->pMaxValueIndex[outDataCounter];
				//�ő�l��X�ɑ΂��덷��ώZ����
				pInputBuffer[indexOfXwithMaxValue] += (*pDLossArray);
				//�`���덷�����l��i�߂�
				pDLossArray++;
				outDataCounter++;
			}
			iCornerInWidth += strideWidth;
		}
		iCornerInHeight += strideHeight;
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
NeuralNetLayerMaxPooling2D_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerMaxPooling2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pMaxPooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = 0;
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(MaxPooling2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(uint32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����̌v�Z�o�b�t�@�[�T�C�Y
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = 0;
	}
	//---------------------------------------------------------------------------------
	//�o�͌`��
	//---------------------------------------------------------------------------------
	NeuralNetLayerMaxPooling2D_getShapeInformation(
		pNeuralNetHeader->inHeight,
		pNeuralNetHeader->inWidth,
		pNeuralNetHeader->inChannel,
		pMaxPooling2DNeuralNetHeader->poolingHeight,
		pMaxPooling2DNeuralNetHeader->poolingWidth,
		pMaxPooling2DNeuralNetHeader->strideHeight,
		pMaxPooling2DNeuralNetHeader->strideWidth,
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
NeuralNetLayerMaxPooling2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	if (ppParameters != NULL) {
		*ppParameters = NULL;
	}
	if (pNumberOfParameters != NULL) {
		*pNumberOfParameters = 0;
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerMaxPooling2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	MaxPooling2DNeuralNetLayer* pMaxPooling2DLayer = (MaxPooling2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pMaxPooling2DLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerMaxPooling2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerMaxPooling2D_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�w�K�p�f�[�^����эœK���A���S���Y���I�u�W�F�N�g�n���h��
		if (fEnableLearning == TRUE) {
			pObjectWork += size_in_type(sizeof(MaxPooling2DNeuralNetLayer), uint32_t);
			pMaxPooling2DLayer->pMaxValueIndex = (uint32_t*)pObjectWork;
		}
		else {
			pMaxPooling2DLayer->pMaxValueIndex = NULL;
		}
		return (handle_t)pMaxPooling2DLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerMaxPooling2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerMaxPooling2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerMaxPooling2D_construct;
	pInterface->pGetShape = NeuralNetLayerMaxPooling2D_getShape;
	pInterface->pForward = NeuralNetLayerMaxPooling2D_forward;
	pInterface->pBackward = NeuralNetLayerMaxPooling2D_backward;
	pInterface->pUpdate = NeuralNetLayerMaxPooling2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerMaxPooling2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerMaxPooling2D_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerMaxPooling2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	poolingHeight,
	uint32_t	poolingWidth,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	outHeight;
	uint32_t	outWidth;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	MaxPooling2DNeuralNetHeader* pMaxPooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^
	//---------------------------------------------------------------------------------
	inHeight = *pInputHeight;
	inWidth = *pInputWidth;
	inChannel = *pInputChannel;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (strideHeight == 0) {
		return FALSE;
	}
	if (strideWidth == 0) {
		return FALSE;
	}
	if (inHeight < poolingHeight) {
		return FALSE;
	}
	if (inWidth < poolingWidth) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(MaxPooling2DNeuralNetHeader), uint32_t);
	sizeLayer = sizeHeader;
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
		// header
		pMaxPooling2DNeuralNetHeader = (MaxPooling2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pMaxPooling2DNeuralNetHeader->super, NET_LAYER_MAX_POOLING2D, inHeight, inWidth, inChannel, sizeLayer);
		pMaxPooling2DNeuralNetHeader->poolingHeight = poolingHeight;
		pMaxPooling2DNeuralNetHeader->poolingWidth = poolingWidth;
		pMaxPooling2DNeuralNetHeader->strideHeight = strideHeight;
		pMaxPooling2DNeuralNetHeader->strideWidth = strideWidth;
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	outHeight		= 1 + (inHeight - poolingHeight) / strideHeight;
	outWidth		= 1 + (inWidth - poolingWidth) / strideWidth;
	*pInputHeight	= outHeight;
	*pInputWidth	= outWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
