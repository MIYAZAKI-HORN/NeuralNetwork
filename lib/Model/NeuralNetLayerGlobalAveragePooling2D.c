#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerGlobalAveragePooling2D.h"

//=====================================================================================
//  GlobalAveragePooling2D�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagGlobalAveragePooling2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} GlobalAveragePooling2DNeuralNetHeader;

//=====================================================================================
//  GlobalAveragePooling2D�w�\����
//=====================================================================================
typedef struct tagGlobalAveragePooling2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
} GlobalAveragePooling2DNeuralNetLayer;

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, 1, 1, pNeuralNetHeader->inChannel);
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
NeuralNetLayerGlobalAveragePooling2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inPixels;
	//�ꎞ�ϐ�
	bool_t		fEnableLearning;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	flt32_t*	pChannelInput;
	flt32_t*	pInput;
	flt32_t*	pOutput;
	flt32_t		normalizationFactor;
	flt32_t		averageValue;
	//�f�[�^�ʒu
	uint32_t	iP;
	uint32_t	iC;
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	fEnableLearning	= pNeuralNetLayer->fEnableLearning;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
	//------------------------------------------------------------------------------------------
	//���ϒl�v�[�����O
	//------------------------------------------------------------------------------------------
	inPixels = inHeight * inWidth;
	normalizationFactor = 1.0f / (flt32_t)inPixels;
	pChannelInput = pInputBuffer;
	pOutput = pOutputBuffer;
	iC = inChannel;
	while(iC--) {
		//�`���l�����Ƃ̕��ϒl���v�Z
		averageValue = 0.0f;
		pInput = pChannelInput;	//���̓o�b�t�@(�`�����l���擪)
		iP = inPixels;
		while(iP--) {
			averageValue += *pInput;
			pInput += inChannel;	//���̃`���l���f�[�^�ֈړ�
		}
		averageValue *= normalizationFactor;
		*pOutput++ = averageValue;
		pChannelInput++;
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, 1, 1, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  �t�`��
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
	uint32_t	i;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inPixels;
	//�ꎞ�ϐ�
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	flt32_t		normalizationFactor;
	flt32_t*	pDLossArray;
	flt32_t*	pChannelInput;
	flt32_t*	pInput;
	//�f�[�^�ʒu
	uint32_t	iP;
	uint32_t	iC;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight		= pNeuralNetHeader->inHeight;
	inWidth			= pNeuralNetHeader->inWidth;
	inChannel		= pNeuralNetHeader->inChannel;
	pInputBuffer	= pPropagationInfo->pInputBuffer;		//���̓o�b�t�@
	pOutputBuffer	= pPropagationInfo->pOutputBuffer;		//�o�̓o�b�t�@
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
	//���ϒl�v�[�����O
	//------------------------------------------------------------------------------------------
	inPixels = inHeight * inWidth;
	normalizationFactor = 1.0f / (flt32_t)inPixels;
	pDLossArray = pOutputBuffer;
	pChannelInput = pInputBuffer;
	iC = inChannel;
	while (iC--) {
		pInput = pChannelInput;	//���̓o�b�t�@(�`�����l���擪)
		iP = inPixels;
		while (iP--) {
			*pInput = (*pDLossArray) * normalizationFactor;
			pInput += inChannel;	//���̃`���l���f�[�^�ֈړ�
		}
		pDLossArray++;
		pChannelInput++;
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
NeuralNetLayerGlobalAveragePooling2D_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerGlobalAveragePooling2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pGlobalAveragePooling2DNeuralNetHeader;
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(GlobalAveragePooling2DNeuralNetLayer), uint32_t);
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
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, 1, 1, pNeuralNetHeader->inChannel);
	}
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
NeuralNetLayerGlobalAveragePooling2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerGlobalAveragePooling2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	GlobalAveragePooling2DNeuralNetLayer* pGlobalAveragePooling2DLayer = (GlobalAveragePooling2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pGlobalAveragePooling2DLayer;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	NeuralNetLayerGlobalAveragePooling2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerGlobalAveragePooling2D_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		return (handle_t)pGlobalAveragePooling2DLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerGlobalAveragePooling2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerGlobalAveragePooling2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerGlobalAveragePooling2D_construct;
	pInterface->pGetShape = NeuralNetLayerGlobalAveragePooling2D_getShape;
	pInterface->pForward = NeuralNetLayerGlobalAveragePooling2D_forward;
	pInterface->pBackward = NeuralNetLayerGlobalAveragePooling2D_backward;
	pInterface->pUpdate = NeuralNetLayerGlobalAveragePooling2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerGlobalAveragePooling2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerGlobalAveragePooling2D_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_constructLayerData(
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
	GlobalAveragePooling2DNeuralNetHeader* pGlobalAveragePooling2DNeuralNetHeader;
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
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(GlobalAveragePooling2DNeuralNetHeader), uint32_t);
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
		pGlobalAveragePooling2DNeuralNetHeader = (GlobalAveragePooling2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pGlobalAveragePooling2DNeuralNetHeader->super, NET_LAYER_GLOBAL_AVERAGE_POOLING2D, inHeight, inWidth, inChannel, sizeLayer);
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight	= 1;
	*pInputWidth	= 1;
	*pInputChannel	= inChannel;
	return TRUE;
}
