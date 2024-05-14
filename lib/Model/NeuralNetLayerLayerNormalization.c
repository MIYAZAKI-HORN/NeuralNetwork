#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerLayerNormalization.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON	(0.00001f)

//=====================================================================================
//  LayerNormalization�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagLayerNormalizationNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} LayerNormalizationNeuralNetHeader;

//=====================================================================================
//  LayerNormalization�w�\����
//=====================================================================================
typedef struct tagLayerNormalizationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@�Fx
	flt32_t			mean;
	flt32_t			invStd;
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} LayerNormalizationNeuralNetLayer;

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	return TRUE;
}

//=====================================================================================
//  ���`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_forward_calculation(
	uint32_t	size,
	flt32_t		gamma,
	flt32_t		beta,
	flt32_t*	pMean,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer) {
	uint32_t	i;
	flt32_t		x;
	flt32_t		X;
	flt32_t		y;
	flt32_t		sumOfX		= 0.0f;
	flt32_t		sumOfVar	= 0.0f;
	flt32_t		mean		= 0.0f;
	flt32_t		var			= 1.0f;
	flt32_t		invVar		= 1.0f;
	flt32_t		diff;
	flt32_t*	pInputData;
	//���ϒl
	pInputData = pInputBuffer;
	i = size;
	while (i--) {
		sumOfX += *pInputData++;
	}
	mean = sumOfX / (flt32_t)size;
	//���U
	pInputData = pInputBuffer;
	i = size;
	while (i--) {
		diff = *pInputData++ - mean;
		sumOfVar += diff * diff;
	}
	var = sumOfVar / (flt32_t)size;
	//���U�t��
	invVar = 1.0f / (var+ EPSILON);
	//���ϒl�ƕ��U��ێ�
	*pMean = mean;
	*pInvStd = invVar;
	//���`���f�[�^
	i = size;
	while (i--) {
		x = *pInputBuffer++;
		X = (x - mean) * invVar;
		//���`���o��
		y = gamma * X + beta;
		*pOutputBuffer++ = y;
		//�t�`���p�ێ�
		if (pX != NULL) {
			*pX++ = X;	//gamma�̔����l�ƂƂ��ĕێ��Fy = gamma * X + beta
		}
	}
	return TRUE;
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_backward_calculation(
	uint32_t	size,
	flt32_t		gamma,
	flt32_t		invStd,
	flt32_t*	pX,
	flt32_t*	pDGamma,
	flt32_t*	pDBeta,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer)
{
	uint32_t	i;
	flt32_t*	pInput;
	flt32_t*	pDLossArray;
	//---------------------------------------------------------------------------------
	//�p�����^�̌덷�t�`��
	//---------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	pInput = pInputBuffer;
	i = size;
	while (i--) {
		//�p�����^�����l
		*pDGamma += (*pX++) * (*pDLossArray);
		*pDBeta += (*pDLossArray);
		//�t�`��
		*pInput++ += gamma * invStd * (*pDLossArray);
		//�t�`���͒l�`���덷�����l�|�C���^�X�V
		pDLossArray++;
	}
	return TRUE;
}

//=====================================================================================
//  ���`��
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	flt32_t*	pX;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//�d�݃p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);	//Header�������ٓ�
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
		if (pLayerNormalizationLayer->pX == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//���́A���͐ώZ�A���U�ώZ
		//-------------------------------------------------------
		pX	= pLayerNormalizationLayer->pX;
	}
	else {
		pX	= NULL;
	}
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//���͎����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���K�������̑Ώ����f�[�^
	//---------------------------------------------------------------------------------
	size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel;
	//---------------------------------------------------------------------------------
	//���K�������Fchannel�����ɂ�����ꍇ��width�����ɂ�����ꍇ������
	//---------------------------------------------------------------------------------
	NeuralNetLayerLayerNormalization_forward_calculation(size,*pGamma,*pBeta,&pLayerNormalizationLayer->mean,&pLayerNormalizationLayer->invStd,pX,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape.height = pPropagationInfo->dataShape.height;
	pPropagationInfo->dataShape.width = pPropagationInfo->dataShape.width;
	pPropagationInfo->dataShape.channel = pPropagationInfo->dataShape.channel;
	return TRUE;
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pDGamma;
	flt32_t*	pDBeta;
	flt32_t*	pInput;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Gamma
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�C���^�[�t�F�C�X�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pLayerNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾�F�p�����^�͘A�����ē����Ă���
	//---------------------------------------------------------------------------------
	pDGamma = optimizerFunctionTable.pGetDeltaParameterBuffer(pLayerNormalizationLayer->hOptimizer);
	pDBeta = pDGamma + 1;
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//�덷�`����o�b�t�@�[�̏�����
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInput = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInput++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//�t�`���v�Z
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	NeuralNetLayerLayerNormalization_backward_calculation(size,*pGamma,pLayerNormalizationLayer->invStd,pLayerNormalizationLayer->pX,pDGamma,pDBeta,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//�o�́i���͕����j�f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape.height = pPropagationInfo->dataShape.height;
	pPropagationInfo->dataShape.width = pPropagationInfo->dataShape.width;
	pPropagationInfo->dataShape.channel = pPropagationInfo->dataShape.channel;
	return TRUE;
}

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_update(handle_t hLayer) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam;
	flt32_t* pGamma;
	flt32_t* pBeta;
	OptimizerFunctionTable	optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�C���^�[�t�F�C�X�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pLayerNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�d�݃p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�w�w�K�p�����^�X�V�FGamma&Beta
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pLayerNormalizationLayer->hOptimizer, pGamma);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t*	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	uint32_t	paramSize;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pLayerNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);	//Header�������ٓ�
	//Gamma
	pGamma = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	//Beta
	pBeta = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * 1, uint32_t);
	//---------------------------------------------------------------------------------
	//�w�p�����^������
	//---------------------------------------------------------------------------------
	//pGamma
	paramSize = 1;
	set_constant_initial_values(pGamma, paramSize, 1.0f);
	//pBeta
	paramSize = 1;
	set_constant_initial_values(pBeta, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	uint32_t inputDataDim;
	if (pLayerNormalizationNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = 1;	//Gamma
		*pNumberOfLearningParameters += 1;	//Beta
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(LayerNormalizationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			//X
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����v�Z�o�b�t�@�[�T�C�Y
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = 0;
	}
	//---------------------------------------------------------------------------------
	//���o�͌`��
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	}
	return TRUE;
}

//=====================================================================================
//  �w�K�p�����^���擾
//=====================================================================================
static
bool_t
NeuralNetLayerLayerNormalization_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerLayerNormalization_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerLayerNormalization_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	LayerNormalizationNeuralNetLayer* pLayerNormalizationLayer = (LayerNormalizationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pLayerNormalizationLayer;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pLayerNormalizationNeuralNetHeader;
	LayerFuncTable	funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	uint32_t parameterSize;
	NeuralNetLayerLayerNormalization_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerLayerNormalization_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		if (fEnableLearning == TRUE) {
			//�w�K�p�����^�T�C�Y�`�F�b�N
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(LayerNormalizationNeuralNetLayer), uint32_t);
			//�w�K�p�o�b�t�@
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			pLayerNormalizationLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//�����o������
			pLayerNormalizationLayer->hOptimizer	= hOptimizer;
			//�o�b�t�@������
			i = inputDataDim;
			while (i--) {
				pLayerNormalizationLayer->pX[i]		= 0.0f;
			}
		}
		else {
			//�w�K�p�o�b�t�@
			pLayerNormalizationLayer->pX			= NULL;
			//�����o������
			pLayerNormalizationLayer->hOptimizer	= hOptimizer;
		}
		return (handle_t)pLayerNormalizationLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerLayerNormalization_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerLayerNormalization_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerLayerNormalization_construct;
	pInterface->pGetShape = NeuralNetLayerLayerNormalization_getShape;
	pInterface->pForward = NeuralNetLayerLayerNormalization_forward;
	pInterface->pBackward = NeuralNetLayerLayerNormalization_backward;
	pInterface->pUpdate = NeuralNetLayerLayerNormalization_update;
	pInterface->pInitializeParameters = NeuralNetLayerLayerNormalization_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerLayerNormalization_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerLayerNormalization_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamGamma;
	uint32_t	sizeOfParamBeta;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	LayerNormalizationNeuralNetHeader* pLayerNormalizationNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^
	//---------------------------------------------------------------------------------
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(LayerNormalizationNeuralNetHeader), uint32_t);
	sizeOfParamGamma = size_in_type(sizeof(flt32_t) * 1, uint32_t);
	sizeOfParamBeta = size_in_type(sizeof(flt32_t) * 1, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamGamma + sizeOfParamBeta;
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
		pLayerNormalizationNeuralNetHeader = (LayerNormalizationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pLayerNormalizationNeuralNetHeader->super, NET_LAYER_LAYER_NORMALIZATION, inHeight, inWidth, inChannel, sizeLayer);
		pLayer += sizeHeader;
		//Gamma
		pLayer += sizeOfParamGamma;
		//Beta
		pLayer += sizeOfParamBeta;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
