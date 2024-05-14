#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDense.h"
#include "RandomValueGenerator.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  Dense�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagDenseNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		unit;			//���j�b�g��
} DenseNeuralNetHeader;

//=====================================================================================
//  Dense�w�\����
//=====================================================================================
typedef struct tagDenseNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} DenseNeuralNetLayer;

//=====================================================================================
//  DENSE�w�@���`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pNeuralNetHeader->inChannel);
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape,1, pDenseNeuralNetHeader->unit,1);
	return TRUE;
}

//=====================================================================================
//  DENSE�w�@���`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerDense_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit * nInput, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//�d�݃}�g���b�N�X�v�Z
	//---------------------------------------------------------------------------------
	weight_matrix_with_bias_forward(pPropagationInfo->pInputBuffer, nInput, pW, pB, pPropagationInfo->pOutputBuffer, pDenseNeuralNetHeader->unit,FALSE);
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		flt32_t*	pInput = pPropagationInfo->pInputBuffer;
		flt32_t*	pX;
		uint32_t	inputDim = nInput;
		//�G���[�n���h�����O
		if (pDenseLayer->pX == NULL) {
			return FALSE;
		}
		//DENSE back propagation�f�[�^
		pX = pDenseLayer->pX;
		while (inputDim--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape,1, pDenseNeuralNetHeader->unit, 1);
	return TRUE;
}

//=====================================================================================
//  DENSE�w�@�t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerDense_backward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pDWeight;
	flt32_t*	pDBias;
	flt32_t*	pInput;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	uint32_t	size;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pDenseLayer->hOptimizer, &optimizerFunctionTable);
	pDWeight = optimizerFunctionTable.pGetDeltaParameterBuffer(pDenseLayer->hOptimizer);
	pDBias = pDWeight + pDenseNeuralNetHeader->unit * nInput;
	//---------------------------------------------------------------------------------
	//�덷�`����o�b�t�@������
	//---------------------------------------------------------------------------------
	size = nInput;
	pInput = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pInput++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//�d�݃p�����^�̌덷�t�`��
	//---------------------------------------------------------------------------------
	weight_matrix_with_bias_backward(pPropagationInfo->pInputBuffer, nInput, pW, pPropagationInfo->pOutputBuffer, pDenseNeuralNetHeader->unit, pDenseLayer->pX, pDWeight, pDBias);
	//---------------------------------------------------------------------------------
	//�o�́i���͕����j�f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, 1, nInput, 1);
	return TRUE;
}

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
static
bool_t
NeuralNetLayerDense_update(handle_t hLayer) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pDenseLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pParameter = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	//W��B
	optimizerFunctionTable.pUpdate(pDenseLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerDense_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	flt32_t*	pW;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	uint32_t	paramSize;
	uint32_t	normSize;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit * nInput, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pDenseNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//�w�p�����^������
	//---------------------------------------------------------------------------------
	//W
	paramSize = pDenseNeuralNetHeader->unit * nInput;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pW, paramSize, normSize);
	//B
	paramSize = pDenseNeuralNetHeader->unit;
	set_constant_initial_values(pB, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	DenseNeuralNetHeader* pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pDenseNeuralNetHeader;
	uint32_t	nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
	if (pDenseNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pDenseNeuralNetHeader->unit * nInput;	//W
		*pNumberOfLearningParameters += pDenseNeuralNetHeader->unit;			//B
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(DenseNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInput, uint32_t);
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����̌v�Z�o�b�t�@�[�T�C�Y
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
		DataShape_construct(pOutputShape, 1, pDenseNeuralNetHeader->unit, 1);
	}
	return TRUE;
}

//=====================================================================================
//  �w�K�p�����^���擾
//=====================================================================================
static
bool_t
NeuralNetLayerDense_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerDense_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerDense_construct(	uint32_t*	pLayerData,
								uint32_t*	pObjectWork,
								uint32_t	sizeObjectIn32BitWord,
								bool_t		fEnableLearning,
								handle_t	hOptimizer) {
	DenseNeuralNetLayer* pDenseLayer = (DenseNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pDenseLayer;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t parameterSize;
	NeuralNetLayerDense_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL,NULL,NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		LayerFuncTable	funcTable;
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerDense_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�@�w�K�p�f�[�^����эœK���A���S���Y���I�u�W�F�N�g�n���h��
		if (fEnableLearning == TRUE) {
			//�w�K�p�����^�T�C�Y�`�F�b�N
			OptimizerFunctionTable	optimizerFunctionTable;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerFunctionTable);
			parameterSize = optimizerFunctionTable.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork		+= size_in_type(sizeof(DenseNeuralNetLayer), uint32_t);
			pDenseLayer->pX			= (flt32_t*)pObjectWork;
			pDenseLayer->hOptimizer = hOptimizer;
		}
		else {
			pDenseLayer->pX			= NULL;
			pDenseLayer->hOptimizer	= NULL;
		}
		return (handle_t)pDenseLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerDense_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerDense_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerDense_construct;
	pInterface->pGetShape = NeuralNetLayerDense_getShape;
	pInterface->pForward = NeuralNetLayerDense_forward;
	pInterface->pBackward = NeuralNetLayerDense_backward;
	pInterface->pUpdate = NeuralNetLayerDense_update;
	pInterface->pInitializeParameters = NeuralNetLayerDense_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerDense_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerDense_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	unit,
	uint32_t*	pSizeOfLayerIn32BitWord)
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamW;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nInput;
	DenseNeuralNetHeader* pDenseNeuralNetHeader;
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
	nInput = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(DenseNeuralNetHeader), uint32_t);
	sizeOfParamW = size_in_type(sizeof(flt32_t) * unit * nInput, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamW + sizeOfParamB;
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
		pDenseNeuralNetHeader = (DenseNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pDenseNeuralNetHeader->super, NET_LAYER_DENSE, inHeight, inWidth, inChannel, sizeLayer);
		pDenseNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//W
		pLayer += sizeOfParamW;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight = 1;
	*pInputWidth = unit;
	*pInputChannel = 1;
	return TRUE;
}
