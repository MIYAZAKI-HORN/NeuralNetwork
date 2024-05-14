#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerPointwiseConv2D.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  PointwiseConv2D�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagPointwiseConv2DNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		nFilter;		//�t�B���^��
} PointwiseConv2DNeuralNetHeader;

//=====================================================================================
//  PointwiseConv2D�w�\����
//=====================================================================================
typedef struct tagPointwiseConv2DNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} PointwiseConv2DNeuralNetLayer;

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pPointwiseConv2DNeuralNetHeader->nFilter);
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
NeuralNetLayerPointwiseConv2D_getLayerParameter(PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader, flt32_t** ppFilter,flt32_t** ppBias)
{
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t* pLayerParam;
	flt32_t* pFilter;
	flt32_t* pBias;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pPointwiseConv2DNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	pFilter = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pPointwiseConv2DNeuralNetHeader->nFilter * pNeuralNetHeader->inChannel, uint32_t);
	pBias = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pPointwiseConv2DNeuralNetHeader->nFilter, uint32_t);
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
static
bool_t
NeuralNetLayerPointwiseConv2D_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	uint32_t	size;
	uint32_t	iFilter;
	uint32_t	iChan;
	flt32_t*	pFilterHead;
	flt32_t*	pInputHead;
	flt32_t*	pOutputData;
	flt32_t*	pBiasHead;
	uint32_t	dataSize;
	flt32_t		filterdData;
	flt32_t*	pInput;
	flt32_t*	pX;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�������ƌ��₷���̂��߈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	nFilter		= pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//point�����̏�ݍ��݂����{����
	//---------------------------------------------------------------------------------
	pOutputData = pPropagationInfo->pOutputBuffer;
	pInput = pPropagationInfo->pInputBuffer;
	//�d�݂́inFilter,nChannel�j�`��
	size = inHeight * inWidth;
	while (size--) {
		pBiasHead = pBias;
		pFilterHead = pFilter;
		iFilter = nFilter;
		while (iFilter--) {
			pInputHead = pInput;
			filterdData = *pBiasHead++;
			iChan = inChannel;
			while (iChan--) {
				filterdData += (*pInputHead++) * (*pFilterHead++);
			}
			*pOutputData++ = filterdData;
		}
		pInput += inChannel;
	}
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�:X
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
		if (pPointwiseConv2DLayer->pX == NULL) {
			return FALSE;
		}
		dataSize = inHeight * inWidth * inChannel;
		pInput = pPropagationInfo->pInputBuffer;
		pX = pPointwiseConv2DLayer->pX;
		while (dataSize--) {
			*pX++ = *pInput++;
		}
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, nFilter);
	return TRUE;
}

//=====================================================================================
//  �t�`��
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_backward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	uint32_t	i;
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	uint32_t	size;
	uint32_t	iChan;
	flt32_t*	pFilterHead;
	uint32_t	dataSize;
	flt32_t*	pInputData;
	flt32_t*	pInputArray;
	flt32_t*	pDLossArray;
	flt32_t		deltaLoss;
	flt32_t*	pDFilter;
	flt32_t*	pDBias;
	flt32_t*	pDFilterHead;
	flt32_t*	pDBiasHead;
	flt32_t*	pXArray;	//���͕ۑ��o�b�t�@
	flt32_t*	pXData;		//���͒l
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, NULL);
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pPointwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	pDFilter = optimizerFunctionTable.pGetDeltaParameterBuffer(pPointwiseConv2DLayer->hOptimizer);
	pDBias = pDFilter + (pPointwiseConv2DNeuralNetHeader->nFilter * inChannel);
	//---------------------------------------------------------------------------------
	// back propagation�p�l�b�g���[�N�����p�����^�o�b�t�@
	//---------------------------------------------------------------------------------
	pDFilterHead	= pDFilter;
	pDBiasHead		= pDBias;
	//---------------------------------------------------------------------------------
	//�덷�o�̓o�b�t�@�[��������
	//---------------------------------------------------------------------------------
	dataSize = inHeight * inWidth * inChannel;
	pInputData = pPropagationInfo->pInputBuffer;
	while (dataSize--) {
		*pInputData++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//�t�`��
	//---------------------------------------------------------------------------------
	pXArray = pPointwiseConv2DLayer->pX;
	pInputArray = pPropagationInfo->pInputBuffer;
	pDLossArray = pPropagationInfo->pOutputBuffer;
	//�t�B���^�̏d�݂��ipw_nFilter,pw_nChannel�j�œ����Ă���Ƃ���
	size = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth;
	while (size--) {
		//-----------------------------------------------------------------
		//nFilter�̃t�B���^�[��������
		//-----------------------------------------------------------------
		pFilterHead		= pFilter;		//�t�B���^�[�o�b�t�@
		pDFilterHead	= pDFilter;	//�t�B���^�[�����l�o�b�t�@
		pDBiasHead		= pDBias;		//�t�B���^�[�o�C�A�X�����l�o�b�t�@
		i = nFilter;
		while (i--) {
			//----------------------------------------------------------
			//�`���덷�����l
			//----------------------------------------------------------
			deltaLoss = *pDLossArray++;
			//----------------------------------------------------------
			//���`�����̃t�B���^�[�u���b�N�ւ̓��͂ƁA�t�`������
			//----------------------------------------------------------
			pXData = pXArray;
			pInputData = pInputArray;
			//----------------------------------------------------------
			//bias(�w�K�p�����^)�����l�ώZ
			//----------------------------------------------------------
			*pDBiasHead++ += deltaLoss;
			//----------------------------------------------------------
			//Filter(�w�K�p�����^)�����l�ώZ
			//----------------------------------------------------------
			iChan = inChannel;
			while (iChan--) {
				//----------------------------------------------------------
				//�t�B���^�[�W��(�w�K�p�����^)�����l�֐ώZ
				//----------------------------------------------------------
				*pDFilterHead++ += (*pXData++) * deltaLoss;
				//----------------------------------------------------------
				//�t�`���o�͂֐ώZ
				//----------------------------------------------------------
				*pInputData++ += (*pFilterHead++) * deltaLoss;
			}
		}
		//����pixel�Ɉړ�
		pXArray += inChannel;
		pInputArray += inChannel;
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
NeuralNetLayerPointwiseConv2D_update(handle_t hLayer) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	bool_t		fStatus;
	OptimizerFunctionTable	optimizerFunctionTable;
	NeuralNetOptimizer_getInterface(pPointwiseConv2DLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pParameter, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pPointwiseConv2DLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	flt32_t*	pFilter;
	flt32_t*	pBias;
	uint32_t	paramSize;
	uint32_t	normSize;
	bool_t		fStatus;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayerPointwiseConv2D_getLayerParameter(pPointwiseConv2DNeuralNetHeader, &pFilter, &pBias);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	//�t�B���^�W��
	paramSize	= nFilter * inChannel;
	normSize	= paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pFilter, paramSize, normSize);
	//�o�C�A�X
	paramSize	= nFilter;
	set_constant_initial_values(pBias, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerPointwiseConv2D_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pPointwiseConv2DNeuralNetHeader;
	uint32_t	inChannel;
	uint32_t	nFilter;
	if (pPointwiseConv2DNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈ꎞ�ϐ��ŗ��p
	//---------------------------------------------------------------------------------
	inChannel = pNeuralNetHeader->inChannel;
	nFilter = pPointwiseConv2DNeuralNetHeader->nFilter;
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = inChannel * nFilter;
		*pNumberOfLearningParameters += nFilter;
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(PointwiseConv2DNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			uint32_t nInput = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
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
	//�o�͌`��
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		DataShape_construct(pOutputShape, pNeuralNetHeader->inHeight, pNeuralNetHeader->inWidth, pPointwiseConv2DNeuralNetHeader->nFilter);
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
NeuralNetLayerPointwiseConv2D_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerPointwiseConv2D_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerPointwiseConv2D_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	PointwiseConv2DNeuralNetLayer* pPointwiseConv2DLayer = (PointwiseConv2DNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pPointwiseConv2DLayer;
	LayerFuncTable	funcTable;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerPointwiseConv2D_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerPointwiseConv2D_getInterface(&funcTable);
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
			pObjectWork += size_in_type(sizeof(PointwiseConv2DNeuralNetLayer), uint32_t);
			pPointwiseConv2DLayer->pX = (flt32_t*)pObjectWork;
			pPointwiseConv2DLayer->hOptimizer = hOptimizer;
		}
		else {
			pPointwiseConv2DLayer->pX = NULL;
			pPointwiseConv2DLayer->hOptimizer = NULL;
		}
		return (handle_t)pPointwiseConv2DLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerPointwiseConv2D_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerPointwiseConv2D_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerPointwiseConv2D_construct;
	pInterface->pGetShape = NeuralNetLayerPointwiseConv2D_getShape;
	pInterface->pForward = NeuralNetLayerPointwiseConv2D_forward;
	pInterface->pBackward = NeuralNetLayerPointwiseConv2D_backward;
	pInterface->pUpdate = NeuralNetLayerPointwiseConv2D_update;
	pInterface->pInitializeParameters = NeuralNetLayerPointwiseConv2D_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerPointwiseConv2D_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerPointwiseConv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	nFilter,
	uint32_t*	pSizeOfLayerIn32BitWord) 
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamFilter;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	PointwiseConv2DNeuralNetHeader* pPointwiseConv2DNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
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
	sizeHeader = size_in_type(sizeof(PointwiseConv2DNeuralNetHeader), uint32_t);
	sizeOfParamFilter = size_in_type(sizeof(flt32_t) * inChannel * nFilter, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * nFilter, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamFilter + sizeOfParamB;
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
		pPointwiseConv2DNeuralNetHeader = (PointwiseConv2DNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pPointwiseConv2DNeuralNetHeader->super, NET_LAYER_POINTWISE_CONV2D, inHeight, inWidth, inChannel, sizeLayer);
		pPointwiseConv2DNeuralNetHeader->nFilter = nFilter;
		pLayer += sizeHeader;
		//Filter
		pLayer += sizeOfParamFilter;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= nFilter;
	return TRUE;
}
