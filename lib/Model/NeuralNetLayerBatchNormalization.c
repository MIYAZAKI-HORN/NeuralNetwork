#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON				(0.001f)
#define DEFAULT_MOMENTUM	(0.99f)

#define INITAL_MEAN			(0.0f)
#define INITAL_INVVAR		(1.0f)

//=====================================================================================
//  BatchNormalization�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagBatchNormalizationNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
	uint32_t		unit;
} BatchNormalizationNeuralNetHeader;

//=====================================================================================
//  BatchNormalization�w�\����
//=====================================================================================
typedef struct tagBatchNormalizationNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//�덷�t�`���p�f�[�^�o�b�t�@�Fx
	flt32_t*		pSumOfX;		//�덷�t�`���p�f�[�^�o�b�t�@�Fx�ώZ
	flt32_t*		pSumOfVar;		//�덷�t�`���p�f�[�^�o�b�t�@�F(x - mean) * (x - mean)�ώZ
	flt32_t			momentum;		//�ړ����σp�����^
	uint32_t		accumulation;	//�ώZ��
	handle_t		hOptimizer;		//�I�v�e�B�}�C�U�[�n���h��
} BatchNormalizationNeuralNetLayer;

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
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
NeuralNetLayerBatchNormalization_forward_calculation(
	uint32_t	size,
	uint32_t	unit,
	flt32_t*	pGamma,
	flt32_t*	pBeta,
	flt32_t*	pMean,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pSumOfX,
	flt32_t*	pSumOfVar,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer) {
	uint32_t	i, j;
	flt32_t*	pGammaHead;
	flt32_t*	pBetaHead;
	flt32_t*	pMeanHead;
	flt32_t*	pInvStdHead;
	flt32_t		x;
	flt32_t		X;
	flt32_t		y;
	flt32_t		gamma;
	flt32_t		beta;
	flt32_t		mean;
	flt32_t		invStd;
	flt32_t*	pSumOfXHead;
	flt32_t*	pSumOfVarHead;
	i = size;
	while (i--) {
		//���`���f�[�^
		pGammaHead = pGamma;
		pBetaHead = pBeta;
		pMeanHead = pMean;
		pInvStdHead = pInvStd;
		//�t�`�d�f�[�^
		pSumOfXHead = pSumOfX;
		pSumOfVarHead = pSumOfVar;
		j = unit;
		while (j--) {
			x = *pInputBuffer++;
			gamma = *pGammaHead++;
			beta = *pBetaHead++;
			mean = *pMeanHead++;
			invStd = *pInvStdHead++;
			X = (x - mean) * invStd;	//���K��
			//���`���o��
			y = gamma * X + beta;
			*pOutputBuffer++ = y;
			//�t�`���p�ێ�
			if (pX != NULL) {
				*pX++ = X;				//gamma�̔����l�ƂƂ��ĕێ��Fy = gamma * X + beta
				*pSumOfXHead++ += x;	//���͂̕��ϒl
				*pSumOfVarHead++ += (x - mean) * (x - mean);	//���͂̕��U�l
			}
		}
	}
	return TRUE;
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_backward_calculation(
	uint32_t	size,
	uint32_t	unit,
	flt32_t*	pGamma,
	flt32_t*	pInvStd,
	flt32_t*	pX,
	flt32_t*	pDGamma,
	flt32_t*	pDBeta,
	flt32_t*	pInputBuffer,
	flt32_t*	pOutputBuffer)
{
	uint32_t	i, j;
	flt32_t* pInput;
	flt32_t* pGammaHead;
	flt32_t* pInvStdHead;
	flt32_t* pDG;
	flt32_t* pDB;
	flt32_t* pDLossArray;
	//---------------------------------------------------------------------------------
	//�p�����^�̌덷�t�`��
	//---------------------------------------------------------------------------------
	pDLossArray = pOutputBuffer;
	pInput = pInputBuffer;
	i = size;
	while (i--) {
		pGammaHead = pGamma;
		pInvStdHead = pInvStd;
		pDG = pDGamma;
		pDB = pDBeta;
		j = unit;
		while (j--) {
			//�p�����^�����l
			*pDG++ += (*pX++) * (*pDLossArray);
			*pDB++ += (*pDLossArray);
			//�t�`��
			*pInput++ += (*pGammaHead++) * (*pInvStdHead++) * (*pDLossArray);
			//�t�`���͒l�`���덷�����l�|�C���^�X�V
			pDLossArray++;
		}
	}
	return TRUE;
}

//=====================================================================================
//  ���`��
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	flt32_t*	pX;
	flt32_t*	pSumOfX;
	flt32_t*	pSumOfVar;
	uint32_t	size;
	uint32_t	unit;
	//---------------------------------------------------------------------------------
	//�d�݃p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header�������ٓ�
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
		if (pBatchNormalizationLayer->pX == NULL) {
			return FALSE;
		}
		if (pBatchNormalizationLayer->pSumOfX == NULL) {
			return FALSE;
		}
		if (pBatchNormalizationLayer->pSumOfVar == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//���́A���͐ώZ�A���U�ώZ
		//-------------------------------------------------------
		pX			= pBatchNormalizationLayer->pX;
		pSumOfX		= pBatchNormalizationLayer->pSumOfX;
		pSumOfVar	= pBatchNormalizationLayer->pSumOfVar;
	}
	else {
		pX			= NULL;
		pSumOfX		= NULL;
		pSumOfVar	= NULL;
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
	if (pPropagationInfo->dataShape.channel == pBatchNormalizationNeuralNetHeader->unit) {
		//channel�����F�ʏ�
		size = pPropagationInfo->dataShape.height * pPropagationInfo->dataShape.width;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	else {
		//width�����FDense��Ȃ�
		if (pBatchNormalizationNeuralNetHeader->unit != (pPropagationInfo->dataShape.width * pPropagationInfo->dataShape.channel)) {
			return FALSE;
		}
		size = pPropagationInfo->dataShape.height;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	//---------------------------------------------------------------------------------
	//���K�������Fchannel�����ɂ�����ꍇ��width�����ɂ�����ꍇ������
	//---------------------------------------------------------------------------------
	NeuralNetLayerBatchNormalization_forward_calculation(size,unit,pGamma,pBeta,pMean,pInvStd,pX,pSumOfX,pSumOfVar,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	if (pX != NULL) {
		//�t�`���p�ώZ��
		pBatchNormalizationLayer->accumulation += size;
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	//�ύX�Ȃ�
	return TRUE;
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t*	pLayerParam;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	flt32_t*	pDGamma;
	flt32_t*	pDBeta;
	flt32_t*	pInput;
	uint32_t	size;
	uint32_t	unit;
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
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header�������ٓ�
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�C���^�[�t�F�C�X�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pBatchNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�����l��ێ�����o�b�t�@�|�C���^�擾�F�p�����^�͘A�����ē����Ă���
	//---------------------------------------------------------------------------------
	pDGamma = optimizerFunctionTable.pGetDeltaParameterBuffer(pBatchNormalizationLayer->hOptimizer);
	pDBeta = pDGamma + pBatchNormalizationNeuralNetHeader->unit;
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
	//���K�������̑Ώۃf�[�^
	//---------------------------------------------------------------------------------
	if (inChannel == pBatchNormalizationNeuralNetHeader->unit) {
		//channel�����F�ʏ�
		size = inHeight * inWidth;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	else {
		//width�����FDense��Ȃ�
		if (pBatchNormalizationNeuralNetHeader->unit != (inWidth * inChannel)) {
			return FALSE;
		}
		size = inHeight;
		unit = pBatchNormalizationNeuralNetHeader->unit;
	}
	//---------------------------------------------------------------------------------
	//�t�`���v�Z
	//---------------------------------------------------------------------------------
	NeuralNetLayerBatchNormalization_backward_calculation(size,unit,pGamma, pInvStd, pBatchNormalizationLayer->pX,pDGamma,pDBeta,pPropagationInfo->pInputBuffer,pPropagationInfo->pOutputBuffer);
	//---------------------------------------------------------------------------------
	//�o�́i���͕����j�f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	//�ύX�Ȃ�
	return TRUE;
}

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_update(handle_t hLayer) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam;
	flt32_t* pGamma;
	flt32_t* pBeta;
	flt32_t* pMean;
	flt32_t* pInvStd;
	flt32_t factor;
	flt32_t currentVar;
	flt32_t measuredVar;
	flt32_t measuredMean;
	flt32_t sigma;
	flt32_t invStd;
	uint32_t i;
	OptimizerFunctionTable	optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//�C���^�[�t�F�C�X�擾
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pBatchNormalizationLayer->hOptimizer, &optimizerFunctionTable);
	//---------------------------------------------------------------------------------
	//�d�݃p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	//Gamma
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	pGamma = (flt32_t*)pLayerParam;
	//Beta
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pBeta = (flt32_t*)pLayerParam;
	//Mean
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pMean = (flt32_t*)pLayerParam;
	//Var
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	pInvStd = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�w�w�K�p�����^�X�V�FGamma&Beta
	//---------------------------------------------------------------------------------
	optimizerFunctionTable.pUpdate(pBatchNormalizationLayer->hOptimizer, pGamma);
	factor = 1.0f / (flt32_t)pBatchNormalizationLayer->accumulation;
	//---------------------------------------------------------------------------------
	//��w�K�p�����^�F����p�����^
	//---------------------------------------------------------------------------------
	//����
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		//�v���l
		measuredMean = pBatchNormalizationLayer->pSumOfX[i] * factor;
		//�v���l�ōX�V
		pMean[i] = pMean[i] * pBatchNormalizationLayer->momentum + measuredMean * (1.0f - pBatchNormalizationLayer->momentum);
		//�ݐσo�b�t�@������
		pBatchNormalizationLayer->pSumOfX[i] = 0.0f;
	}
	//���U�̋t��
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		//�v���l
		measuredVar = pBatchNormalizationLayer->pSumOfVar[i] * factor;
#if 0
		//���̎����ł́Amomentum����̒��ɂ���ꍇ�ƍX�V�X�s�[�h���قȂ�
		//���\���򉻂���ꍇ���������B
		sigma = low_cost_sqrt(measuredVar + EPSILON, 2);
		if (sigma > 0.0f) {
			invStd = 1.0f / sigma;
		}
		else {
			invStd = 1.0f;
		}
		pInvStd[i] = pInvStd[i] * pBatchNormalizationLayer->momentum + invStd * (1.0f - pBatchNormalizationLayer->momentum);
#else
		//---------------------------------------------------------
		//Var�̈ړ����ρF�_�����e�Ɉ�v
		//pInvStd[i]=1.0/��(Var+EPSILON) ��Var =�@(x-mean)*(x-mean)
		//---------------------------------------------------------
		//���s�l
		if (pInvStd[i] > 0.0f) {
			currentVar = pInvStd[i];  
		}
		else {
			currentVar = 1.0f;
		}
		currentVar = 1.0f / currentVar;
		currentVar = currentVar * currentVar - EPSILON;
		//�v���l�ōX�V
		currentVar = currentVar * pBatchNormalizationLayer->momentum + measuredVar * (1.0f - pBatchNormalizationLayer->momentum);
		sigma = low_cost_sqrt(currentVar + EPSILON, 2);
		if (sigma > 0.0f) {
			invStd = 1.0f / sigma;
		}
		else {
			invStd = 1.0f;
		}
		pInvStd[i] = invStd;
#endif
		//�ݐσo�b�t�@������
		pBatchNormalizationLayer->pSumOfVar[i] = 0.0f;
	}
	//�ώZ�J�E���^������
	pBatchNormalizationLayer->accumulation = 0;
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t*	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	flt32_t*	pGamma;
	flt32_t*	pBeta;
	flt32_t*	pMean;
	flt32_t*	pInvStd;
	uint32_t	paramSize;
	uint32_t	i;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pBatchNormalizationNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);	//Header�������ٓ�
	//Gamma
	pGamma = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//Beta
	pBeta = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//Mean
	pMean = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//invStd
	pInvStd = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//�w�p�����^������
	//---------------------------------------------------------------------------------
	//pGamma
	paramSize	= pBatchNormalizationNeuralNetHeader->unit;
	set_constant_initial_values(pGamma, paramSize, 1.0f);
	//pBeta
	paramSize = pBatchNormalizationNeuralNetHeader->unit;
	set_constant_initial_values(pBeta, paramSize, 0.0f);
	//pMean,pInvStd
	for (i = 0; i < pBatchNormalizationNeuralNetHeader->unit; i++) {
		pMean[i]	= INITAL_MEAN;
		pInvStd[i]	= INITAL_INVVAR;
	}
	return TRUE;
}

//=====================================================================================
//  �X�V�p�����^�ݒ�
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_setMomentum(handle_t hLayer, flt32_t momentum)
{
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)hLayer;
	NetLayerType	layerType;
	bool_t			fStatus;
	fStatus = NeuralNetLayer_getType(hLayer,&layerType);
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (layerType != NET_LAYER_BATCH_NORMALIZATION) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�͈̓`�F�b�N
	//---------------------------------------------------------------------------------
	if (momentum < 0.0f) {
		momentum = 0.0f;
	}
	if (momentum > 1.0f) {
		momentum = 1.0f;
	}
	pBatchNormalizationLayer->momentum = momentum;
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerBatchNormalization_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	uint32_t inputDataDim;
	if (pBatchNormalizationNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pBatchNormalizationNeuralNetHeader->unit;	//Gamma
		*pNumberOfLearningParameters += pBatchNormalizationNeuralNetHeader->unit;	//Beta
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(BatchNormalizationNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			//X
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//SumX,SumVar
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit * 2, uint32_t);
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
NeuralNetLayerBatchNormalization_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
#if 0
		//�w�K�p�����^�̂�
		NeuralNetLayerBatchNormalization_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
#else
		//�w�K�p�����^
		*pNumberOfParameters = pBatchNormalizationNeuralNetHeader->unit;	//Gamma
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//Beta
		//�w�K�����v�i����j�p�����^
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//Mean
		*pNumberOfParameters += pBatchNormalizationNeuralNetHeader->unit;	//invStd
#endif
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerBatchNormalization_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	BatchNormalizationNeuralNetLayer* pBatchNormalizationLayer = (BatchNormalizationNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pBatchNormalizationLayer;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pBatchNormalizationNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	uint32_t parameterSize;
	NeuralNetLayerBatchNormalization_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerBatchNormalization_getInterface(&funcTable);
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
			pObjectWork += size_in_type(sizeof(BatchNormalizationNeuralNetLayer), uint32_t);
			//�w�K�p�o�b�t�@
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			pBatchNormalizationLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			pBatchNormalizationLayer->pSumOfX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pBatchNormalizationNeuralNetHeader->unit, uint32_t);
			pBatchNormalizationLayer->pSumOfVar = (flt32_t*)pObjectWork;
			//�����o������
			pBatchNormalizationLayer->momentum			= DEFAULT_MOMENTUM;
			pBatchNormalizationLayer->accumulation		= 0;
			pBatchNormalizationLayer->hOptimizer		= hOptimizer;
			//�o�b�t�@������
			i = inputDataDim;
			while (i--) {
				pBatchNormalizationLayer->pX[i] = 0.0f;
			}
			i = pBatchNormalizationNeuralNetHeader->unit;
			while (i--) {
				pBatchNormalizationLayer->pSumOfX[i] = 0.0f;
				pBatchNormalizationLayer->pSumOfVar[i] = 0.0f;
			}
		}
		else {
			//�w�K�p�o�b�t�@
			pBatchNormalizationLayer->pX = NULL;
			pBatchNormalizationLayer->pSumOfX = NULL;
			pBatchNormalizationLayer->pSumOfVar = NULL;
			//�����o������
			pBatchNormalizationLayer->momentum			= DEFAULT_MOMENTUM;
			pBatchNormalizationLayer->accumulation		= 0;
			pBatchNormalizationLayer->hOptimizer		= hOptimizer;
		}
		return (handle_t)pBatchNormalizationLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerBatchNormalization_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerBatchNormalization_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerBatchNormalization_construct;
	pInterface->pGetShape = NeuralNetLayerBatchNormalization_getShape;
	pInterface->pForward = NeuralNetLayerBatchNormalization_forward;
	pInterface->pBackward = NeuralNetLayerBatchNormalization_backward;
	pInterface->pUpdate = NeuralNetLayerBatchNormalization_update;
	pInterface->pInitializeParameters = NeuralNetLayerBatchNormalization_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerBatchNormalization_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerBatchNormalization_constructLayerData(
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
	uint32_t	sizeOfParamMean;
	uint32_t	sizeOfParamVar;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	unit;
	BatchNormalizationNeuralNetHeader* pBatchNormalizationNeuralNetHeader;
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
	//Normalization�����������
	//---------------------------------------------------------------------------------
	if (inChannel > 1) {
		unit = inChannel;	//in channel direction
	}
	else {
		unit = inWidth;		//in width direction
	}
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(BatchNormalizationNeuralNetHeader), uint32_t);
	sizeOfParamGamma = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamBeta = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamMean = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeOfParamVar = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamGamma + sizeOfParamBeta + sizeOfParamMean + sizeOfParamVar;
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
		pBatchNormalizationNeuralNetHeader = (BatchNormalizationNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pBatchNormalizationNeuralNetHeader->super, NET_LAYER_BATCH_NORMALIZATION, inHeight, inWidth, inChannel, sizeLayer);
		pBatchNormalizationNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//Gamma
		pLayer += sizeOfParamGamma;
		//Beta
		pLayer += sizeOfParamBeta;
		//Mean
		pLayer += sizeOfParamMean;
		//Var
		pLayer += sizeOfParamVar;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}
