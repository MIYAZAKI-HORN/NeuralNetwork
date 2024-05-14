#include "STDTypeDefinition.h"
#include "SequentialNet.h" 
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerDense.h"
#include "NeuralNetLayerConv2D.h"
#include "NeuralNetLayerMaxPooling2D.h"
#include "NeuralNetLayerGlobalAveragePooling2D.h"
#include "NeuralNetLayerDepthwiseConv2D.h"
#include "NeuralNetLayerPointwiseConv2D.h"
#include "NeuralNetLayerActivation.h"
#include "NeuralNetLayerSimpleRNN.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "NeuralNetLayerLayerNormalization.h"
#include "NeuralNetLayerPreDeconv2D.h"
#include "NeuralNetLayerResidualConnection.h"
#include "RandomValueGenerator.h"
#include "NeuralNetOptimizer.h"

//=====================================================================================
//  �t�@�C���w�b�_�[
//=====================================================================================
typedef struct tagSequentialNetHeader {
	uint32_t	version;
	uint32_t	revision;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	numberOfLayers;
} SequentialNetHeader;


//=====================================================================================
// �e�w�̃C���^�[�t�F�C�X���擾����
//=====================================================================================
static
bool_t
NetLayer_getInterface(NeuralNetHeader* pNeuralNetHeader, LayerFuncTable* pInterface) {
	switch (pNeuralNetHeader->layerType) {
	case NET_LAYER_DENSE:
		NeuralNetLayerDense_getInterface(pInterface);
		break;
	case NET_LAYER_SIMPLE_RNN:
		NeuralNetLayerSimpleRNN_getInterface(pInterface);
		break;
	case NET_LAYER_CONV2D:
		NeuralNetLayerConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_DEPTHWISE_CONV2D:
		NeuralNetLayerDepthwiseConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_POINTWISE_CONV2D:
		NeuralNetLayerPointwiseConv2D_getInterface(pInterface);
		break;
	case NET_LAYER_MAX_POOLING2D:
		NeuralNetLayerMaxPooling2D_getInterface(pInterface);
		break;
	case NET_LAYER_GLOBAL_AVERAGE_POOLING2D:
		NeuralNetLayerGlobalAveragePooling2D_getInterface(pInterface);
		break;
	case NET_LAYER_BATCH_NORMALIZATION:
		NeuralNetLayerBatchNormalization_getInterface(pInterface);
		break;
	case NET_LAYER_LAYER_NORMALIZATION:
		NeuralNetLayerLayerNormalization_getInterface(pInterface);
		break;
	case NET_LAYER_ACTIVATION:
		NeuralNetLayerActivation_getInterface(pInterface, (uint32_t*)pNeuralNetHeader);
		break;
	case NET_LAYER_PREDECONV2D:
		NeuralNetLayerPreDeconv2D_getInterface(pInterface);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_SENDER:
		NeuralNetLayerResidualConnectionSender_getInterface(pInterface);
		break;
	case NET_LAYER_RESIDUAL_CONNECTION_RECEIVER:
		NeuralNetLayerResidualConnectionReceiver_getInterface(pInterface);
		break;
	default:
		//�G���[
		return FALSE;
		break;
	}
	return TRUE;
}

//=====================================================================================
//  �V�[�P���V�������f���\����
//=====================================================================================
typedef struct tagSequentialNet {
	uint32_t*	pModelData;
	flt32_t*	ppDataBuffer[2];
	uint32_t*	pTemporaryBuffer;
	uint32_t	dataBufferSize;
	uint32_t	temporaryBufferSize;
	flt32_t*	pInputBuffer;
	flt32_t*	pOutputBuffer;
	DataShape	inputShape;
	DataShape	outputShape;
	uint32_t	numberOfLayers;
	bool_t		fEnableLearning;
	bool_t		fSkipLastSoftmaxWhenBackpropagation;
	handle_t*	pLayerArray;
	handle_t*	pOptimizerArray;
	uint32_t	backPropagationBatchSize;
	uint32_t	backPropagationCounter;
	uint32_t	backPropagationEndLayerIndex;
	handle_t	hRandomValueGenerator;
} SequentialNet;

//=====================================================================================
//  �v�Z���[�N�G���A�T�C�Y�֘A���̎擾
//=====================================================================================
static 
bool_t
SequentialNet_getLayerWorkAreaInfo(
	uint32_t*				pModelData,
	uint32_t*				pLayerObjectSizeIn32BitWord,
	uint32_t*				pMaxDataSize, 
	uint32_t*				pMaxTempWorkAreaSizeIn32BitWord, 
	DataShape*				pFinalOutputShape,
	bool_t					fEnableLearning, 
	NeuralNetOptimizerType	optimizerType,
	uint32_t*				pOptimizerObjectSizeIn32BitWord,
	uint32_t				numberOfBackPropagationLayers) {
	uint32_t				i;
	bool_t					fLearnLayerParameters;			//in:back propagation flag
	uint32_t				layerObjectSizeIn32BitWord;		//out:layer object size
	uint32_t				numberOfLearningParameters;		//out:number of learning prameters for optimizer
	uint32_t				tempWorkAreaSizeIn32BitWord;	//out:temporary work area size for prediction and back propagation
	DataShape				inputShape;						//out:input data shape
	DataShape				outputShape;					//out:output data shape
	uint32_t				outputDim;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	uint32_t				totalOptimizerObjectSizeIn32BitWord;
	uint32_t				totalLayerObjectSizeIn32BitWord;
	SequentialNetHeader*	pHeader;
	NeuralNetHeader*		pNeuralNetHeader;
	DataShape				dataShape;
	uint32_t*				pLayerData;
	bool_t					fStatus;
	uint32_t				bpEndLayerIndex;
	LayerFuncTable			netLayerFuncTable;
	//----------------------------------------------------------------------------------
	//�w�b�_�[�̃Z�b�g����уo�[�W�����`�F�b�N
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	if (pHeader->version != MODEL_FILE_VERSION) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//�w���ɕK�v�ȃ��[�N�������T�C�Y�����Z����
	//----------------------------------------------------------------------------------
	dataShape.height	= pHeader->inHeight;
	dataShape.width		= pHeader->inWidth;
	dataShape.channel	= pHeader->inChannel;
	maxDataSize			= DataShape_getSize(&dataShape);
	maxTempWorkAreaSizeIn32BitWord	= 0;
	totalOptimizerObjectSizeIn32BitWord = 0;
	totalLayerObjectSizeIn32BitWord = 0;
	DataShape_initialize(&inputShape);
	DataShape_initialize(&outputShape);
	bpEndLayerIndex = pHeader->numberOfLayers - numberOfBackPropagationLayers;
	pLayerData = pModelData + size_in_type(sizeof(SequentialNetHeader), uint32_t);
	for( i=0; i< pHeader->numberOfLayers; i++ ) {
		pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
		//----------------------------------------------------------------------------------
		//�w�\�z�֘A�֐��C���^�[�t�F�[�X�擾
		//----------------------------------------------------------------------------------
		NetLayer_getInterface(pNeuralNetHeader,&netLayerFuncTable);
		//----------------------------------------------------------------------------------
		//back propagation�p�o�b�t�@�[�T�C�Y
		//----------------------------------------------------------------------------------
		if (i >= bpEndLayerIndex) {
			fLearnLayerParameters = fEnableLearning;
		}
		else {
			fLearnLayerParameters = FALSE;
		}
		//----------------------------------------------------------------------------------
		//�w���
		//----------------------------------------------------------------------------------
		tempWorkAreaSizeIn32BitWord		= 0;
		fStatus = netLayerFuncTable.pGetLayerInformation(
				pLayerData,						//in:image data
				fLearnLayerParameters,			//in:back propagation flag
				&layerObjectSizeIn32BitWord,	//out:layer object size
				&numberOfLearningParameters,	//out:number of learning prameters for optimizer
				&tempWorkAreaSizeIn32BitWord,	//out:temporary work area size for prediction and back propagation
				&inputShape,					//out:input data shape
				&outputShape					//out:output data shape
			);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//----------------------------------------------------------------------------------
		//�w�I�u�W�F�N�g�̃T�C�Y��ώZ
		//----------------------------------------------------------------------------------
		totalLayerObjectSizeIn32BitWord += layerObjectSizeIn32BitWord;
		//----------------------------------------------------------------------------------
		//optimizer
		//----------------------------------------------------------------------------------
		if (fLearnLayerParameters == TRUE && numberOfLearningParameters > 0) {
			OptimizerFunctionTable optimizerFunctionTable;
			fStatus = NeuralNetOptimizer_getInterfaceByType(optimizerType, &optimizerFunctionTable);
			if (fStatus == FALSE) {
				return FALSE;
			}
			totalOptimizerObjectSizeIn32BitWord += optimizerFunctionTable.pGetSizeIn32BitWord(numberOfLearningParameters);
		}
		//----------------------------------------------------------------------------------
		//�ő���o�̓f�[�^�T�C�Y�����W
		//----------------------------------------------------------------------------------
		outputDim = DataShape_getSize(&outputShape);
		maxDataSize = larger_of(maxDataSize, outputDim);
		//----------------------------------------------------------------------------------
		//�ő�ꎟ���[�N�G���A�T�C�Y�����W
		//----------------------------------------------------------------------------------
		maxTempWorkAreaSizeIn32BitWord = larger_of(maxTempWorkAreaSizeIn32BitWord,tempWorkAreaSizeIn32BitWord);
		//---------------------------------------------------------------------------------
		//���̑w���Ɉړ�
		//---------------------------------------------------------------------------------
		pLayerData += pNeuralNetHeader->sizeIn32BitWord;
	}
	//----------------------------------------------------------------------------------
	//�ő�f�[�^���o�͐��ƈꎟ�v�Z�̈�T�C�Y������э��v���[�N�G���A�T�C�Y
	//----------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = totalLayerObjectSizeIn32BitWord;
	}
	if( pMaxDataSize != NULL ) {
		*pMaxDataSize = maxDataSize;
	}
	if( pMaxTempWorkAreaSizeIn32BitWord != NULL ) {
		*pMaxTempWorkAreaSizeIn32BitWord	= maxTempWorkAreaSizeIn32BitWord;
	}
	if (pOptimizerObjectSizeIn32BitWord != NULL) {
		*pOptimizerObjectSizeIn32BitWord = totalOptimizerObjectSizeIn32BitWord;
	}
	if (pFinalOutputShape != NULL) {
		*pFinalOutputShape = outputShape;
	}
	return TRUE;
}

//=====================================================================================
//  �p�����^�����l�ɂ��p�����^���X�V����
//=====================================================================================
static
bool_t
SequentialNet_update(handle_t hModel) {
	uint32_t		i;
	SequentialNet*	This = (SequentialNet*)hModel;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//���f���n���h�����`�F�b�N
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	if (This->backPropagationBatchSize == 0) {
		return FALSE;
	}
	if (This->numberOfLayers == 0) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//�p�����^�X�V����
	//----------------------------------------------------------------------------------
	if (This->backPropagationCounter < This->backPropagationBatchSize) {
		return TRUE;
	}
	//---------------------------------------------------------------------------------
	//�e�w�̃p�����^���X�V����
	//---------------------------------------------------------------------------------
	for (i = This->backPropagationEndLayerIndex; i < This->numberOfLayers; i++) {
		//---------------------------------------------------------------------------------
		//
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_update(This->pLayerArray[i]);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//�t�`���J�E���^��������
	//---------------------------------------------------------------------------------
	This->backPropagationCounter = 0;
	return TRUE;
}

//=====================================================================================
//  �o�[�W�������擾
//=====================================================================================
#define NEURAL_NETWORK_MODEL_VERSION_MAJOR		(1)
#define NEURAL_NETWORK_MODEL_VERSION_MINOR		(0)
#define NEURAL_NETWORK_MODEL_VERSION_REVISION	(0)

void
SequentialNet_getVersion(uint16_t* pMajorVersion, uint16_t* pMinorVersion, uint16_t* pRevision) {
	if (pMajorVersion != NULL) {
		*pMajorVersion = NEURAL_NETWORK_MODEL_VERSION_MAJOR;
	}
	if (pMinorVersion != NULL) {
		*pMinorVersion = NEURAL_NETWORK_MODEL_VERSION_MINOR;
	}
	if (pRevision != NULL) {
		*pRevision = NEURAL_NETWORK_MODEL_VERSION_REVISION;
	}
}

//=====================================================================================
//  �K�v���[�N�G���A�T�C�Y�擾
//=====================================================================================
uint32_t	
SequentialNet_getSizeIn32BitWord(uint32_t* pModelData, bool_t fEnableLearning,NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers) {
	bool_t					fStatus;
	uint32_t				sizeOfEngineIn32BitWord = 0;
	SequentialNetHeader*	pHeader;
	uint32_t				layerObjectSizeIn32BitWord;
	uint32_t				optimizerObjectSizeIn32BitWord;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	DataShape				outputShape;
	//----------------------------------------------------------------------------------
	//���f���f�[�^�`�F�b�N
	//----------------------------------------------------------------------------------
	if (pModelData == NULL) {
		return 0;
	}
	//----------------------------------------------------------------------------------
	//�w�b�_�[�̃Z�b�g
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	//----------------------------------------------------------------------------------
	//�t�`���w�������w���𒴂���ꍇ�́A�w�����ɒu����������
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//�t�`���w����0��ݒ肵���ꍇ�͑��w���Ƃ���
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers == 0) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//��Ɨ̈�T�C�Y���擾
	//----------------------------------------------------------------------------------
	fStatus = SequentialNet_getLayerWorkAreaInfo(pModelData,&layerObjectSizeIn32BitWord ,&maxDataSize,&maxTempWorkAreaSizeIn32BitWord,&outputShape,fEnableLearning, optimizer,&optimizerObjectSizeIn32BitWord,numberOfBackPropagationLayers);
	if (fStatus == FALSE) {
		return 0;
	}
	//----------------------------------------------------------------------------------
	//���f���T�C�Y
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord = size_in_type(sizeof(SequentialNet), uint32_t);
	//----------------------------------------------------------------------------------
	//pLayerArray & pOptimizerArray
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += size_in_type(pHeader->numberOfLayers * sizeof(handle_t)*2, uint32_t);
	//----------------------------------------------------------------------------------
	//�w�I�u�W�F�N�g�T�C�Y�����Z  
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += layerObjectSizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//�I�v�e�B�}�C�U�[�T�C�Y�����Z����
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += optimizerObjectSizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//���o�̓_�u���o�b�t�@�ƈꎞ���[�N�G���A�T�C�Y
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += 2 * size_in_type(maxDataSize * sizeof(flt32_t), uint32_t);
	sizeOfEngineIn32BitWord += size_in_type(maxTempWorkAreaSizeIn32BitWord * sizeof(flt32_t), uint32_t);
	//----------------------------------------------------------------------------------
	//����������I�u�W�F�N�g�T�C�Y�����Z����
	//----------------------------------------------------------------------------------
	sizeOfEngineIn32BitWord += RandomValueGenerator_getSizeIn32BitWord();
	return sizeOfEngineIn32BitWord;
}

//=====================================================================================
//  ���f���\�z
//=====================================================================================
handle_t
SequentialNet_construct(uint32_t* pModelData,bool_t fEnableLearning,uint32_t batchSize, NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers,uint32_t* pWorkArea,uint32_t sizeOfWorkAreaIn32BitWord) {
	SequentialNet*			This;
	uint32_t				i;
	bool_t					fStatus;
	uint32_t*				pWorkAreaHead; 
	uint32_t				requiredSizeOfWorkAreaIn32BitWord;
	uint32_t				maxDataSize;
	uint32_t				maxTempWorkAreaSizeIn32BitWord;
	uint32_t				sizeIn32BitWord;
	SequentialNetHeader*	pHeader;
	NeuralNetHeader*		pNeuralNetHeader;
	uint32_t*				pLayerData;
	bool_t					fLearnLayerParameters;
	uint32_t				layerObjectSizeIn32BitWord;
	uint32_t				numberOfLearningParameters;
	uint32_t				tempWorkAreaSizeIn32BitWord;
	DataShape				inputShape;
	DataShape				outputShape;
	handle_t				hOptimizer;
	handle_t				hLayer;
	uint32_t				bpEndLayerIndex;
	LayerFuncTable			netLayerFuncTable;
	//----------------------------------------------------------------------------------
	//���f���f�[�^�`�F�b�N
	//----------------------------------------------------------------------------------
	if( pModelData == NULL ) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//��Ɨ̈�`�F�b�N
	//----------------------------------------------------------------------------------
	if( pWorkArea == NULL ) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//----------------------------------------------------------------------------------
	if (fEnableLearning == TRUE && batchSize == 0) {
		return NULL;
	}
	//----------------------------------------------------------------------------------
	//�w�b�_�[�̃Z�b�g
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)pModelData;
	//----------------------------------------------------------------------------------
	//�t�`���w�������w���𒴂���ꍇ�͑w�����ɒu������
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers > pHeader->numberOfLayers) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//�t�`���w����0��ݒ肷��Ƒ��w��
	//----------------------------------------------------------------------------------
	if (numberOfBackPropagationLayers == 0) {
		numberOfBackPropagationLayers = pHeader->numberOfLayers;
	}
	//----------------------------------------------------------------------------------
	//���[�N�G���A�Z�b�g
	//----------------------------------------------------------------------------------
	pWorkAreaHead = pWorkArea;
	//----------------------------------------------------------------------------------
	//��Ɨ̈�T�C�Y���`�F�b�N
	//----------------------------------------------------------------------------------
	requiredSizeOfWorkAreaIn32BitWord = SequentialNet_getSizeIn32BitWord(pModelData,fEnableLearning,optimizer,numberOfBackPropagationLayers);
	if(requiredSizeOfWorkAreaIn32BitWord == 0 ) {
		return NULL;
	}
	if( sizeOfWorkAreaIn32BitWord < requiredSizeOfWorkAreaIn32BitWord ) { 
		return NULL;
	}
	//-----------------------------------------------------------------------------------
	//�V�[�P���V�������f���I�u�W�F�N�g
	//-----------------------------------------------------------------------------------
	This = (SequentialNet*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(sizeof(SequentialNet), uint32_t);
	This->pLayerArray = (handle_t*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(pHeader->numberOfLayers * sizeof(handle_t), uint32_t);
	This->pOptimizerArray = (handle_t*)pWorkAreaHead;
	pWorkAreaHead += size_in_type(pHeader->numberOfLayers * sizeof(handle_t), uint32_t);
	//----------------------------------------------------------------------------------
	//����������
	//----------------------------------------------------------------------------------
	sizeIn32BitWord = RandomValueGenerator_getSizeIn32BitWord();
	This->hRandomValueGenerator = RandomValueGenerator_construct(0, pWorkAreaHead, sizeIn32BitWord);
	pWorkAreaHead += sizeIn32BitWord;
	//-----------------------------------------------------------------------------------
	//�f�[�^����уp�����^�m��
	//-----------------------------------------------------------------------------------
	This->pModelData							= pModelData;
	This->numberOfLayers						= pHeader->numberOfLayers;
	This->fEnableLearning						= fEnableLearning;
	This->fSkipLastSoftmaxWhenBackpropagation	= TRUE;
	This->backPropagationBatchSize				= batchSize;
	This->backPropagationCounter				= 0;
	This->backPropagationEndLayerIndex			= pHeader->numberOfLayers - numberOfBackPropagationLayers;
	//-----------------------------------------------------------------------------------
	//���[�N�G���A���̎擾
	//-----------------------------------------------------------------------------------
	fStatus = SequentialNet_getLayerWorkAreaInfo(
		pModelData,
		NULL,
		&maxDataSize, 
		&maxTempWorkAreaSizeIn32BitWord, 
		&outputShape,
		fEnableLearning, 
		optimizer,
		NULL, 
		numberOfBackPropagationLayers);
	if (fStatus == FALSE) {
		return NULL;
	}
	//-----------------------------------------------------------------------------------
	//�o�b�t�@�T�C�Y
	//-----------------------------------------------------------------------------------
	This->dataBufferSize = maxDataSize;	// double buffer�̂��߂���2�{�̃T�C�Y���m�ۂ����
	This->temporaryBufferSize = maxTempWorkAreaSizeIn32BitWord;
	//-----------------------------------------------------------------------------------
	//���o�͌`��̃Z�b�g
	//-----------------------------------------------------------------------------------
	DataShape_construct(&This->inputShape,pHeader->inHeight,pHeader->inWidth,pHeader->inChannel);
	This->outputShape = outputShape;
	//-----------------------------------------------------------------------------------
	//���o�̓o�b�t�@�̈���Z�b�g
	//-----------------------------------------------------------------------------------
	sizeIn32BitWord		= size_in_type(sizeof(flt32_t) * maxDataSize, uint32_t);
	for(i=0;i<2;i++) {
		This->ppDataBuffer[i]	= (flt32_t*)pWorkAreaHead;
		pWorkAreaHead			+= sizeIn32BitWord;
	}
	//-----------------------------------------------------------------------------------
	//�w�����v�Z�̈���Z�b�g
	//-----------------------------------------------------------------------------------
	sizeIn32BitWord		= size_in_type(sizeof(flt32_t) * maxTempWorkAreaSizeIn32BitWord, uint32_t);
	This->pTemporaryBuffer		= pWorkAreaHead;
	pWorkAreaHead				+= sizeIn32BitWord;
	//----------------------------------------------------------------------------------
	//�w�\�z
	//----------------------------------------------------------------------------------
	maxDataSize = pHeader->inHeight * pHeader->inWidth * pHeader->inChannel;
	maxTempWorkAreaSizeIn32BitWord = 0;
	DataShape_initialize(&inputShape);
	DataShape_initialize(&outputShape);
	bpEndLayerIndex = pHeader->numberOfLayers - numberOfBackPropagationLayers;
	pLayerData = pModelData + size_in_type(sizeof(SequentialNetHeader), uint32_t);
	for (i = 0; i < pHeader->numberOfLayers; i++) {
		pNeuralNetHeader = (NeuralNetHeader*)pLayerData;
		//----------------------------------------------------------------------------------
		//�w�C���^�[�t�F�[�X�擾
		//----------------------------------------------------------------------------------
		NetLayer_getInterface(pNeuralNetHeader, &netLayerFuncTable);
		//----------------------------------------------------------------------------------
		//back propagation�p�o�b�t�@�[�T�C�Y
		//----------------------------------------------------------------------------------
		if (i >= bpEndLayerIndex) {
			fLearnLayerParameters = fEnableLearning;
		}
		else {
			fLearnLayerParameters = FALSE;
		}
		//----------------------------------------------------------------------------------
		//�w���
		//----------------------------------------------------------------------------------
		tempWorkAreaSizeIn32BitWord = 0;
		fStatus = netLayerFuncTable.pGetLayerInformation(
			pLayerData,						// in:image data
			fLearnLayerParameters,			// in:back propagation flag
			&layerObjectSizeIn32BitWord,	// out:layer object size
			&numberOfLearningParameters,	// out:number of learning prameters for optimizer
			&tempWorkAreaSizeIn32BitWord,	// out:temporary work area size for prediction and back propagation
			&inputShape,					// out:input data shape
			&outputShape					// out:output data shape
		);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//----------------------------------------------------------------------------------
		//optimizer�쐬
		//----------------------------------------------------------------------------------
		if (fLearnLayerParameters == TRUE && numberOfLearningParameters > 0) {
			OptimizerFunctionTable optimizerFunctionTable;
			NeuralNetOptimizer_getInterfaceByType(optimizer, &optimizerFunctionTable);
			sizeIn32BitWord = optimizerFunctionTable.pGetSizeIn32BitWord(numberOfLearningParameters);
			hOptimizer = optimizerFunctionTable.pConstruct(numberOfLearningParameters, batchSize, pWorkAreaHead, sizeIn32BitWord);
			pWorkAreaHead += sizeIn32BitWord;
		}
		else {
			hOptimizer = NULL;
		}
		//----------------------------------------------------------------------------------
		//�w�쐬
		//----------------------------------------------------------------------------------
		hLayer = netLayerFuncTable.pConstruct(pLayerData, pWorkAreaHead, layerObjectSizeIn32BitWord, fLearnLayerParameters, hOptimizer);
		if (hLayer == NULL) {
			return FALSE;
		}
		pWorkAreaHead += layerObjectSizeIn32BitWord;
		This->pLayerArray[i] = hLayer;
		This->pOptimizerArray[i] = hOptimizer;
		//---------------------------------------------------------------------------------
		//���̑w���Ɉړ�
		//---------------------------------------------------------------------------------
		pLayerData += pNeuralNetHeader->sizeIn32BitWord;
	}
	//---------------------------------------------------------------------------------
	//���o�̓o�b�t�@�[�|�C���^
	//---------------------------------------------------------------------------------
	This->pInputBuffer	= NULL;
	This->pOutputBuffer	= NULL;
	//---------------------------------------------------------------------------------
	//�`�F�b�N
	//---------------------------------------------------------------------------------
	if( (uint32_t)(pWorkAreaHead - pWorkArea) > sizeOfWorkAreaIn32BitWord ) {
		return NULL;
	}
	return This;
}

//=====================================================================================
//  �\�����s
//=====================================================================================
bool_t
SequentialNet_predict(handle_t hModel,flt32_t* pInputData,uint32_t inputDataArraySize) {
	uint32_t				i;
	SequentialNet*			This = (SequentialNet*)hModel;
	SequentialNetHeader*	pHeader;
	uint32_t				inputDim;
	uint32_t				outputDim;
	PropagationInfo			propagationInfo;
	flt32_t*				pLayerParam;
	bool_t					fStatus;
	uint32_t				layerOrder;
	//---------------------------------------------------------------------------------
	//���f���n���h�����`�F�b�N
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���o�̓f�[�^����
	//---------------------------------------------------------------------------------
	inputDim	= DataShape_getSize(&This->inputShape);
	outputDim	= DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//�p�����[�^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (pInputData == NULL) {
		return FALSE;
	}
	if (inputDim != inputDataArraySize) {
		return FALSE;
	}
	//----------------------------------------------------------------------------------
	//�w�b�_
	//----------------------------------------------------------------------------------
	pHeader = (SequentialNetHeader*)This->pModelData;
	//---------------------------------------------------------------------------------
	//���o�̓o�b�t�@�[�|�C���^�̃Z�b�g
	//---------------------------------------------------------------------------------
	This->pInputBuffer	= This->ppDataBuffer[0];
	This->pOutputBuffer	= This->ppDataBuffer[1];
	//---------------------------------------------------------------------------------
	//���o�̓o�b�t�@�̏�����
	//---------------------------------------------------------------------------------
	for(i=0;i<inputDim;i++) {
		This->pInputBuffer[i] = pInputData[i];
	}
	for(i=0;i<outputDim;i++) {
		This->pOutputBuffer[i] = 0.0f;
	}
	//----------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//----------------------------------------------------------------------------------
	propagationInfo.dataShape.height = pHeader->inHeight;
	propagationInfo.dataShape.width = pHeader->inWidth;
	propagationInfo.dataShape.channel = pHeader->inChannel;
	propagationInfo.layerOrder = 0;
	propagationInfo.inputBufferSize = This->dataBufferSize;
	propagationInfo.outputBufferSize = This->dataBufferSize;
	propagationInfo.temporaryBufferSize = This->temporaryBufferSize;
	propagationInfo.pInputBuffer = This->pInputBuffer;
	propagationInfo.pOutputBuffer = This->pOutputBuffer;
	propagationInfo.pTemporaryBuffer = This->pTemporaryBuffer;
	//---------------------------------------------------------------------------------
	// �e�w���V�[�P���V�����Ɍv�Z
	//---------------------------------------------------------------------------------
	for( i=0; i<This->numberOfLayers; i++ ) {
		//---------------------------------------------------------------------------------
		//�o�b�t�@�ݒ�
		//---------------------------------------------------------------------------------
		propagationInfo.pInputBuffer = This->pInputBuffer;
		propagationInfo.pOutputBuffer = This->pOutputBuffer;
		//---------------------------------------------------------------------------------
		//���`��
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_forward(This->pLayerArray[i], &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//�w�v�Z�����擾:�V�[�P���V�����Ȃ̂ŏ��Ԓʂ�ł��邱�Ƃ��m�F
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_getOrder(This->pLayerArray[i],&layerOrder);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//���o�̓o�b�t�@�̓���ւ�
		//---------------------------------------------------------------------------------
		pLayerParam = This->pInputBuffer;
		This->pInputBuffer	= This->pOutputBuffer;
		This->pOutputBuffer	= pLayerParam;
	}
	//---------------------------------------------------------------------------------
	//�o�̓o�b�t�@�[�ݒ�
	//---------------------------------------------------------------------------------
	This->pOutputBuffer	= This->pInputBuffer;
	return TRUE;
}

//=====================================================================================
//  �ʏo�͌��ʎ擾
//=====================================================================================
bool_t
SequentialNet_getPrediction(handle_t hModel,uint32_t stateIndex, flt32_t* pValue){
	SequentialNet*	This = (SequentialNet*)hModel;
	uint32_t		outputDim;
	//---------------------------------------------------------------------------------
	//���f���n���h�����`�F�b�N
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^����
	//---------------------------------------------------------------------------------
	outputDim = DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//��ԃC���f�b�N�X���`�F�b�N
	//---------------------------------------------------------------------------------
	if( stateIndex >= outputDim ) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�o�̓o�b�t�@�[����яo�̓p�����^���`�F�b�N
	//---------------------------------------------------------------------------------
	if( This->pOutputBuffer == NULL ) {
		return FALSE;
	}
	if (pValue == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�o�͒l�i�\���l�j��ݒ�
	//---------------------------------------------------------------------------------
	*pValue = This->pOutputBuffer[stateIndex];
	return TRUE;
}

//=====================================================================================
// ���͌`��擾
//=====================================================================================
bool_t
SequentialNet_getInputShape(handle_t hModel, uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel) {
	SequentialNet*	This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�`��Z�b�g
	//---------------------------------------------------------------------------------
	if (pHeight != NULL) {
		*pHeight = This->inputShape.height;
	}
	if (pWidth != NULL) {
		*pWidth = This->inputShape.width;
	}
	if (pChannel != NULL) {
		*pChannel = This->inputShape.channel;
	}
	return TRUE;
}

//=====================================================================================
//�o�͎����擾
//=====================================================================================
bool_t
SequentialNet_getOutputShape(handle_t hModel,uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�`��Z�b�g
	//---------------------------------------------------------------------------------
	if (pHeight != NULL) {
		*pHeight = This->outputShape.height;
	}
	if (pWidth != NULL) {
		*pWidth = This->outputShape.width;
	}
	if (pChannel != NULL) {
		*pChannel = This->outputShape.channel;
	}
	return TRUE;
}

//=====================================================================================
//  �w�̐����擾
//=====================================================================================
bool_t
SequentialNet_getNumberOfLayers(handle_t hModel, uint32_t* pNumOfLayers) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	if (pNumOfLayers != NULL) {
		*pNumOfLayers = This->numberOfLayers;
	}
	return TRUE;
}

//=====================================================================================
//  �w�̃^�C�v���擾
//=====================================================================================
bool_t		
SequentialNet_getLayerType(handle_t hModel,uint32_t layerIndex, NetLayerType* pNetLayerType) {
	SequentialNet*		This = (SequentialNet*)hModel;
	NeuralNetHeader*	pNeuralNetHeader = NULL;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (layerIndex >= This->numberOfLayers) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�ۃv���擾
	//---------------------------------------------------------------------------------
	return NeuralNetLayer_getType(This->pLayerArray[layerIndex], pNetLayerType);
}

//=====================================================================================
//  �w�n���h�����擾
//=====================================================================================
bool_t
SequentialNet_getLayerHandle(handle_t hModel, uint32_t layerIndex, handle_t* phLayer) {
	SequentialNet*		This = (SequentialNet*)hModel;
	NeuralNetHeader*	pNeuralNetHeader = NULL;
	//---------------------------------------------------------------------------------
	//�G���[�n���h�����O
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (layerIndex >= This->numberOfLayers) {
		return FALSE;
	}
	if (phLayer == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�n���h�����Z�b�g
	//---------------------------------------------------------------------------------
	*phLayer = This->pLayerArray[layerIndex];
	return TRUE;
}

//=====================================================================================
//  ���f���p�����^������
//=====================================================================================
bool_t
SequentialNet_initializeParameter(handle_t hModel) {
	SequentialNet*	This = (SequentialNet*)hModel;
	uint32_t		i;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//���f���n���h�����`�F�b�N
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^������
	//---------------------------------------------------------------------------------
	for (i = 0; i < This->numberOfLayers; i++) {
		//---------------------------------------------------------------------------------
		//
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_initializeParameters(This->pLayerArray[i], This->hRandomValueGenerator);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//�t�`���J�E���^��������
	//---------------------------------------------------------------------------------
	This->backPropagationCounter = 0;
	return TRUE;
}

//=====================================================================================
//  �I�v�e�B�}�C�U�[���擾����
//=====================================================================================
handle_t
SequentialNet_getOptimizer(handle_t hModel, uint32_t layerIndex) {
	SequentialNet* This = (SequentialNet*)hModel;
	NeuralNetHeader* pNeuralNetHeader = NULL;
	if (This == NULL) {
		return NULL;
	}
	if (layerIndex >= This->numberOfLayers) {
		return NULL;
	}
	return This->pOptimizerArray[layerIndex];
}

//=====================================================================================
//  �덷�t�`�����ɍŏI�w��softmax���X�L�b�v���邩�ۂ��̐ݒ���s���i�f�t�H���g�̓X�L�b�v�j
//	���ޖ��̃��X�v�Z���A�isoftmax�{�N���X�G���g���s�[�j�Ƃ����ꍇ�A�i����l-�\���l�j�ƊȒP�ɂȂ邽�߁A
//	�ŏI�w��softmax�̓X�L�b�v����Ɨ^���郍�X�v�Z���ȒP�ɂȂ�B
//=====================================================================================
bool_t
SequentialNet_skipLastSoftmaxWhenBackpropagation(handle_t hModel, bool_t fSkip) {
	SequentialNet* This = (SequentialNet*)hModel;
	if (This == NULL) {
		return FALSE;
	}
	This->fSkipLastSoftmaxWhenBackpropagation = fSkip;
	return TRUE;
}

//=====================================================================================
//  �w�K�i�덷�t�`���j
//=====================================================================================
bool_t
SequentialNet_fit(handle_t hModel, flt32_t* pLoss, uint32_t arraySize) {
	uint32_t				i,j;
	SequentialNet*			This = (SequentialNet*)hModel;
	uint32_t				inputDim;
	uint32_t				outputDim;
	PropagationInfo			propagationInfo;
	flt32_t*				pNextLayerOutput;
	handle_t				hLastLayer;
	NetLayerType			layerType;
	NeuralNetActivationType activationType;
	int32_t					targetLayerOrder;
	bool_t					fStatus;
	//---------------------------------------------------------------------------------
	//���f���n���h�����`�F�b�N
	//---------------------------------------------------------------------------------
	if (This == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�K���[�h�Ŗ�����΃G���[
	//---------------------------------------------------------------------------------
	if (This->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���o�̓f�[�^����
	//---------------------------------------------------------------------------------
	inputDim	= DataShape_getSize(&This->inputShape);
	outputDim	= DataShape_getSize(&This->outputShape);
	//---------------------------------------------------------------------------------
	//�p�����[�^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (pLoss == NULL) {
		return FALSE;
	}
	if (outputDim != arraySize) {
		return FALSE;
	}
	if (This->numberOfLayers == 0) {
		return FALSE;
	}
	if (This->backPropagationBatchSize == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�o�b�`�T�C�Y�𒴂��Ă���ꍇ�͖���
	//---------------------------------------------------------------------------------
	if (This->backPropagationCounter > This->backPropagationBatchSize) {
		return TRUE;
	}
	//----------------------------------------------------------------------------------
	//�p�����^���X�V�����̂��Ɍ덷�t�`�������{������
	//----------------------------------------------------------------------------------
	This->backPropagationCounter++;
	//---------------------------------------------------------------------------------
	//���o�̓o�b�t�@�[�|�C���^�̃Z�b�g
	//---------------------------------------------------------------------------------
	This->pInputBuffer = This->ppDataBuffer[0];
	This->pOutputBuffer = This->ppDataBuffer[1];
	//---------------------------------------------------------------------------------
	//���o�̓o�b�t�@�̏�����
	//---------------------------------------------------------------------------------
	for (i = 0; i < inputDim; i++) {
		This->pInputBuffer[i] = 0.0f;
	}
	for (i = 0; i < outputDim; i++) {
		This->pOutputBuffer[i] = pLoss[i];
	}
	//----------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//----------------------------------------------------------------------------------
	propagationInfo.dataShape = This->outputShape;
	propagationInfo.layerOrder = This->numberOfLayers;
	propagationInfo.inputBufferSize = This->dataBufferSize;
	propagationInfo.outputBufferSize = This->dataBufferSize;
	propagationInfo.temporaryBufferSize = This->temporaryBufferSize;
	propagationInfo.pInputBuffer = This->pInputBuffer;
	propagationInfo.pOutputBuffer = This->pOutputBuffer;
	propagationInfo.pTemporaryBuffer = This->pTemporaryBuffer;
	//---------------------------------------------------------------------------------
	//�t�`���̊J�n�w��I��
	//layerOrder : 1�@�`�@This->numberOfLayers
	//�ŏI�w��activation�w�ŁAsoftmax�̏ꍇ�́At-y�����̑w�����݂Ƃ������X�Ƃ��ė^�����邱�Ƃ�O��Ƃ��āA�X�L�b�v����i�ȒP�̂��߁j
	//---------------------------------------------------------------------------------
	if (This->fSkipLastSoftmaxWhenBackpropagation == TRUE) {
		hLastLayer = This->pLayerArray[This->numberOfLayers - 1];	//�ŏI�w
		NeuralNetLayer_getType(hLastLayer, &layerType);	//�w�̃^�C�v���擾
		switch (layerType) {
		case NET_LAYER_ACTIVATION:
			//activation�w�̏ꍇ
			fStatus = NeuralNetLayerActivation_getType(hLastLayer, &activationType);	//activation�̃^�C�v���擾
			if (activationType == NEURAL_NET_ACTIVATION_SOFTMAX) {
				//softmax�̏ꍇ�́A�X�L�b�v����
				//�`��͕ς��Ȃ�
				if (This->numberOfLayers == 1) {
					return FALSE;
				}
				targetLayerOrder = This->numberOfLayers - 1;	// skip softmax
			}
			else {
				//����ȊO�̊������֐��͋t�`���̑ΏۂƂȂ�
				targetLayerOrder = This->numberOfLayers;
			}
			break;
		default:
			//activation�w�ȊO�̑w
			//�ŏI�w����t�`��
			targetLayerOrder = This->numberOfLayers;
			break;
		}
	}
	else {
		//�ŏI�w����t�`��
		targetLayerOrder = This->numberOfLayers;
	}
	//---------------------------------------------------------------------------------
	//�t�`��
	//---------------------------------------------------------------------------------
#if 1
	do {
		uint32_t layerIndex = targetLayerOrder - 1;
		//---------------------------------------------------------------------------------
		//�o�b�t�@�ݒ�
		//---------------------------------------------------------------------------------
		propagationInfo.pInputBuffer = This->pInputBuffer;
		propagationInfo.pOutputBuffer = This->pOutputBuffer;
		//---------------------------------------------------------------------------------
		//�e�w�̋t�`��
		//---------------------------------------------------------------------------------
		fStatus = NeuralNetLayer_backward(This->pLayerArray[layerIndex], &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
		//---------------------------------------------------------------------------------
		//���o�̓o�b�t�@�̓���ւ�
		//---------------------------------------------------------------------------------
		pNextLayerOutput = This->pInputBuffer;
		This->pInputBuffer = This->pOutputBuffer;
		This->pOutputBuffer = pNextLayerOutput;
	} while (--targetLayerOrder > 0); 
#else
	do {
		for (j = 0; j < This->numberOfLayers; j++) {
			uint32_t layerOrder;
			uint32_t layerIndex;
			//---------------------------------------------------------------------------------
			//�w�v�Z�����擾:�V�[�P���V�����Ȃ̂ŏ��Ԓʂ�
			//layerOrder : 1�@�`�@This->numberOfLayers
			//layerIndex : 0�@�`�@This->numberOfLayers-1
			//---------------------------------------------------------------------------------
			fStatus = NeuralNetLayer_getOrder(This->pLayerArray[j], &layerOrder);
			if (fStatus == FALSE) {
				return FALSE;
			}
			if (layerOrder != targetLayerOrder) {
				continue;
			}
			layerIndex = targetLayerOrder - 1;
			//---------------------------------------------------------------------------------
			//�o�b�t�@�ݒ�
			//---------------------------------------------------------------------------------
			propagationInfo.pInputBuffer = This->pInputBuffer;
			propagationInfo.pOutputBuffer = This->pOutputBuffer;
			//---------------------------------------------------------------------------------
			//�e�w�̋t�`��
			//---------------------------------------------------------------------------------
			fStatus = NeuralNetLayer_backward(This->pLayerArray[layerIndex], &propagationInfo);
			if (fStatus == FALSE) {
				return FALSE;
			}
			//---------------------------------------------------------------------------------
			//���o�̓o�b�t�@�̓���ւ�
			//---------------------------------------------------------------------------------
			pNextLayerOutput = This->pInputBuffer;
			This->pInputBuffer = This->pOutputBuffer;
			This->pOutputBuffer = pNextLayerOutput;
		}
	} while (--targetLayerOrder > 0);
#endif
	//---------------------------------------------------------------------------------
	//�덷�t�`�d�񐔂��o�b�`�T�C�Y�ɒB�����ꍇ�Ƀl�b�g���[�N�p�����^���X�V����
	//---------------------------------------------------------------------------------
	SequentialNet_update(hModel);
	return TRUE;
}

//=====================================================================================
//  ���f���w�b�_�쐬
//=====================================================================================
bool_t
SequentialNet_createHeader(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t	inHeight,
	uint32_t	inWidth,
	uint32_t	inChannel,
	uint32_t	numberOfLayers,
	uint32_t*	pSizeOfHeaderIn32BitWord) {
	uint32_t	sizeHeader;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	SequentialNetHeader* pSequentialNetHeader;
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(SequentialNetHeader), uint32_t);
	sizeLayer = sizeHeader;
	if (pSizeOfHeaderIn32BitWord != NULL) {
		*pSizeOfHeaderIn32BitWord = sizeLayer;
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
		// header
		pSequentialNetHeader = (SequentialNetHeader*)pLayer;
		pSequentialNetHeader->version			= MODEL_FILE_VERSION;
		pSequentialNetHeader->revision			= 0;
		pSequentialNetHeader->inHeight		= inHeight;
		pSequentialNetHeader->inWidth		= inWidth;
		pSequentialNetHeader->inChannel		= inChannel;
		pSequentialNetHeader->numberOfLayers	= numberOfLayers;
		pLayer += sizeHeader;
	}
	return TRUE;
}

//=====================================================================================
//  Dense�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendDense(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel,uint32_t unit,uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerDense_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, unit, pSizeOfLayerIn32BitWord);
}
	
//=====================================================================================
//  SimpleRNN�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendSimpleRNN(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t unit, NeuralNetActivationType activation, bool_t returnSequence, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerSimpleRNN_constructLayerData(pBuffer,sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel,unit, activation,returnSequence,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  Conv2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel,
	uint32_t nFilter, uint32_t	kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t	strideWidth, bool_t fPadding,
	uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerConv2D_constructLayerData(pBuffer,sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel,nFilter,kernelHeight,kernelWidth,strideHeight,strideWidth, fPadding,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  DepthwiseConv2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendDepthwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t nFilter, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t strideWidth, bool_t fPadding, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerDepthwiseConv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, nFilter, kernelHeight, kernelWidth, strideHeight, strideWidth, fPadding, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  PointwiseConv2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendPointwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t pw_nFilter, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerPointwiseConv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pw_nFilter, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  MaxPooling2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendMaxPooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t poolinghHeight, uint32_t poolingWidth, uint32_t strideHeight, uint32_t strideWidth, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerMaxPooling2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, poolinghHeight, poolingWidth, strideHeight, strideWidth, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  GlobalAveragePooling2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendGlobalAveragePooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerGlobalAveragePooling2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  BatchNormalization�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendBatchNormalization(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerBatchNormalization_constructLayerData(pBuffer,sizeOfBufferIn32BitWord,pInputHeight,pInputWidth,pInputChannel,pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  LayerNormalization�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendLayerNormalization(uint32_t * pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t * pInputHeight, uint32_t * pInputWidth, uint32_t * pInputChannel, uint32_t * pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerLayerNormalization_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  Activation�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendActivation(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, NeuralNetActivationType activation, uint32_t* pSizeOfLayerIn32BitWord) {
	return NeuralNetLayerActivation_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, activation, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  PreDeconv2D�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendPreDeconv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t strideHeight, uint32_t strideWidth, uint32_t outHeight, uint32_t outWidth, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerPreDeconv2D_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, strideHeight, strideWidth, outHeight, outWidth, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  NeuralNetLayerResidualConnectionSender�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendResidualConnectionSender(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerResidualConnectionSender_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

//=====================================================================================
//  NeuralNetLayerResidualConnectionReceiver�w�쐬
//=====================================================================================
bool_t
SequentialNet_appendResidualConnectionReceiver(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord) {
	return 	NeuralNetLayerResidualConnectionReceiver_constructLayerData(pBuffer, sizeOfBufferIn32BitWord, pInputHeight, pInputWidth, pInputChannel, pSizeOfLayerIn32BitWord);
}

