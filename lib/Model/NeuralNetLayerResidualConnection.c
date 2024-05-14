#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerResidualConnection.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#define EPSILON				(0.001f)
#define DEFAULT_MOMENTUM	(0.99f)

//=====================================================================================
//  ResidualConnectionSender�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagResidualConnectionSenderNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} ResidualConnectionSenderNeuralNetHeader;

//=====================================================================================
//  ResidualConnectionSender�w�\����
//=====================================================================================
typedef struct tagResidualConnectionSenderNeuralNetLayer {
	NeuralNetLayer						super;			//base layer class
	flt32_t*							pReceiverLoss;	//�f�[�^�o�b�t�@
	handle_t							hReceiver;		//��M��w�n���h��
	ResidualConnectionSenderInterface	funcTable;		//sender funcTable
} ResidualConnectionSenderNeuralNetLayer;

//=====================================================================================
//  ResidualConnectionReceiver�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagResidualConnectionReceiverNeuralNetHeader {
	NeuralNetHeader	super;			//base layer header
} ResidualConnectionReceiverNeuralNetHeader;

//=====================================================================================
//  ResidualConnectionReceiver�w�\����
//=====================================================================================
typedef struct tagResidualConnectionReceiverNeuralNetLayer {
	NeuralNetLayer						super;		//base layer class
	flt32_t*							pX;			//�f�[�^�o�b�t�@
	handle_t							hSender;	//���M��w�n���h��
	ResidualConnectionReceiverInterface	funcTable;	//receiver funcTable
} ResidualConnectionReceiverNeuralNetLayer;

//=====================================================================================
//=====================================================================================
// 
//  ResidualConnectionSender�w
// 
//=====================================================================================
//=====================================================================================

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
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
//  ���`��
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	flt32_t*	pLoss;
	flt32_t*	pInputData;
	flt32_t*	pOutputData;
	uint32_t	size;
	PropagationInfo	 propagationInfo;
	ResidualConnectionReceiverInterface	receiverInterface;
	bool_t	fStatus;
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//���͎����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//back propagation�p���̓f�[�^�ێ�
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//�G���[�n���h�����O
		if (pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
			return FALSE;
		}
		//-------------------------------------------------------
		//�t�`���p:���������Ă���
		//-------------------------------------------------------
		pLoss	= pResidualConnectionSenderLayer->pReceiverLoss;
		size = inputDataDim;
		while (size--) {
			*pLoss++ = 0.0f;
		}
	}
	//---------------------------------------------------------------------------------
	//��M�w�Ƀf�[�^�𑗐M
	//---------------------------------------------------------------------------------
	if (pResidualConnectionSenderLayer->hReceiver != NULL) {
		propagationInfo = *pPropagationInfo;
		propagationInfo.outputBufferSize	= 0;	//�o�̓o�b�t�@�z��T�C�Y
		propagationInfo.pOutputBuffer		= NULL;	//�o�̓o�b�t�@]
		fStatus = NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(pResidualConnectionSenderLayer->hReceiver,&receiverInterface);
		if (fStatus == FALSE) {
			return FALSE;
		}
		fStatus = receiverInterface.pSetForwardData(pResidualConnectionSenderLayer->hReceiver, hLayer, &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//�o�̓o�b�t�@�Ƀf�[�^���R�s�[
	//---------------------------------------------------------------------------------
	pInputData	= pPropagationInfo->pInputBuffer;
	pOutputData = pPropagationInfo->pOutputBuffer;
	size	= inputDataDim;
	while (size--) {
		*pOutputData++ = *pInputData++;
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
NeuralNetLayerResidualConnectionSender_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	flt32_t*	pInputData;
	flt32_t*	pLoss;
	flt32_t*	pReceiverLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	if (pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight	= pNeuralNetHeader->inHeight;
	inWidth		= pNeuralNetHeader->inWidth;
	inChannel	= pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�덷�`����o�b�t�@�[
	//---------------------------------------------------------------------------------
	size = inHeight * inWidth * inChannel;
	pInputData = pPropagationInfo->pInputBuffer;
	pLoss = pPropagationInfo->pOutputBuffer;
	pReceiverLoss = pResidualConnectionSenderLayer->pReceiverLoss;
	while (size--) {
		*pInputData++ = (*pLoss++) + (*pReceiverLoss++);
	}
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
NeuralNetLayerResidualConnectionSender_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  �X�V�p�����^�ݒ�
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_setBackwardData(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	flt32_t*	pLoss;
	flt32_t*	pReceiverLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	//---------------------------------------------------------------------------------
	//�w�K�ΏۂłȂ��ꍇ
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE ) {
		return TRUE;
	}
	//---------------------------------------------------------------------------------
	//�o�b�t�@�`�F�b�N
	//---------------------------------------------------------------------------------
	if( pResidualConnectionSenderLayer->pReceiverLoss == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//Loss���o�b�t�@�Ɋi�[����
	//---------------------------------------------------------------------------------
	size = inputDataDim;
	pLoss = pPropagationInfo->pOutputBuffer;
	pReceiverLoss = pResidualConnectionSenderLayer->pReceiverLoss;
	while (size--) {
		*pReceiverLoss++ = *pLoss++;
	}
	return TRUE;
}

//=====================================================================================
//  �X�V�p�����^�ݒ�
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_setReceiver(handle_t hLayer, handle_t hReceiver)
{
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NetLayerType	senderLayerType;
	NetLayerType	receiverLayerType;
	DataShape		senderInputShape;
	DataShape		receiverInputShape;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//�w�^�C�v�擾
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getType(hLayer, &senderLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus	= NeuralNetLayer_getType(hReceiver, &receiverLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (senderLayerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		return FALSE;
	}
	if (receiverLayerType != NET_LAYER_RESIDUAL_CONNECTION_RECEIVER) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w���͌`��擾
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getShape(hLayer, &senderInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus = NeuralNetLayer_getShape(hReceiver, &receiverInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (DataShape_equal(&senderInputShape,&receiverInputShape) == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//��M�w�Z�b�g
	//---------------------------------------------------------------------------------
	pResidualConnectionSenderLayer->hReceiver = hReceiver;
	return TRUE;
}

//=====================================================================================
//  sender �C���^�[�t�F�[�X�擾
//=====================================================================================
static
void
NeuralNetLayerResidualConnectionSender_getExtraInterface(ResidualConnectionSenderInterface* pInterface) {
	pInterface->pSetReceiver		= NeuralNetLayerResidualConnectionSender_setReceiver;
	pInterface->pSetBackwardData	= NeuralNetLayerResidualConnectionSender_setBackwardData;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionSender_getLayerInformation(
	uint32_t*	pLayerData,
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	uint32_t inputDataDim;
	if (pResidualConnectionSenderNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(ResidualConnectionSenderNeuralNetLayer), uint32_t);
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
NeuralNetLayerResidualConnectionSender_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerResidualConnectionSender_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionSenderNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	NeuralNetLayerResidualConnectionSender_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerResidualConnectionSender_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�ǉ��C���^�[�t�F�C�X�Z�b�g
		NeuralNetLayerResidualConnectionSender_getExtraInterface(&pResidualConnectionSenderLayer->funcTable);
		//receiver handle
		pResidualConnectionSenderLayer->hReceiver	= NULL;
		if (fEnableLearning == TRUE) {
			//layer
			pObjectWork += size_in_type(sizeof(ResidualConnectionSenderNeuralNetLayer), uint32_t);
			//�o�b�t�@
			inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
			//pReceiverLoss
			pResidualConnectionSenderLayer->pReceiverLoss = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
			//�o�b�t�@������
			i = inputDataDim;
			while (i--) {
				pResidualConnectionSenderLayer->pReceiverLoss[i] = 0.0f;
			}
		}
		else {
			//�o�b�t�@
			pResidualConnectionSenderLayer->pReceiverLoss = NULL;
		}
		return (handle_t)pResidualConnectionSenderLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerResidualConnectionSender_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerResidualConnectionSender_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerResidualConnectionSender_construct;
	pInterface->pGetShape = NeuralNetLayerResidualConnectionSender_getShape;
	pInterface->pForward = NeuralNetLayerResidualConnectionSender_forward;
	pInterface->pBackward = NeuralNetLayerResidualConnectionSender_backward;
	pInterface->pUpdate = NeuralNetLayerResidualConnectionSender_update;
	pInterface->pInitializeParameters = NeuralNetLayerResidualConnectionSender_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerResidualConnectionSender_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_constructLayerData(
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
	ResidualConnectionSenderNeuralNetHeader* pResidualConnectionSenderNeuralNetHeader;
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
	sizeHeader = size_in_type(sizeof(ResidualConnectionSenderNeuralNetHeader), uint32_t);
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
		//header
		pResidualConnectionSenderNeuralNetHeader = (ResidualConnectionSenderNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pResidualConnectionSenderNeuralNetHeader->super, NET_LAYER_RESIDUAL_CONNECTION_SENDER, inHeight, inWidth, inChannel, sizeLayer);
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight	= inHeight;
	*pInputWidth	= inWidth;
	*pInputChannel	= inChannel;
	return TRUE;
}

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionSender_getSenderInterface(handle_t hLayer, ResidualConnectionSenderInterface* pInterface) {
	ResidualConnectionSenderNeuralNetLayer* pResidualConnectionSenderLayer = (ResidualConnectionSenderNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionSenderLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pNeuralNetHeader->layerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		pInterface->pSetReceiver		= NULL;
		pInterface->pSetBackwardData	= NULL;
		return FALSE;
	}
	else {
		*pInterface = pResidualConnectionSenderLayer->funcTable;
	}
	return TRUE;
}

//=====================================================================================
//=====================================================================================
// 
//  ResidualConnectionReceiver�w
// 
//=====================================================================================
//=====================================================================================

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
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
//  ���`��
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_forward(handle_t hLayer, PropagationInfo* pPropagationInfo) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	flt32_t*	pInputData;
	flt32_t*	pInputDataReceived;
	flt32_t*	pOutputData;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//���͎����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�o�̓o�b�t�@�ɁA�c�Ԑڑ�����̃f�[�^�������ďo�͂���
	//---------------------------------------------------------------------------------
	pInputData = pPropagationInfo->pInputBuffer;
	pOutputData = pPropagationInfo->pOutputBuffer;
	pInputDataReceived = pResidualConnectionReceiverLayer->pX;
	size = inputDataDim;
	while (size--) {
		*pOutputData++ = (*pInputData++) + (*pInputDataReceived++);
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
NeuralNetLayerResidualConnectionReceiver_backward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	flt32_t*	pInputData;
	flt32_t*	pLoss;
	uint32_t	size;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	inputDataDim;
	ResidualConnectionSenderInterface	senderInterface;
	PropagationInfo	 propagationInfo;
	bool_t	fStatus;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = pNeuralNetHeader->inHeight;
	inWidth = pNeuralNetHeader->inWidth;
	inChannel = pNeuralNetHeader->inChannel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//���M�w�Ƀf�[�^�𑗐M
	//---------------------------------------------------------------------------------
	if (pResidualConnectionReceiverLayer->hSender != NULL) {
		propagationInfo = *pPropagationInfo;
		propagationInfo.inputBufferSize = 0;	//���̓o�b�t�@�z��T�C�Y
		propagationInfo.pInputBuffer = NULL;	//���̓o�b�t�@
		fStatus = NeuralNetLayerResidualConnectionSender_getSenderInterface(pResidualConnectionReceiverLayer->hSender, &senderInterface);
		if (fStatus == FALSE) {
			return FALSE;
		}
		fStatus = senderInterface.pSetBackwardData(pResidualConnectionReceiverLayer->hSender, &propagationInfo);
		if (fStatus == FALSE) {
			return FALSE;
		}
	}
	//---------------------------------------------------------------------------------
	//loss���R�s�[
	//---------------------------------------------------------------------------------
	pInputData	= pPropagationInfo->pInputBuffer;
	pLoss	= pPropagationInfo->pOutputBuffer;
	size = inputDataDim;
	while (size--) {
		*pInputData++ = *pLoss++;
	}
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
NeuralNetLayerResidualConnectionReceiver_update(handle_t hLayer) {
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	return TRUE;
}

//=====================================================================================
//  ResidualConnectionReceiver�w
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_setForwardData(handle_t hLayer, handle_t hSender, PropagationInfo* pPropagationInfo)
{
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer*	pLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;			//base layer class
	NetLayerType	senderLayerType;
	DataShape		senderInputShape;
	DataShape		receiverInputShape;
	uint32_t		size;
	uint32_t		inHeight;
	uint32_t		inWidth;
	uint32_t		inChannel;
	uint32_t		inputDataDim;
	flt32_t*		pInputData;
	flt32_t*		pX;
	bool_t			fStatus;
	//---------------------------------------------------------------------------------
	//�v�Z����
	//---------------------------------------------------------------------------------
	pPropagationInfo->layerOrder++;
	if (pLayer->layerOrder < pPropagationInfo->layerOrder) {
		pLayer->layerOrder = pPropagationInfo->layerOrder;
	}
	//---------------------------------------------------------------------------------
	//�w�^�C�v�擾
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getType(hSender, &senderLayerType);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (senderLayerType != NET_LAYER_RESIDUAL_CONNECTION_SENDER) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w���͌`��擾
	//---------------------------------------------------------------------------------
	fStatus = NeuralNetLayer_getShape(hSender, &senderInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	fStatus = NeuralNetLayer_getShape(hLayer, &receiverInputShape, NULL);
	if (fStatus == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (DataShape_equal(&senderInputShape, &receiverInputShape) == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//
	//---------------------------------------------------------------------------------
	pResidualConnectionReceiverLayer->hSender = hSender;
	//---------------------------------------------------------------------------------
	//���͎���
	//---------------------------------------------------------------------------------
	inHeight = receiverInputShape.height;
	inWidth = receiverInputShape.width;
	inChannel = receiverInputShape.channel;
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�����`�F�b�N
	//---------------------------------------------------------------------------------
	if ((pPropagationInfo->dataShape.height != inHeight) ||
		(pPropagationInfo->dataShape.width != inWidth) ||
		(pPropagationInfo->dataShape.channel != inChannel)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�덷�`����o�b�t�@�[
	//---------------------------------------------------------------------------------
	size = inputDataDim;
	pInputData = pPropagationInfo->pInputBuffer;
	pX = pResidualConnectionReceiverLayer->pX;
	while (size--) {
		*pX++ = *pInputData++;
	}
	return TRUE;
}

//=====================================================================================
//  receiver �C���^�[�t�F�[�X�擾
//=====================================================================================
static
void
NeuralNetLayerResidualConnectionReceiver_getExtraInterface(ResidualConnectionReceiverInterface* pInterface) {
	pInterface->pSetForwardData = NeuralNetLayerResidualConnectionReceiver_setForwardData;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerResidualConnectionReceiver_getLayerInformation(
	uint32_t* pLayerData,
	bool_t		fEnableLearning,
	uint32_t* pLayerObjectSizeIn32BitWord,
	uint32_t* pNumberOfLearningParameters,
	uint32_t* pTempWorkAreaSizeIn32BitWord,
	DataShape* pInputShape,
	DataShape* pOutputShape) {
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	uint32_t inputDataDim;
	if (pResidualConnectionReceiverNeuralNetHeader == NULL) {
		return FALSE;
	}
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
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(ResidualConnectionReceiverNeuralNetLayer), uint32_t);
		//X
		inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
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
NeuralNetLayerResidualConnectionReceiver_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
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
NeuralNetLayerResidualConnectionReceiver_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	uint32_t i;
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pResidualConnectionReceiverNeuralNetHeader;
	LayerFuncTable funcTable;
	uint32_t requiredSize = 0;
	uint32_t numberOfLearningParameters = 0;
	uint32_t inputDataDim;
	NeuralNetLayerResidualConnectionReceiver_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerResidualConnectionReceiver_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�ǉ��C���^�[�t�F�C�X�Z�b�g
		NeuralNetLayerResidualConnectionReceiver_getExtraInterface(&pResidualConnectionReceiverLayer->funcTable);
		//layer
		pObjectWork += size_in_type(sizeof(ResidualConnectionReceiverNeuralNetLayer), uint32_t);
		//�o�b�t�@
		inputDataDim = pNeuralNetHeader->inHeight * pNeuralNetHeader->inWidth * pNeuralNetHeader->inChannel;
		//pX
		pResidualConnectionReceiverLayer->pX = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
		//sender
		pResidualConnectionReceiverLayer->hSender = NULL;
		//�o�b�t�@������
		i = inputDataDim;
		while (i--) {
			pResidualConnectionReceiverLayer->pX[i] = 0.0f;
		}
		return (handle_t)pResidualConnectionReceiverLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerResidualConnectionReceiver_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerResidualConnectionReceiver_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerResidualConnectionReceiver_construct;
	pInterface->pGetShape = NeuralNetLayerResidualConnectionReceiver_getShape;
	pInterface->pForward = NeuralNetLayerResidualConnectionReceiver_forward;
	pInterface->pBackward = NeuralNetLayerResidualConnectionReceiver_backward;
	pInterface->pUpdate = NeuralNetLayerResidualConnectionReceiver_update;
	pInterface->pInitializeParameters = NeuralNetLayerResidualConnectionReceiver_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerResidualConnectionReceiver_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_constructLayerData(
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
	uint32_t	inputDataDim;
	ResidualConnectionReceiverNeuralNetHeader* pResidualConnectionReceiverNeuralNetHeader;
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
	inputDataDim = inHeight * inWidth * inChannel;
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(ResidualConnectionReceiverNeuralNetHeader), uint32_t);
	//sizeOfBuffer = size_in_type(sizeof(flt32_t) * inputDataDim, uint32_t);
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
		//header
		pResidualConnectionReceiverNeuralNetHeader = (ResidualConnectionReceiverNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pResidualConnectionReceiverNeuralNetHeader->super, NET_LAYER_RESIDUAL_CONNECTION_RECEIVER, inHeight, inWidth, inChannel, sizeLayer);
		pLayer += sizeHeader;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	*pInputHeight = inHeight;
	*pInputWidth = inWidth;
	*pInputChannel = inChannel;
	return TRUE;
}

//=====================================================================================
//  �ǉ��C���^�[�t�F�C�X
//=====================================================================================
bool_t
NeuralNetLayerResidualConnectionReceiver_getReceiverInterface(handle_t hLayer, ResidualConnectionReceiverInterface* pInterface) {
	ResidualConnectionReceiverNeuralNetLayer* pResidualConnectionReceiverLayer = (ResidualConnectionReceiverNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pResidualConnectionReceiverLayer;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pNeuralNetHeader->layerType != NET_LAYER_RESIDUAL_CONNECTION_RECEIVER) {
		pInterface->pSetForwardData = NULL;
		return FALSE;
	}
	else {
		*pInterface = pResidualConnectionReceiverLayer->funcTable;
	}
	return TRUE;
}
