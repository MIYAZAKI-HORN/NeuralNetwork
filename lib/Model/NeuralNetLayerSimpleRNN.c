#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetLayerSimpleRNN.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

//=====================================================================================
//  SimpleRNN�w�u���b�N���w�b�_�[
//=====================================================================================
typedef struct tagSimpleRNNNeuralNetHeader {
	NeuralNetHeader			super;			//base layer header
	uint32_t				unit;			//���j�b�g��
	NeuralNetActivationType	activation;		//�������֐�
	bool_t					returnSequence;	//���^�[���V�[�P���X�@TRUE�F�S���ԁ@FALSE�F�ŏI����
} SimpleRNNNeuralNetHeader;

//=====================================================================================
//  SimpleRNN�w�\����
//=====================================================================================
typedef struct tagSimpleRNNNeuralNetLayer {
	NeuralNetLayer	super;					//base layer class
	flt32_t*		pX;						//�덷�t�`���p�f�[�^�o�b�t�@�FW�ւ̍ŏI���́@timeSteps�̓��͒l
	flt32_t*		pH;						//�덷�t�`���p�f�[�^�o�b�t�@�F�o�͒l�@timeSteps���̏o�͒l
	flt32_t*		pBPData;				//�덷�t�`���p�f�[�^�o�b�t�@�F�����������֐��ւ̍ŏI���͂܂��͓����������֐�����̍ŏI�o�́@timeSteps���̏o�͒l
	flt32_t*		pTimeLossBuffer;		//�o�̓T�C�Y�̋t�`�d��Ɨp�o�b�t�@�[�F1���ԕ��̏o�͒l
	uint32_t		maxPropagationTime;		//�ő�덷�`�d����
	handle_t		hOptimizer;				//�I�v�e�B�}�C�U�[�n���h��
} SimpleRNNNeuralNetLayer;

//=====================================================================================
//  �`��擾
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	//---------------------------------------------------------------------------------
	//���̓f�[�^�`��
	//---------------------------------------------------------------------------------
	pInputShape->height = pNeuralNetHeader->inHeight;
	pInputShape->width = pNeuralNetHeader->inWidth;
	pInputShape->channel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//�o�̓f�[�^�`��
	//---------------------------------------------------------------------------------
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		pOutputShape->height = timeSteps;
	}
	else {
		pOutputShape->height = 1;
	}
	pOutputShape->width = pSimpleRNNNeuralNetHeader->unit;
	pOutputShape->channel = 1;
	return TRUE;
}

//=====================================================================================
//  ���`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_forward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps	= pNeuralNetHeader->inHeight;
	uint32_t	nInputDim	= pNeuralNetHeader->inWidth;
	uint32_t	i;
	uint32_t	t;
	flt32_t*	pW;
	flt32_t*	pU;
	flt32_t*	pB;
	uint32_t*	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	flt32_t*	pTimeIn;
	flt32_t*	pTimeOut;
	uint32_t	outputBufferSize;
	flt32_t*	pInternalOutputBuffer;
	flt32_t*	pUnitOut;
	flt32_t*	pInput;
	flt32_t*	pX;
	flt32_t*	pH;
	uint32_t	size;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//�o�̓o�b�t�@�[��������
	//---------------------------------------------------------------------------------
	outputBufferSize = timeSteps * pSimpleRNNNeuralNetHeader->unit;
	pInternalOutputBuffer = (flt32_t*)pPropagationInfo->pTemporaryBuffer;
	pUnitOut = pInternalOutputBuffer;
	i = outputBufferSize;
	while (i--) {
		*pUnitOut++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//timeSteps���v�Z
	//---------------------------------------------------------------------------------
	pTimeIn = pPropagationInfo->pInputBuffer;
	for (t = 0; t < timeSteps; t++) {
		//------------------------------------------------------
		// �덷�t�`���FW�u���b�N�ւ̓���
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE) {
			pInput = pTimeIn;
			pX = pSimpleRNNLayer->pX + t * nInputDim;
			size = nInputDim;
			while (size--) {
				*pX++ = *pInput++;
			}
		}
		//------------------------------------------------------
		//t�t���[���ɏo�͂���
		//------------------------------------------------------
		pTimeOut = pInternalOutputBuffer + t * pSimpleRNNNeuralNetHeader->unit;
		//------------------------------------------------------
		//W�Ein[t]�{B
		//------------------------------------------------------
		weight_matrix_with_bias_forward(pTimeIn, nInputDim, pW, pB, pTimeOut, pSimpleRNNNeuralNetHeader->unit,FALSE);
		//------------------------------------------------------
		//U�Eh[t-1]
		//------------------------------------------------------
		if (t > 0) {
			//------------------------------------------------------
			//�O�̎����̏o�͒l h(t-1) ����͂Ƃ���
			//------------------------------------------------------
			pUnitOut = pInternalOutputBuffer + (t - 1) * pSimpleRNNNeuralNetHeader->unit;
			//------------------------------------------------------
			//pTimeOut�̒l�Ɂit-1�j�o�͂�U�ɑ΂���d�݌v�Z�l�����Z�����
			//------------------------------------------------------
			weight_matrix_with_bias_forward(pUnitOut, pSimpleRNNNeuralNetHeader->unit, pU,NULL, pTimeOut, pSimpleRNNNeuralNetHeader->unit,TRUE);
		}
		//------------------------------------------------------
		//�덷�t�`���v�Z�Ŋ������֐��ւ̓��͂𗘗p����ꍇ:relu
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE ) {
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				pInput	= pTimeOut;
				pX		= pSimpleRNNLayer->pBPData + t * pSimpleRNNNeuralNetHeader->unit;
				size	= pSimpleRNNNeuralNetHeader->unit;
				while (size--) {
					*pX++ = *pInput++;
				}
				break;
			default:
				break;
			}
		}
		//------------------------------------------------------
		//�������֐�
		//------------------------------------------------------
		switch (pSimpleRNNNeuralNetHeader->activation) {
		case NEURAL_NET_ACTIVATION_RELU:
			relu_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit,0.0f);
			break;
		case NEURAL_NET_ACTIVATION_TANH:
			tanh_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		case NEURAL_NET_ACTIVATION_SIGMOID:
			sigmoid_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		case NEURAL_NET_ACTIVATION_SOFTMAX:
			softmax_forward(pTimeOut, pTimeOut, pSimpleRNNNeuralNetHeader->unit);
			break;
		default:
			return FALSE;
		}
		//------------------------------------------------------
		//�덷�t�`���v�Z�Ŋ������֐��ւ̏o�͒l�𗘗p����ꍇ�Fhyperbolic_tangent,sigmoid
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE ) {
			//�o�͒l�𗘗p����ꍇ
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_TANH:
			case NEURAL_NET_ACTIVATION_SIGMOID:
			case NEURAL_NET_ACTIVATION_SOFTMAX:
				pInput	= pTimeOut;
				pX		= pSimpleRNNLayer->pBPData + t * pSimpleRNNNeuralNetHeader->unit;
				size	= pSimpleRNNNeuralNetHeader->unit;
				while (size--) {
					*pX++ = *pInput++;
				}
				break;
			default:
				break;
			}
		}
		//------------------------------------------------------
		//�덷�t�`���FU�ւ̓��́@h(t)
		//------------------------------------------------------
		if (pNeuralNetLayer->fEnableLearning == TRUE) {
			pInput	= pTimeOut;
			pH = pSimpleRNNLayer->pH + t * pSimpleRNNNeuralNetHeader->unit;
			size = pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pH++ = *pInput++;
			}
		}
		//------------------------------------------------------
		//���̃t���[���Ɉړ�
		//------------------------------------------------------
		pTimeIn += nInputDim;
	}
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		//---------------------------------------------------------------------------------
		//�S�t���[���̃f�[�^���o�̓o�b�t�@�ɃR�s�[
		//---------------------------------------------------------------------------------
		pTimeOut = pPropagationInfo->pOutputBuffer;
		pUnitOut = pInternalOutputBuffer;
		i = pSimpleRNNNeuralNetHeader->unit * timeSteps;
		while (i--) {
			*pTimeOut++ = *pUnitOut++;
		}
		//---------------------------------------------------------------------------------
		//timeSteps���̃f�[�^���o��
		//---------------------------------------------------------------------------------
		DataShape_construct(&pPropagationInfo->dataShape, timeSteps, pSimpleRNNNeuralNetHeader->unit, 1);
	}
	else {
		//---------------------------------------------------------------------------------
		//�ŏI�i�ŐV�j�t���[���̃f�[�^���o�̓o�b�t�@�̐擪�ɃR�s�[
		//---------------------------------------------------------------------------------
		if (timeSteps > 0) {
			pTimeOut = pPropagationInfo->pOutputBuffer;
			pUnitOut = pInternalOutputBuffer + (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			i = pSimpleRNNNeuralNetHeader->unit;
			while (i--) {
				*pTimeOut++ = *pUnitOut++;
			}
		}
		//---------------------------------------------------------------------------------
		//�o�̓f�[�^�T�C�Y�`��
		//---------------------------------------------------------------------------------
		DataShape_construct(&pPropagationInfo->dataShape, 1, pSimpleRNNNeuralNetHeader->unit, 1);
	}
	return TRUE;
}

//=====================================================================================
//  �t�`���v�Z
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	flt32_t*	pW;
	flt32_t*	pU;
	uint32_t*	pLayerParam;
	flt32_t*	pDW;
	flt32_t*	pDB;
	flt32_t*	pDU;
	uint32_t	size;
	flt32_t*	pInternalOutputBuffer;
	flt32_t*	pX;
	flt32_t*	pOutput;
	flt32_t*	pBPDataT;
	flt32_t*	pH;
	flt32_t*	pInputWX;
	flt32_t*	pTimeLossBuffer;
	int32_t		t;
	int32_t		frameTime;
	flt32_t*	pInputTime;
	flt32_t*	pOutputTime;
	int32_t		lastTime;
	OptimizerFunctionTable optimizerInterface;
	//---------------------------------------------------------------------------------
	//�t�`���ΏۂłȂ��ꍇ�̓G���[
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�덷�`���o�b�t�@�[������
	//---------------------------------------------------------------------------------
	size = nInputDim * timeSteps;
	pX = pPropagationInfo->pInputBuffer;
	while (size--) {
		*pX++ = 0.0f;
	}
	//---------------------------------------------------------------------------------
	//�����l�o�b�t�@�|�C���^
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pSimpleRNNLayer->hOptimizer, &optimizerInterface);
	pDW = optimizerInterface.pGetDeltaParameterBuffer(pSimpleRNNLayer->hOptimizer);
	pDU = pDW + nInputDim * timeSteps;
	pDB = pDU + pSimpleRNNNeuralNetHeader->unit * timeSteps;
	//---------------------------------------------------------------------------------
	//returnSequence��FALSE�̏ꍇ�́A�擪�t���[���̃f�[�^�i�����l�j�����Ƃ̍ŏI�t���[���ɖ߂��Ă���
	//---------------------------------------------------------------------------------
	pInternalOutputBuffer = (flt32_t*)pPropagationInfo->pTemporaryBuffer;
	if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
		//�o�̓o�b�t�@�̌덷�f�[�^���ꎞ�o�b�t�@�ɂ��ׂăR�s�[����
		pOutput = pInternalOutputBuffer;
		pX = pPropagationInfo->pOutputBuffer;
		size = pSimpleRNNNeuralNetHeader->unit * timeSteps;
		while (size--) {
			*pOutput++ = *pX++;
		}
	}
	else{
		if (timeSteps > 0) {
			//�擪�t���[�������Ɍ덷���`�����Ă���̂ŁA���̌덷���ŏI�t���[���ɃR�s�[����
			pOutput = pInternalOutputBuffer + (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			pX = pPropagationInfo->pOutputBuffer;
			size = pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pOutput++ = *pX++;
			}
			//���̑��̎��ԓ`�d�͗��p����Ă��Ȃ��̂ł�0�ɂ���
			pOutput = pInternalOutputBuffer;
			size = (timeSteps - 1) * pSimpleRNNNeuralNetHeader->unit;
			while (size--) {
				*pOutput++ = 0.0f;
			}
		}
	}
	//---------------------------------------------------------------------------------
	//�t�`�d
	//---------------------------------------------------------------------------------
	for (t = timeSteps - 1; t >= 0; t--) {
		//---------------------------------------------------------------------------------
		//  �덷�`�d�ő厞�Ԑ�
		//---------------------------------------------------------------------------------
		if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
			lastTime = t - pSimpleRNNLayer->maxPropagationTime;
			if (lastTime < 0) {
				lastTime = 0;
			}
		}
		else {
			lastTime = 0;
		}
		//---------------------------------------------------------------------------------
		//�덷�`���o�b�t�@�[
		//���̎��Ԃ̃f�[�^�͍��㗘�p���Ȃ����߁A���ԓ`���p�̃o�b�t�@�Ƃ��Ĉꎞ�o�b�t�@�Ƃ��čė��p����邱�Ƃɒ���
		//---------------------------------------------------------------------------------
		pOutputTime = pInternalOutputBuffer + t * pSimpleRNNNeuralNetHeader->unit;
		//---------------------------------------------------------------------------------
		// �w�肳�ꂽ���Ԑ������덷�`�d�����{����
		//---------------------------------------------------------------------------------
		for (frameTime = t; frameTime >= lastTime; frameTime--) {
			//---------------------------------------------------------------------------------
			//�����������֐��ɑ΂���t�`�d
			//---------------------------------------------------------------------------------
			pBPDataT = pSimpleRNNLayer->pBPData + frameTime * pSimpleRNNNeuralNetHeader->unit;
			size = pSimpleRNNNeuralNetHeader->unit;
			switch (pSimpleRNNNeuralNetHeader->activation) {
			case NEURAL_NET_ACTIVATION_RELU:
				relu_backword(pBPDataT, pOutputTime, pOutputTime, size,0.0f);
				break;
			case NEURAL_NET_ACTIVATION_TANH:
				tanh_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			case NEURAL_NET_ACTIVATION_SIGMOID:
				sigmoid_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			case NEURAL_NET_ACTIVATION_SOFTMAX:
				softmax_backword(pBPDataT, pOutputTime, pOutputTime, size);
				break;
			default:
				break;
			}
			//---------------------------------------------------------------------------------
			//W�u���b�N�ɑ΂���t�`�d�FDense�w�Ɠ���
			//�덷��ێ����Ă���o�b�t�@���A�����������֐�����̏o�͂ł���_�Əo�͐�̃o�b�t�@�����ԂňقȂ�
			//---------------------------------------------------------------------------------
			pInputTime = pPropagationInfo->pInputBuffer + frameTime * nInputDim;
			pInputWX = pSimpleRNNLayer->pX + frameTime * nInputDim;
			weight_matrix_with_bias_backward(
				pInputTime,							//�덷�o�́i���`�������́j
				nInputDim,							//�덷�o�͎����i���`�������͎����j
				pW,									//W�d��
				pOutputTime,						//�����������֐�����o�͂����덷��ێ�����o�b�t�@�̈�
				pSimpleRNNNeuralNetHeader->unit,	//�o�͎���
				pInputWX,							//W�ɓ��͂����f�[�^�̃o�b�t�@
				pDW,								//W�����l
				pDB									//B�����l
			);
			//---------------------------------------------------------------------------------
			//�����������֐�����̌덷�o�͂�U�u���b�N�Ɍ덷�t�`�d
			//---------------------------------------------------------------------------------
			if (frameTime > 0) {
				//---------------------------------------------------------------------------------
				//�o�͍�ƃo�b�t�@��������
				//---------------------------------------------------------------------------------
				size = pSimpleRNNNeuralNetHeader->unit;
				pTimeLossBuffer = pSimpleRNNLayer->pTimeLossBuffer;
				while (size--) {
					*pTimeLossBuffer++ = 0.0f;
				}
				//---------------------------------------------------------------------------------
				//U�u���b�N�t�`�d U�u���b�N���́Fh(t-1)�@�t�`�d
				//---------------------------------------------------------------------------------
				pH = pSimpleRNNLayer->pH + (frameTime - 1) * pSimpleRNNNeuralNetHeader->unit;
				weight_matrix_with_bias_backward(
					pSimpleRNNLayer->pTimeLossBuffer,	//U�덷�o�́i���`����U���́j
					pSimpleRNNNeuralNetHeader->unit,		//�덷�o�͎����i���`����U���͎����j
					pU,									//U�d��
					pOutputTime,						//�����������֐�����o�͂����덷��ێ�����o�b�t�@�̈�
					pSimpleRNNNeuralNetHeader->unit,		//U�o�͎���
					pH,									//U�ɓ��͂����f�[�^�̃o�b�t�@�it-1�j
					pDU,								//U�����l
					NULL								//�o�C�A�X�͖���
				);
				//---------------------------------------------------------------------------------
				//U�u���b�N����o�͂����덷�Ō덷�`�d�o�b�t�@��u��������
				//pOutputTime��t�����̃f�[�^�o�b�t�@�ł��邪�A���̌㗘�p����Ȃ����߁A
				//���̎����i�O�̎����j�̋t�`���p�̃f�[�^�o�b�t�@�Ƃ��čė��p����Ă���
				//---------------------------------------------------------------------------------
				size = pSimpleRNNNeuralNetHeader->unit;
				pOutput = pOutputTime;
				pTimeLossBuffer = pSimpleRNNLayer->pTimeLossBuffer;
				while (size--) {
					*pOutput++ = *pTimeLossBuffer++;
				}
			}
		}
		//---------------------------------------------------------------------------------
		//���`���ł���ȏ�̏o�͂��Ȃ�
		//---------------------------------------------------------------------------------
		if (pSimpleRNNNeuralNetHeader->returnSequence == FALSE) {
			break;
		}
	}
	//---------------------------------------------------------------------------------
	//���͕����f�[�^�T�C�Y�`��
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, timeSteps, nInputDim, 1);
	return TRUE;
}

//=====================================================================================
//  �p�����^�X�V
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_update(handle_t hLayer) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	uint32_t*	pLayerParam;
	OptimizerFunctionTable	optimizerInterface;
	NeuralNetOptimizer_getInterface(pSimpleRNNLayer->hOptimizer, &optimizerInterface);
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pParameter = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//�w�p�����^�X�V
	//---------------------------------------------------------------------------------
	//�@W&U&B
	optimizerInterface.pUpdate(pSimpleRNNLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  �p�����^������
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	flt32_t*	pW;
	flt32_t*	pU;
	flt32_t*	pB;
	uint32_t*	pLayerParam;
	uint32_t	paramSize;
	uint32_t	normSize;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSimpleRNNNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	pW = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * nInputDim, uint32_t);
	pU = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	pB = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);
	//---------------------------------------------------------------------------------
	//�w�p�����^������
	//---------------------------------------------------------------------------------
	//W
	paramSize = pSimpleRNNNeuralNetHeader->unit * nInputDim;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pW, paramSize, normSize);
	//U
	paramSize = pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit;
	normSize = paramSize;
	set_random_initial_values_by_sqrt(hRandomValueGenerator, pU, paramSize, normSize);
	//B
	paramSize = pSimpleRNNNeuralNetHeader->unit;
	set_constant_initial_values(pB, paramSize, 0.0f);
	return TRUE;
}

//=====================================================================================
//  �ő�덷�`�d���Ԃ̐ݒ�
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_setMaxPropagationTime(handle_t hLayer,uint32_t maxPropagationTime)
{
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	layerType	= pNeuralNetHeader->layerType;
	uint32_t	timeSteps	= pNeuralNetHeader->inHeight;
	uint32_t	nInputDim	= pNeuralNetHeader->inWidth;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (layerType != NET_LAYER_SIMPLE_RNN) {
		return FALSE;
	}
	if (timeSteps == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�ő�덷�`�d���� 0�`timeSteps-1
	//---------------------------------------------------------------------------------
	if (maxPropagationTime < timeSteps) {	//�ő�덷�`�d����t=t-1�����܂�
		pSimpleRNNLayer->maxPropagationTime = maxPropagationTime;
	}
	else {
		pSimpleRNNLayer->maxPropagationTime = timeSteps - 1;
	}
	return TRUE;
}

//=====================================================================================
//  �w���擾
//=====================================================================================
static
bool_t
NeuralNetLayerSimpleRNN_getLayerInformation(
	uint32_t*	pLayerData,						// in:image data
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	if (pSimpleRNNNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�w�K�p�����^��
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters = pSimpleRNNNeuralNetHeader->unit * nInputDim;	// W
		*pNumberOfLearningParameters += pSimpleRNNNeuralNetHeader->unit * pSimpleRNNNeuralNetHeader->unit;	// U
		*pNumberOfLearningParameters += pSimpleRNNNeuralNetHeader->unit;	// B
	}
	//---------------------------------------------------------------------------------
	//�I�u�W�F�N�g�T�C�Y&���̓f�[�^
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(SimpleRNNNeuralNetLayer), uint32_t);
		if (fEnableLearning == TRUE) {
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * nInputDim * timeSteps, uint32_t);						//�@W�ւ̓���
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);	//�@U�ւ̓���
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);	//�@�����������֐��ւ̓��͂܂��͏o�́F�����������֐��ɂ���ė��p�����l���قȂ�
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);				//	�o�̓T�C�Y�̂̋t�`�d�o�b�t�@�[
		}
	}
	//---------------------------------------------------------------------------------
	//�w�����̌v�Z�o�b�t�@�[�T�C�Y
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		//�������ԓ`�������p
		*pTempWorkAreaSizeIn32BitWord = pSimpleRNNNeuralNetHeader->unit * timeSteps;
	}
	//---------------------------------------------------------------------------------
	//�o�͌`��
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		if (pSimpleRNNNeuralNetHeader->returnSequence == TRUE) {
			pOutputShape->height = timeSteps;
		}
		else {
			pOutputShape->height = 1;
		}
		pOutputShape->width = pSimpleRNNNeuralNetHeader->unit;
		pOutputShape->channel = 1;
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
NeuralNetLayerSimpleRNN_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//�w�p�����^
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerSimpleRNN_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  �w�\�z
//=====================================================================================
static
handle_t
NeuralNetLayerSimpleRNN_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	SimpleRNNNeuralNetLayer* pSimpleRNNLayer = (SimpleRNNNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSimpleRNNLayer;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSimpleRNNNeuralNetHeader;
	LayerFuncTable	funcTable;
	uint32_t	timeSteps = pNeuralNetHeader->inHeight;
	uint32_t	nInputDim = pNeuralNetHeader->inWidth;
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerSimpleRNN_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	else {
		//�w�C���^�[�t�F�C�X�擾
		NeuralNetLayerSimpleRNN_getInterface(&funcTable);
		//�w�\���̍\�z
		NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
		//�w�K�p�f�[�^����эœK���A���S���Y���I�u�W�F�N�g�n���h��
		if (fEnableLearning == TRUE) {
			// �w�K�p�����^�T�C�Y�`�F�b�N
			OptimizerFunctionTable	optimizerInterface;
			NeuralNetOptimizer_getInterface(hOptimizer, &optimizerInterface);
			parameterSize = optimizerInterface.pGetParameterSize(hOptimizer);
			if (parameterSize < numberOfLearningParameters) {
				return NULL;
			}
			//layer
			pObjectWork += size_in_type(sizeof(SimpleRNNNeuralNetLayer), uint32_t);
			//data
			pSimpleRNNLayer->pX = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * nInputDim * timeSteps, uint32_t);								//�@W�ւ̓���;
			pSimpleRNNLayer->pH = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);		//�@U�ւ̓���
			pSimpleRNNLayer->pBPData = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit * timeSteps, uint32_t);		//�@�����������֐��ւ̓��͂܂��͏o�́F�����������֐��ɂ���ė��p�����l���قȂ�
			pSimpleRNNLayer->pTimeLossBuffer = (flt32_t*)pObjectWork;
			pObjectWork += size_in_type(sizeof(flt32_t) * pSimpleRNNNeuralNetHeader->unit, uint32_t);					//	�o�̓T�C�Y�̂̋t�`�d�o�b�t�@�[;
			pSimpleRNNLayer->hOptimizer = hOptimizer;
		}
		else {
			pSimpleRNNLayer->pX = NULL;
			pSimpleRNNLayer->hOptimizer = NULL;
		}
		return (handle_t)pSimpleRNNLayer;
	}
}

//=====================================================================================
//  �C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerSimpleRNN_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerSimpleRNN_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerSimpleRNN_construct;
	pInterface->pGetShape = NeuralNetLayerSimpleRNN_getShape;
	pInterface->pForward = NeuralNetLayerSimpleRNN_forward;
	pInterface->pBackward = NeuralNetLayerSimpleRNN_backward;
	pInterface->pUpdate = NeuralNetLayerSimpleRNN_update;
	pInterface->pInitializeParameters = NeuralNetLayerSimpleRNN_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerSimpleRNN_getParameters;
}

//=====================================================================================
//  �w�쐬
//=====================================================================================
bool_t
NeuralNetLayerSimpleRNN_constructLayerData(
	uint32_t*	pBuffer, 
	uint32_t	sizeOfBufferIn32BitWord, 
	uint32_t*	pInputHeight,	// time steps
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	unit,
	NeuralNetActivationType activation,
	bool_t		returnSequence,
	uint32_t*	pSizeOfLayerIn32BitWord) 
{
	uint32_t	sizeHeader;
	uint32_t	sizeOfParamW;
	uint32_t	sizeOfParamU;
	uint32_t	sizeOfParamB;
	uint32_t	sizeLayer;
	uint32_t*	pLayer;
	uint32_t	inHeight;
	uint32_t	inWidth;
	uint32_t	inChannel;
	uint32_t	timeSteps;
	SimpleRNNNeuralNetHeader* pSimpleRNNNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//�p�����^�`�F�b�N
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	if (*pInputChannel != 1) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//�p�����^
	//---------------------------------------------------------------------------------
	inHeight	= *pInputHeight;
	inWidth		= *pInputWidth;
	inChannel	= *pInputChannel;
	timeSteps	= inHeight;
	//---------------------------------------------------------------------------------
	//�w�T�C�Y
	//---------------------------------------------------------------------------------
	sizeHeader = size_in_type(sizeof(SimpleRNNNeuralNetHeader), uint32_t);
	sizeOfParamW = size_in_type(sizeof(flt32_t) * unit * inWidth, uint32_t);
	sizeOfParamU = size_in_type(sizeof(flt32_t) * unit * unit, uint32_t);
	sizeOfParamB = size_in_type(sizeof(flt32_t) * unit, uint32_t);
	sizeLayer = sizeHeader + sizeOfParamW + sizeOfParamU + sizeOfParamB;
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
		pSimpleRNNNeuralNetHeader = (SimpleRNNNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pSimpleRNNNeuralNetHeader->super, NET_LAYER_SIMPLE_RNN, inHeight, inWidth, inChannel, sizeLayer);
		pSimpleRNNNeuralNetHeader->returnSequence = returnSequence;
		pSimpleRNNNeuralNetHeader->activation = activation;
		pSimpleRNNNeuralNetHeader->unit = unit;
		pLayer += sizeHeader;
		//W
		pLayer += sizeOfParamW;
		//U
		pLayer += sizeOfParamU;
		//B
		pLayer += sizeOfParamB;
	}
	//---------------------------------------------------------------------------------
	//�o�͎���
	//---------------------------------------------------------------------------------
	if (returnSequence == TRUE) {
		*pInputHeight = timeSteps;
	}
	else {
		*pInputHeight = 1;
	}
	*pInputWidth	= unit;
	*pInputChannel	= 1;
	return TRUE;
}
