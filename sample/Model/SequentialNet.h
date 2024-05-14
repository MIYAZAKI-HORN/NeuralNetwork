#ifndef SEQUENTAIL_NET_H
#define SEQUENTAIL_NET_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetOptimizer.h"
#include "NeuralNetLayerActivation.h"

//==================================================================================================
// 
// �V�[�P���V�������f��
// 
//==================================================================================================
//-------------------------------------------------------------------------
// �o�[�W�������擾
//-------------------------------------------------------------------------
void		SequentialNet_getVersion(uint16_t* pMajorVersion, uint16_t* pMinorVersion, uint16_t* pRevision);
//-------------------------------------------------------------------------
// �G���W���T�C�Y�擾
//-------------------------------------------------------------------------
uint32_t	SequentialNet_getSizeIn32BitWord(uint32_t* pModelData, bool_t fEnableLearning, NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers);
//-------------------------------------------------------------------------
// �G���W���\�z
//-------------------------------------------------------------------------
handle_t	SequentialNet_construct(uint32_t* pModelData, bool_t fEnableLearning, uint32_t batchSize, NeuralNetOptimizerType optimizer, uint32_t numberOfBackPropagationLayers, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord);
//-------------------------------------------------------------------------
// �v�Z����
//-------------------------------------------------------------------------
bool_t		SequentialNet_predict(handle_t hModel, flt32_t* pInputData, uint32_t inputDataArraySize);
//-------------------------------------------------------------------------
// �ʗ\���l�擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getPrediction(handle_t hModel, uint32_t stateIndex, flt32_t* pValue);
//-------------------------------------------------------------------------
// ���͌`��擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getInputShape(handle_t hModel, uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel);
//-------------------------------------------------------------------------
//�o�͌`��擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getOutputShape(handle_t hModel, uint32_t* pHeight, uint32_t* pWidth, uint32_t* pChannel);
//-------------------------------------------------------------------------
// �w���擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getNumberOfLayers(handle_t hModel,uint32_t* pNumOfLayers);
//-------------------------------------------------------------------------
// �w�^�C�v�擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getLayerType(handle_t hModel, uint32_t layerIndex, NetLayerType* pLayerType);
//-------------------------------------------------------------------------
// �w�n���h���擾
//-------------------------------------------------------------------------
bool_t		SequentialNet_getLayerHandle(handle_t hModel, uint32_t layerIndex, handle_t* phLayer);
//-------------------------------------------------------------------------
// ���f���p�����^������
//-------------------------------------------------------------------------
bool_t		SequentialNet_initializeParameter(handle_t hModel);
//-------------------------------------------------------------------------
// Optimizer�n���h���擾
//-------------------------------------------------------------------------
handle_t	SequentialNet_getOptimizer(handle_t hModel, uint32_t layerIndex);
//-------------------------------------------------------------------------
// �w�K�i�덷�t�`���j
//-------------------------------------------------------------------------
bool_t		SequentialNet_fit(handle_t hModel, flt32_t* pLoss, uint32_t arraySize);
//-------------------------------------------------------------------------
// ���f���w�b�_�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_createHeader(uint32_t* pBuffer,uint32_t sizeOfBufferIn32BitWord, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel,uint32_t numberOfLayers,uint32_t* pSizeOfHeaderIn32BitWord);
//-------------------------------------------------------------------------
// Dense�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendDense(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel, uint32_t unit, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// SimpleRNN�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendSimpleRNN(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t unit, NeuralNetActivationType activation, bool_t returnSequence, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// Conv2D�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel, uint32_t nFilter, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t strideWidth, bool_t fPadding, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// DepthwiseConv2D�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendDepthwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t nFilter, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t strideHeight, uint32_t strideWidth, bool_t fPadding, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// PointwiseConv2D�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendPointwiseConv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t pw_nFilter, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// MaxPooling2D�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendMaxPooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight,uint32_t* pInputWidth,uint32_t* pInputChannel, uint32_t poolinghHeight, uint32_t poolingWidth, uint32_t strideHeight, uint32_t strideWidth, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
//  GlobalAveragePooling2D�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendGlobalAveragePooling2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// BatchNormalization�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendBatchNormalization(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
///-------------------------------------------------------------------------
// LayerNormalization�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendLayerNormalization(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// Activation�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendActivation(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, NeuralNetActivationType activation, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
// PreDeconv2D�w�쐬
//-------------------------------------------------------------------------
bool_t		SequentialNet_appendPreDeconv2D(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t strideHeight, uint32_t strideWidth, uint32_t outHeight, uint32_t outWidth, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
//  NeuralNetLayerResidualConnectionSender�w�쐬
//-------------------------------------------------------------------------
bool_t
SequentialNet_appendResidualConnectionSender(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);
//-------------------------------------------------------------------------
//  NeuralNetLayerResidualConnectionReceiver�w�쐬
//-------------------------------------------------------------------------
bool_t
SequentialNet_appendResidualConnectionReceiver(uint32_t* pBuffer, uint32_t sizeOfBufferIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, uint32_t* pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif


#endif
