#ifndef NetworkNetModelConstructor_H
#define NetworkNetModelConstructor_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "SequentialNet.h"

//--------------------------------------------------------------------
// ���C���[���\����
//--------------------------------------------------------------------
#define LAYER_INFORMATION_MAX_PARAMS	(10)

typedef struct tagLayerInformation {
	NetLayerType	layerType;
	uint32_t		parameterArray[LAYER_INFORMATION_MAX_PARAMS];
	uint32_t		numberOfParams;
} LayerInformation;

//--------------------------------------------------------------------
// �V�[�P���V�������f�����\����
//--------------------------------------------------------------------
#define MAX_LAYERS	(200)
typedef struct tagModelInformation {
	uint32_t			inHeight;
	uint32_t			inWidth;
	uint32_t			inChannel;
	uint32_t			numberOfLayers;
	LayerInformation	layerArray[MAX_LAYERS];
} ModelInformation;

//--------------------------------------------------------------------
// �V�[�P���V�������f���C���[�W���쐬����֐�
//--------------------------------------------------------------------
bool_t	
constructNeuralNetModel(	ModelInformation*	pModelInformation,	//�l�b�g���[�N���
							uint32_t**			ppNeuralNetworkImage,	//��������j���[�����l�b�g���[�N�C���[�W�̃|�C���^�̃|�C���^
							uint32_t*			pSizeOfImageIn32BitWord	//��������j���[�����l�b�g���[�N�C���[�W�̃T�C�Y���擾����p�����^�̃|�C���^
);

//--------------------------------------------------------------------
// �V�[�P���V�������f���̃��C���[�����Z�b�g����w���p�[�֐�
//--------------------------------------------------------------------
void	sequential_model_header(ModelInformation* pModelInformation, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);
bool_t	dense(ModelInformation* pModelInformation, uint32_t units);
bool_t	simple_rnn(ModelInformation* pModelInformation, uint32_t units, uint32_t activation, uint32_t retuenSequence);
bool_t	activation(ModelInformation* pModelInformation, uint32_t activation);
bool_t	conv2d(ModelInformation* pModelInformation, uint32_t filters,uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,bool_t	fPadding);
bool_t	depthwise_conv2d(ModelInformation* pModelInformation, uint32_t nfilters, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width, bool_t fPadding);
bool_t	pointwise_conv2d(ModelInformation* pModelInformation, uint32_t pw_filters);
bool_t	max_pooling2d(ModelInformation* pModelInformation, uint32_t pool_height, uint32_t pool_width, uint32_t stride_height, uint32_t stride_width);
bool_t	global_average_pooling2d(ModelInformation* pModelInformation);
bool_t	batch_normalization(ModelInformation* pModelInformation);
bool_t	layer_normalization(ModelInformation* pModelInformation);
bool_t	predeconv2d(ModelInformation* pModelInformation,uint32_t stride_height,uint32_t stride_width,uint32_t out_height,uint32_t out_width);
bool_t	residual_connection_sender(ModelInformation* pModelInformation);
bool_t	residual_connection_receiver(ModelInformation* pModelInformation);

//--------------------------------------------------------------------
// �V�[�P���V�������f����`�e�L�X�g�t�@�C�����烌�C���[�����擾����
//--------------------------------------------------------------------
void	getLayerInformation(char* pLayerInformationText, uint32_t* pInputDim, LayerInformation* pLayerInformationArray, uint32_t* pNumberOfLayers);

//--------------------------------------------------------------------
// �w��񂩂�w�C���[�W���쐬����
//--------------------------------------------------------------------
bool_t	constructLayer(uint32_t* pBuffer, uint32_t bufferSizeIn32BitWord, uint32_t* pInputHeight, uint32_t* pInputWidth, uint32_t* pInputChannel, LayerInformation* pLayerInformation, uint32_t* pLayerImageSizeIn32BitWord, char* pLayerName);

//--------------------------------------------------------------------
// ��`�t�@�C����ǎ惂�f�����쐬����
//--------------------------------------------------------------------
bool_t	constructNeuralNetModelByFile(const char* pNetworkDefinitionFileName, uint32_t** ppNeuralNetworkImage, uint32_t* pSizeOfNeuralNetworkImageIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
