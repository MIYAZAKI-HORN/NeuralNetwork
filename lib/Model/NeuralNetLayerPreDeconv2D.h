#ifndef NEURAL_NET_LAYER_DECONV2D_H
#define NEURAL_NET_LAYER_DECONV2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  PreDeconv2D�w�C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerPreDeconv2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  PreDeconv2D�w�쐬
//=====================================================================================
bool_t
NeuralNetLayerPreDeconv2D_constructLayerData(
	uint32_t*	pBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t	strideHeight,
	uint32_t	strideWidth,
	uint32_t	outHeight,
	uint32_t	outWidth,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
