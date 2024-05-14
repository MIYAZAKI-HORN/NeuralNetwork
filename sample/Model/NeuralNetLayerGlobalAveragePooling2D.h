#ifndef NEURAL_NET_LAYER_GLOBAL_AVERAGE_POOLING2D_H
#define NEURAL_NET_LAYER_GLOBAL_AVERAGE_POOLING2D_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  GlobalAveragePooling2D�w�C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerGlobalAveragePooling2D_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  GlobalAveragePooling2D�w�쐬
//=====================================================================================
bool_t
NeuralNetLayerGlobalAveragePooling2D_constructLayerData(	
	uint32_t*	ppBuffer,
	uint32_t	sizeOfBufferIn32BitWord,
	uint32_t*	pInputHeight,
	uint32_t*	pInputWidth,
	uint32_t*	pInputChannel,
	uint32_t*	pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
