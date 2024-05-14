#ifndef NEURAL_NET_LAYER_DENSE_H
#define NEURAL_NET_LAYER_DENSE_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"

//=====================================================================================
//  Dense�w�C���^�[�t�F�[�X�擾
//=====================================================================================
void
NeuralNetLayerDense_getInterface(LayerFuncTable* pInterface);
	
//=====================================================================================
//  Dense�w�쐬
//=====================================================================================
bool_t
NeuralNetLayerDense_constructLayerData(
	uint32_t*		pBuffer,
	uint32_t		sizeOfBufferIn32BitWord,
	uint32_t*		pInputHeight,
	uint32_t*		pInputWidth,
	uint32_t*		pInputChannel,
	uint32_t		unit,
	uint32_t*		pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
