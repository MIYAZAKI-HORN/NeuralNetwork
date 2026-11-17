#ifndef NEURAL_NET_LAYER_SELF_ATTENSION_H
#define NEURAL_NET_LAYER_SELF_ATTENSION_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"
#include "NeuralNetLayerType.h"
#include "NeuralNetLayerActivation.h"

//=====================================================================================
// Attentionの計算対象
//=====================================================================================
typedef enum tagSelfAttentionMode {
	SELF_ATTENTION_MODE_SPATIAL_2D	= 0x0, // H x W 全体をシーケンスとみなすアテンション
	SELF_ATTENTION_MODE_CHANNEL		= 0x1, // チャネル方向 (C) を対象としたチャンネルアテンション
	SELF_ATTENTION_MODE_AXIS		= 0x2, // 軸方向に限定したアテンション
} SelfAttentionMode;

//=====================================================================================
//  SelfAttention層インターフェース取得
//=====================================================================================
void
NeuralNetLayerSelfAttention_getInterface(LayerFuncTable* pInterface);

//=====================================================================================
//  SelfAttention層作成
//=====================================================================================
bool_t
NeuralNetLayerSelfAttention_constructLayerData(
	uint32_t*			pBuffer,
	uint32_t			sizeOfBufferIn32BitWord,
	uint32_t*			pInputHeight,
	uint32_t*			pInputWidth,
	uint32_t*			pInputChannel,
	SelfAttentionMode	mode,
	uint32_t			num_heads,
	uint32_t			dim_k,
	uint32_t			dim_v,
	uint32_t*			pSizeOfLayerIn32BitWord);

#ifdef __cplusplus
}
#endif

#endif
