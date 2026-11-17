#ifndef NEURAL_NET_ATTENSION_MODULE_H
#define NEURAL_NET_ATTENSION_MODULE_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"

//=====================================================================================
//  Attentionモジュールサイズ情報取得
//=====================================================================================
void
NeuralNetAttentionModule_getSize(
	uint32_t	seq_len,						//シーケンス長
	uint32_t	dim_k,							//query,key次元数
	uint32_t	dim_v,							//value次元数
	bool_t		fEnableLearning,				//逆伝播フラグ
	uint32_t*	pRequiredSizeIn32BitWord,		//モジュールサイズ
	uint32_t*	pTempWorkAreaSizeIn32BitWord);	//ワークエリアサイズ

//=====================================================================================
//  Attentionモジュール構築
//=====================================================================================
handle_t
NeuralNetAttentionModule_construct(
	uint32_t	seq_len,						//シーケンス長
	uint32_t	dim_k,							//query,key次元数
	uint32_t	dim_v,							//value次元数
	bool_t		fEnableLearning,				//逆伝播フラグ
	uint32_t*	pObjectWork,					//オブジェクトメモリ領域
	uint32_t	sizeObjectIn32BitWord);			//メモリ領域サイズ
			
//=====================================================================================
//  順伝播計算
//=====================================================================================
bool_t
NeuralNetAttentionModule_forward(handle_t hAttention, const flt32_t* pQ,const flt32_t* pK,const flt32_t* pV, flt32_t* pVOut,uint32_t* pTemporaryBuffer);

//=====================================================================================
//  逆伝播計算
//=====================================================================================
bool_t
NeuralNetAttentionModule_backward(handle_t hAttention, flt32_t* pDQ, flt32_t* pDK, flt32_t* pDV, const flt32_t* pDLoss, uint32_t* pTemporaryBuffer);

#ifdef __cplusplus
}
#endif

#endif
