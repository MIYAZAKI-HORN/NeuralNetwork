#include "STDTypeDefinition.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetAttentionModule.h"

//=====================================================================================
//  Attention module構造体
//=====================================================================================
typedef struct tagAttentionModule {
	uint32_t	seq_len;			//シーケンス長
	uint32_t	dim_k;				//query,key次元数
	uint32_t	dim_v;				//value次元数
	bool_t		fEnableLearning;	//逆伝播フラグ
	flt32_t*	pQ;					//誤差逆伝播用データバッファ：Q値
	flt32_t*	pK;					//誤差逆伝播用データバッファ：K値
	flt32_t*	pV;					//誤差逆伝播用データバッファ：V値
	flt32_t*	pScore;				//誤差逆伝播用データバッファ：Score値
} NeuralNetAttentionModule;

//=====================================================================================
//  モジュールサイズ情報取得
//=====================================================================================
void
NeuralNetAttentionModule_getSize(
	uint32_t	seq_len,
	uint32_t	dim_k,
	uint32_t	dim_v,
	bool_t		fEnableLearning,
	uint32_t*	pRequiredSizeIn32BitWord,
	uint32_t*	pTempWorkAreaSizeIn32BitWord) {
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ
	//---------------------------------------------------------------------------------
	if (pRequiredSizeIn32BitWord != NULL) {
		*pRequiredSizeIn32BitWord = size_in_type(sizeof(NeuralNetAttentionModule), uint32_t);
		if (fEnableLearning == TRUE) {
			*pRequiredSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　Q
			*pRequiredSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　K
			*pRequiredSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);		//　V
			*pRequiredSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * seq_len, uint32_t);	//	score
		}
	}
	//---------------------------------------------------------------------------------
	//内部計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord = size_in_type(sizeof(flt32_t) * seq_len * seq_len, uint32_t);	//	score
	}
}

//=====================================================================================
//  モジュール構築
//=====================================================================================
handle_t
NeuralNetAttentionModule_construct(
	uint32_t	seq_len,
	uint32_t	dim_k,
	uint32_t	dim_v,
	bool_t		fEnableLearning,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord) {
	NeuralNetAttentionModule* pAttentionModule = NULL;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (seq_len == 0 || dim_k == 0 || dim_v == 0) {
		return NULL;
	}
	//---------------------------------------------------------------------------------
	//モジュールサイズ情報取得
	//---------------------------------------------------------------------------------
	uint32_t	requiredSize = 0;
	NeuralNetAttentionModule_getSize(seq_len, dim_k, dim_v, fEnableLearning, &requiredSize, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	//---------------------------------------------------------------------------------
	//モジュール構築
	//---------------------------------------------------------------------------------
	pAttentionModule = (NeuralNetAttentionModule*)pObjectWork;
	pObjectWork += size_in_type(sizeof(NeuralNetAttentionModule), uint32_t);
	//学習用データ
	if (fEnableLearning == TRUE) {
		//data
		pAttentionModule->pQ = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　Q;
		pAttentionModule->pK = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　K
		pAttentionModule->pV = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);		//　V
		pAttentionModule->pScore = (flt32_t*)pObjectWork;
		pObjectWork += size_in_type(sizeof(flt32_t) * seq_len * seq_len, uint32_t);		//	SCORE
	}
	else {
		pAttentionModule->pQ = NULL;
		pAttentionModule->pK = NULL;
		pAttentionModule->pV = NULL;
		pAttentionModule->pScore = NULL;
	}
	//パラメタのセット
	pAttentionModule->seq_len	= seq_len;
	pAttentionModule->dim_k		= dim_k;
	pAttentionModule->dim_v		= dim_v;
	pAttentionModule->fEnableLearning = fEnableLearning;
	return (handle_t)pAttentionModule;
}

//=====================================================================================
//  順伝播計算
//=====================================================================================
bool_t
NeuralNetAttentionModule_forward(	
	handle_t		hAttention, 
	const flt32_t*	pQ,
	const flt32_t*	pK,
	const flt32_t*	pV,
	flt32_t*		pVOut,
	uint32_t*		pTemporaryBuffer) {
	uint32_t	i;
	NeuralNetAttentionModule* pAttentionModule = (NeuralNetAttentionModule*)hAttention;
	uint32_t	seq_len = pAttentionModule->seq_len;
	uint32_t	dim_k = pAttentionModule->dim_k;
	uint32_t	dim_v = pAttentionModule->dim_v;
	//---------------------------------------------------------------------------------
	//逆伝播のためにQ、K、Vを保持
	//---------------------------------------------------------------------------------
	if (pAttentionModule->fEnableLearning == TRUE) {
		//Q
		mem_copy_flt(pAttentionModule->pQ, pQ, (seq_len * dim_k));
		//K
		mem_copy_flt(pAttentionModule->pK, pK, (seq_len * dim_k));
		//V
		mem_copy_flt(pAttentionModule->pV, pV, (seq_len * dim_v));
	}
	//---------------------------------------------------------------------------------
	//scoreの計算領域の確保
	//---------------------------------------------------------------------------------
	flt32_t* pScore = (flt32_t*)pTemporaryBuffer;
	//---------------------------------------------------------------------------------
	//スコア行列 (Q * K^T): [seq_len, seq_len]計算
	//---------------------------------------------------------------------------------
	matmul_transposed_forward(pQ, pK, pScore, seq_len, dim_k, seq_len);
	//---------------------------------------------------------------------------------
	//スケーリング (/ sqrt(dim_k))
	//---------------------------------------------------------------------------------
	flt32_t scale = 1.0f / low_cost_sqrt((flt32_t)dim_k,2);
	mem_multiply_flt(pScore, scale, (seq_len * seq_len));
	//---------------------------------------------------------------------------------
	//重み（確率分布）生成
	//---------------------------------------------------------------------------------
	flt32_t* pScoreHead = pScore;
	i = seq_len;
	while (i--) {
		softmax_forward(pScoreHead, pScoreHead, seq_len);
		pScoreHead += seq_len;
	}
	//---------------------------------------------------------------------------------
	//逆伝播のためのscoresを保持
	//---------------------------------------------------------------------------------
	if (pAttentionModule->fEnableLearning == TRUE) {
		mem_copy_flt(pAttentionModule->pScore, pScore, (seq_len * seq_len));
	}
	//---------------------------------------------------------------------------------
	//Value出力を生成
	//---------------------------------------------------------------------------------
	matmul_forward(pScore, pV, pVOut, seq_len, seq_len, dim_v);
	return TRUE;
}

//=====================================================================================
//  逆伝播計算
//=====================================================================================
bool_t
NeuralNetAttentionModule_backward(	
	handle_t		hAttention,
	flt32_t*		pDQ, 
	flt32_t*		pDK, 
	flt32_t*		pDV, 
	const flt32_t*	pDLoss, 
	uint32_t*		pTemporaryBuffer) {
	uint32_t	i;
	NeuralNetAttentionModule* pAttentionModule = hAttention;
	uint32_t	seq_len	= pAttentionModule->seq_len;
	uint32_t	dim_k	= pAttentionModule->dim_k;
	uint32_t	dim_v	= pAttentionModule->dim_v;
	//---------------------------------------------------------------------------------
	//逆伝播対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pAttentionModule->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//重みパラメタ(Score)の微分値領域の確保と初期化
	//---------------------------------------------------------------------------------
	flt32_t* pDScore	= (flt32_t*)pTemporaryBuffer;
	mem_set_flt(pDScore,0.0f, (seq_len * seq_len));
	//---------------------------------------------------------------------------------
	//Vの逆伝播
	//重みパラメタ(Score)の誤差逆伝播　微分値はpDScoreに保持される
	//---------------------------------------------------------------------------------
	matmul_backward(pAttentionModule->pScore, pAttentionModule->pV, pDLoss, pDScore, pDV, seq_len, seq_len, dim_v);
	//---------------------------------------------------------------------------------
	//重み（確率分布）生成の逆伝播
	//---------------------------------------------------------------------------------
	flt32_t* pScoreHead		= pAttentionModule->pScore;
	flt32_t* pDScoreHead	= pDScore;
	i = seq_len;
	while (i--) {
		softmax_backward(pScoreHead, pDScoreHead, pDScoreHead, seq_len);
		pScoreHead	+= seq_len;
		pDScoreHead += seq_len;
	}
	//---------------------------------------------------------------------------------
	//スケーリング (/ sqrt(d_k))された分微分値にかける
	//---------------------------------------------------------------------------------
	flt32_t scale = 1.0f / low_cost_sqrt((flt32_t)dim_k, 2);
	mem_multiply_flt(pDScore, scale, (seq_len * seq_len));
	//---------------------------------------------------------------------------------
	//スコア行列Score = (Q * K^T): [seq_len, seq_len]計算の逆伝播
	//---------------------------------------------------------------------------------
	matmul_transposed_backward(pAttentionModule->pQ, pAttentionModule->pK, pDScore, pDQ, pDK,seq_len, dim_k, seq_len);
	return TRUE;
}