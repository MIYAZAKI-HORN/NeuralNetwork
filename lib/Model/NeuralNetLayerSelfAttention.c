#include "STDTypeDefinition.h"
#include "NeuralNetLayer.h"
#include "NeuralNetLayerFunction.h"
#include "NeuralNetAttentionModule.h"
#include "NeuralNetLayerSelfAttention.h"
#include "NeuralNetOptimizer.h"
#include "RandomValueGenerator.h"

#ifndef max
	#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

//=====================================================================================
//  SelfAttention層ブロック情報ヘッダー
//=====================================================================================
typedef struct tagSelfAttentionNeuralNetHeader {
	NeuralNetHeader		super;		//base layer header
	uint32_t			num_heads;	//multi-head attentionヘッド数
	uint32_t			dim_k;		//query,key次元数
	uint32_t			dim_v;		//key次元数
	SelfAttentionMode	mode;		//attentionの対象
} SelfAttentionNeuralNetHeader;

//=====================================================================================
//  SelfAttention層構造体
//=====================================================================================
typedef struct tagSelfAttentionNeuralNetLayer {
	NeuralNetLayer	super;			//base layer class
	flt32_t*		pX;				//誤差逆伝播用データバッファ：入力
	flt32_t*		pVxWO_in;		//誤差逆伝播用データバッファ：Value出力
	handle_t		hOptimizer;		//オプティマイザーハンドル
	handle_t*		phAttention;	//attention moduleハンドル配列
} SelfAttentionNeuralNetLayer;

//=====================================================================================
//  形状取得
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_getShape(handle_t hLayer, DataShape* pInputShape, DataShape* pOutputShape) {
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	uint32_t	inHeight = pNeuralNetHeader->inHeight;
	uint32_t	inWidth = pNeuralNetHeader->inWidth;
	uint32_t	inChannel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//出力データ形状
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_SPATIAL_2D:	// H x W 全体をシーケンスとみなすアテンション
		DataShape_construct(pOutputShape, inHeight, inWidth, dim_v);
		break;
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		if (dim_v == (inHeight * inWidth)) {
			//出力形状を入力形状と同じと考える
			DataShape_construct(pOutputShape, inHeight, inWidth, inChannel);
		}
		else {
			//出力形状において、チャネルはチャネルの位置に配置する
			DataShape_construct(pOutputShape, 1, dim_v, inChannel);
		}
		break;
	case SELF_ATTENTION_MODE_AXIS:			// 軸方向に限定したアテンション
		if (inHeight > 1) {
			//特徴量次元を幅にセット
			DataShape_construct(pOutputShape, inHeight, dim_v, 1);
		}
		else {
			//特徴量次元をチャネルにセット
			DataShape_construct(pOutputShape, 1, inWidth, dim_v);
		}
		break;
	}
	//---------------------------------------------------------------------------------
	//入力データ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(pInputShape, inHeight, inWidth, inChannel);
	return TRUE;
}

//=====================================================================================
//  入出力形状
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_getAttentionMatrixShape(SelfAttentionMode mode, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel,uint32_t dim_v, uint32_t* pSeq_len, uint32_t* pSeq_dim,DataShape* pOutputShape) {
	uint32_t	seq_len = 0;
	uint32_t	dim = 0;
	//---------------------------------------------------------------------------------
	//アテンション領域の選択
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_SPATIAL_2D:	// H x W 全体をシーケンスとみなすアテンション
		seq_len = inHeight * inWidth;
		dim = inChannel;
		break;
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//入力をglobal average poolingを取ったものとするため、次元は1にする
		seq_len = inChannel;
		dim = 1;
		break;
	case SELF_ATTENTION_MODE_AXIS:			// 軸方向に限定したアテンション
		if (inHeight > 1) {
			//Heightをシーケンスとし、Width x Channelを特徴量とする
			seq_len = inHeight;
			dim = inWidth * inChannel;
		}
		else {
			//Widthを次元とし、Channelを特徴量とする
			seq_len = inWidth;
			dim = inChannel;
		}
		break;
	default:
		return FALSE;
	}
	//シーケンス長と次元数をセット
	if (pSeq_len != NULL) {
		*pSeq_len = seq_len;
	}
	if (pSeq_dim != NULL) {
		*pSeq_dim = dim;
	}
	//---------------------------------------------------------------------------------
	//出力形状
	//---------------------------------------------------------------------------------
	if (pOutputShape != NULL) {
		switch (mode) {
		case SELF_ATTENTION_MODE_SPATIAL_2D:	// H x W 全体をシーケンスとみなすアテンション
			DataShape_construct(pOutputShape, inHeight, inWidth, dim_v);
			break;
		case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
			if (dim_v == (inHeight * inWidth)) {
				//出力形状を入力形状と同じと考える
				DataShape_construct(pOutputShape, inHeight, inWidth, inChannel);
			}
			else {
				//出力形状において、チャネルはチャネルの位置に配置する
				DataShape_construct(pOutputShape, 1, dim_v, inChannel);
			}
			break;
		case SELF_ATTENTION_MODE_AXIS:			// 軸方向に限定したアテンション
			if (inHeight > 1) {
				//特徴量次元を幅にセット
				DataShape_construct(pOutputShape, inHeight, dim_v, 1);
			}
			else {
				//特徴量次元をチャネルにセット
				DataShape_construct(pOutputShape, 1, inWidth, dim_v);
			}
			break;
		default:
			return FALSE;
		}
	}
	return TRUE;
}

//=====================================================================================
//  single headのマトリクスとマルチヘッドのマトリクスを変換する関数
//=====================================================================================
static
void
SingleHead2MultiHead(uint32_t seq_len,uint32_t dim, uint32_t num_heads, flt32_t* pMatrix, flt32_t* pWorkArea,bool_t fForward) {
	//---------------------------------------------------------------------------------
	//各ヘッドの特徴量次元数
	//---------------------------------------------------------------------------------
	uint32_t	dim_h = dim / num_heads;
	//---------------------------------------------------------------------------------
	//各ヘッドの特徴量が並ぶ様に、並び替え　
	// seq_len x dim(nHeads x dim_h) →　nHeads x seq_len x dim_h
	//---------------------------------------------------------------------------------
	flt32_t* pQ_i	= NULL;
	flt32_t* pQH_i	= NULL;
	if (fForward == TRUE) {
		pQ_i	= pMatrix;
		pQH_i	= pWorkArea;
	}
	else {
		pQ_i	= pWorkArea;
		pQH_i	= pMatrix;
	}
	uint32_t i = num_heads;
	while (i--) {
		flt32_t* pQ_is = pQ_i;
		//各ヘッドにヘッドのdim_k_h（dim_v_h）分の特徴量をシーケンス分コピーする
		uint32_t j = seq_len;
		while (j--) {
			//各ヘッド分の特徴量をコピー
			if (fForward == TRUE) {
				mem_copy_flt(pQH_i, pQ_is, dim_h);
			}
			else {
				mem_copy_flt(pQ_is, pQH_i, dim_h);
			}
			//QKVの次の特徴量へ移動
			pQ_is += dim;
			//各ヘッドにシーケンス分
			pQH_i += dim_h;
		}
		pQ_i += dim_h;
	}
	//変換されたマトリクスをコピーする
	mem_copy_flt(pMatrix, pWorkArea, (seq_len * dim));
}

//=====================================================================================
//  順伝播計算
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_forward(handle_t hLayer, PropagationInfo* pPropagationInfo)
{
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	uint32_t	i;
	uint32_t	seq_len	= 0;
	uint32_t	dim		= 0;
	flt32_t*	pWQ		= NULL;
	flt32_t*	pWK		= NULL;
	flt32_t*	pWV		= NULL;
	flt32_t*	pWO		= NULL;
	uint32_t*	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	num_heads = pSelfAttentionNeuralNetHeader->num_heads;
	uint32_t	dim_k = pSelfAttentionNeuralNetHeader->dim_k;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	uint32_t	inHeight= pNeuralNetHeader->inHeight;
	uint32_t	inWidth = pNeuralNetHeader->inWidth;
	uint32_t	inChannel = pNeuralNetHeader->inChannel;
	DataShape	outputShape;
	//---------------------------------------------------------------------------------
	//アテンション領域の選択/出力形状の取得
	//---------------------------------------------------------------------------------
	if (FALSE == NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, &outputShape)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//シーケンスにchannelを対象とする場合、global average poolingをとり、チャネル方向の重みを入力とする 形状は1xC(Cx1)になる
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//channelごとにHxWの平均値をとる
		global_average_pooling2D_forward(pPropagationInfo->pInputBuffer, pPropagationInfo->pInputBuffer, inHeight, inWidth, inChannel);
		break;
	default:
		break;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	pWQ = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWK = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWV = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_v, uint32_t);
	if (num_heads > 1) {
		pWO = (flt32_t*)pLayerParam;
		pLayerParam += size_in_type(sizeof(flt32_t) * dim_v * dim_v, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//Q,K,Vおよびマルチヘッド用並び替えワークエリアの計算領域の確保
	//---------------------------------------------------------------------------------
	uint32_t*	pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;
	flt32_t*	pQ		= (flt32_t*)pTemporaryBuffer;
	flt32_t*	pK		= pQ + seq_len * dim_k;
	flt32_t*	pV		= pK + seq_len * dim_k;
	pTemporaryBuffer =  (uint32_t*)(pV + seq_len * dim_v);
	//single headとmulti-headとの変換に必要なワークエリア
	flt32_t* pMHwork = NULL;
	if (num_heads > 1) {
		pMHwork = (flt32_t*)pTemporaryBuffer;
		uint32_t dim_max = max(dim_k, dim_v);
		pTemporaryBuffer += seq_len * dim_max;
	}
	//---------------------------------------------------------------------------------
	//Q,K,Vの生成計算 Q = X x WQ
	//---------------------------------------------------------------------------------
	matmul_forward(pPropagationInfo->pInputBuffer, pWQ, pQ, seq_len, dim, dim_k);
	matmul_forward(pPropagationInfo->pInputBuffer, pWK, pK, seq_len, dim, dim_k);
	matmul_forward(pPropagationInfo->pInputBuffer, pWV, pV, seq_len, dim, dim_v);
	if (pNeuralNetLayer->fEnableLearning == TRUE) {
		//逆伝播のために入力値を保持
		mem_copy_flt(pSelfAttentionLayer->pX, pPropagationInfo->pInputBuffer, (seq_len * dim));
	}
	if (num_heads > 1) {
		//---------------------------------------------------------------------------------
		//マルチヘッドが並ぶマトリクスに変換
		//---------------------------------------------------------------------------------
		SingleHead2MultiHead(seq_len, dim_k, num_heads, pQ, pMHwork,TRUE);
		SingleHead2MultiHead(seq_len, dim_k, num_heads, pK, pMHwork,TRUE);
		SingleHead2MultiHead(seq_len, dim_v, num_heads, pV, pMHwork,TRUE);
		//---------------------------------------------------------------------------------
		//multi-head attention moduleの分割次元数
		//---------------------------------------------------------------------------------
		uint32_t	dim_k_h = dim_k / num_heads;
		uint32_t	dim_v_h = dim_v / num_heads;
		//---------------------------------------------------------------------------------
		//multi-head attentionの順伝播
		//Value出力生成：pMHworkに作成
		//---------------------------------------------------------------------------------
		flt32_t*	pQH = pQ;
		flt32_t*	pKH = pK;
		flt32_t*	pVH = pV;
		flt32_t*	pVHout = pMHwork;
		for (i = 0; i < num_heads; i++) {
			NeuralNetAttentionModule_forward(pSelfAttentionLayer->phAttention[i], pQH, pKH, pVH, pVHout, pTemporaryBuffer);
			pQH += seq_len * dim_k_h;
			pKH += seq_len * dim_k_h;
			pVH += seq_len * dim_v_h;
			pVHout += seq_len * dim_v_h;
		}
		//---------------------------------------------------------------------------------
		//concatenated Value　マルチヘッドが並ぶマトリクスからsingle headに変換
		//---------------------------------------------------------------------------------
		mem_copy_flt(pV, pMHwork, (seq_len * dim_v));	// Vにコピー
		SingleHead2MultiHead(seq_len, dim_v, num_heads, pV, pMHwork, FALSE);	// もとのV形状に変換
		//---------------------------------------------------------------------------------
		// WOをかけて最終Valueを生成する　V = V x WO
		//Value出力生成：InputBufferに作成
		//---------------------------------------------------------------------------------
		matmul_forward(pV, pWO, pPropagationInfo->pInputBuffer, seq_len, dim_v, dim_v);
		if (pNeuralNetLayer->fEnableLearning == TRUE) {
			//入力値Vを保持
			mem_copy_flt(pSelfAttentionLayer->pVxWO_in, pV, (seq_len * dim_v));
		}
	}
	else {
		//---------------------------------------------------------------------------------
		//Value出力生成：InputBufferに作成
		//---------------------------------------------------------------------------------
		NeuralNetAttentionModule_forward(pSelfAttentionLayer->phAttention[0], pQ, pK, pV, pPropagationInfo->pInputBuffer, pTemporaryBuffer);
	}
	switch (mode) {
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//チャネルラストとするため形状を変換する
		//C(sequence) x dim_v →dim_v x C(sequence)
		channel_first_2_last(pPropagationInfo->pInputBuffer, pPropagationInfo->pOutputBuffer, 1, seq_len, dim_v);
		break;
	default:
		//OutputBufferにコピーする
		mem_copy_flt(pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, (seq_len * dim_v));
		break;
	}
	//---------------------------------------------------------------------------------
	//データ出力形式
	//---------------------------------------------------------------------------------
	pPropagationInfo->dataShape = outputShape;
	return TRUE;
}

//=====================================================================================
//  逆伝播計算
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_backward(handle_t hLayer,PropagationInfo* pPropagationInfo)
{
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	uint32_t*	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	uint32_t	seq_len	= 0;
	uint32_t	dim		= 0;
	uint32_t	i;
	flt32_t*	pWQ	= NULL;
	flt32_t*	pWK = NULL;
	flt32_t*	pWV = NULL;
	flt32_t*	pWO = NULL;
	flt32_t*	pDWQ = NULL;
	flt32_t*	pDWK = NULL;
	flt32_t*	pDWV = NULL;
	flt32_t*	pDWO = NULL;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	num_heads = pSelfAttentionNeuralNetHeader->num_heads;
	uint32_t	dim_k = pSelfAttentionNeuralNetHeader->dim_k;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	uint32_t	inHeight = pNeuralNetHeader->inHeight;
	uint32_t	inWidth = pNeuralNetHeader->inWidth;
	uint32_t	inChannel = pNeuralNetHeader->inChannel;
	OptimizerFunctionTable optimizerFunctionTable;
	//---------------------------------------------------------------------------------
	//逆伝播対象でない場合はエラー
	//---------------------------------------------------------------------------------
	if (pNeuralNetLayer->fEnableLearning == FALSE) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//アテンション領域の選択/出力形状の取得
	//---------------------------------------------------------------------------------
	if (FALSE == NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, NULL)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	pWQ = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWK = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWV = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_v, uint32_t);
	if (num_heads > 1) {
		pWO = (flt32_t*)pLayerParam;
		pLayerParam += size_in_type(sizeof(flt32_t) * dim_v * dim_v, uint32_t);
	}
	//---------------------------------------------------------------------------------
	//WQ,WK,WVの微分値を保持するバッファポインタ取得
	//---------------------------------------------------------------------------------
	NeuralNetOptimizer_getInterface(pSelfAttentionLayer->hOptimizer, &optimizerFunctionTable);
	pDWQ = optimizerFunctionTable.pGetDeltaParameterBuffer(pSelfAttentionLayer->hOptimizer);
	pDWK = pDWQ + dim * dim_k;
	pDWV = pDWK + dim * dim_k;
	pDWO = pDWV + dim * dim_v;	//マルチヘッドの時にのみ利用
	//---------------------------------------------------------------------------------
	//Q,K,Vおよびscoreの微分値領域の確保と初期化
	//---------------------------------------------------------------------------------
	uint32_t* pTemporaryBuffer = pPropagationInfo->pTemporaryBuffer;
	flt32_t* pDQ = (flt32_t*)pTemporaryBuffer;
	flt32_t* pDK = pDQ + seq_len * dim_k;
	flt32_t* pDV = pDK + seq_len * dim_k;
	pTemporaryBuffer = (uint32_t*)(pDV + seq_len * dim_v);
	//single headとmulti-headとの変換に必要なワークエリア
	flt32_t* pMHwork = NULL;
	if (num_heads > 1) {
		pMHwork = (flt32_t*)pTemporaryBuffer;
		uint32_t dim_max = max(dim_k, dim_v);
		pTemporaryBuffer += seq_len * dim_max;
	}
	//---------------------------------------------------------------------------------
	//形状変換
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//順伝播で変換した形状を戻す
		//dim_v x C(sequence) →　C(sequence) x dim_v
		channel_last_2_first(pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, 1, dim_v, seq_len);
		//OutputBufferにコピーする
		mem_copy_flt(pPropagationInfo->pOutputBuffer, pPropagationInfo->pInputBuffer, (seq_len * dim_v));
		break;
	default:
		break;
	}
	//---------------------------------------------------------------------------------
	//重みパラメタ(Score)の誤差逆伝播　微分値はpOutputBufferに保持されている
	//---------------------------------------------------------------------------------
	mem_set_flt(pDQ, 0.0f, (seq_len * dim_k));
	mem_set_flt(pDK, 0.0f, (seq_len * dim_k));
	mem_set_flt(pDV, 0.0f, (seq_len * dim_v));
	if (num_heads > 1) {
		//損失保持バッファ：pOutputBuffer
		//逆伝播値バッファ：pInputBuffer
		flt32_t* pDVLoss = pPropagationInfo->pInputBuffer;
		//損失保持バッファを初期化
		mem_set_flt(pDVLoss, 0.0f, (seq_len * dim_v));
		mem_set_flt(pDWO, 0.0f, (dim_v * dim_v));
		//逆伝播 V = V x WO
		matmul_backward(pSelfAttentionLayer->pVxWO_in, pWO, pPropagationInfo->pOutputBuffer, pDVLoss, pDWO, seq_len, dim_v, dim_v);
		//---------------------------------------------------------------------------------
		//multi-head attention moduleの分割次元数
		//---------------------------------------------------------------------------------
		uint32_t	dim_k_h = dim_k / num_heads;
		uint32_t	dim_v_h = dim_v / num_heads;
		//---------------------------------------------------------------------------------
		//伝播誤差をマルチヘッドが並ぶマトリクスに変換
		//---------------------------------------------------------------------------------
		SingleHead2MultiHead(seq_len, dim_v, num_heads, pDVLoss, pMHwork, TRUE);
		//---------------------------------------------------------------------------------
		//各ヘッドでVの逆伝播　pDQ, pDK, pDV　←　pDVLoss（pInputBuffer）（損失保持バッファ）
		//---------------------------------------------------------------------------------
		flt32_t* pDQH = pDQ; //出力値
		flt32_t* pDKH = pDK; //出力値
		flt32_t* pDVH = pDV; //出力値
		flt32_t* pDVLossH = pDVLoss;	// 入力値:pInputBufferに前誤差逆伝播値が格納されている
		for (i = 0; i < num_heads; i++) {
			NeuralNetAttentionModule_backward(pSelfAttentionLayer->phAttention[i], pDQH, pDKH, pDVH, pDVLossH, pTemporaryBuffer);
			pDQH += seq_len * dim_k_h;
			pDKH += seq_len * dim_k_h;
			pDVH += seq_len * dim_v_h;
			pDVLossH += seq_len * dim_v_h;
		}
		//---------------------------------------------------------------------------------
		//各ヘッドの誤差伝播値をconcatenated（並べ替え）
		//---------------------------------------------------------------------------------
		SingleHead2MultiHead(seq_len, dim_k, num_heads, pDQ, pMHwork, FALSE);
		SingleHead2MultiHead(seq_len, dim_k, num_heads, pDK, pMHwork, FALSE);
		SingleHead2MultiHead(seq_len, dim_v, num_heads, pDV, pMHwork, FALSE);
	}
	else {
		//---------------------------------------------------------------------------------
		//Vの逆伝播　pDQ, pDK, pDV　←　pOutputBuffer（損失保持バッファ）
		//---------------------------------------------------------------------------------
		NeuralNetAttentionModule_backward(pSelfAttentionLayer->phAttention[0], pDQ, pDK, pDV, pPropagationInfo->pOutputBuffer, pTemporaryBuffer);
	}
	//---------------------------------------------------------------------------------
	//Q,K,Vの生成計算 X x WQ → Qの逆伝播　Qの微分値が伝搬され、Xの微分値とWQの微分値が生成される
	//Xの微分値はpPropagationInfo->pInputBufferに保持され、次の層の微分値入力となる
	//---------------------------------------------------------------------------------
	flt32_t* pDX = pPropagationInfo->pInputBuffer;
	mem_set_flt(pDX,  0.0f, (seq_len * dim));
	mem_set_flt(pDWQ, 0.0f, (dim * dim_k));
	mem_set_flt(pDWK, 0.0f, (dim * dim_k));
	mem_set_flt(pDWV, 0.0f, (dim * dim_v));
	//逆伝播 pDXは加算される
	matmul_backward(pSelfAttentionLayer->pX, pWQ, pDQ, pDX, pDWQ, seq_len, dim, dim_k);
	matmul_backward(pSelfAttentionLayer->pX, pWK, pDK, pDX, pDWK, seq_len, dim, dim_k);
	matmul_backward(pSelfAttentionLayer->pX, pWV, pDV, pDX, pDWV, seq_len, dim, dim_v);
	//---------------------------------------------------------------------------------
	//global average poolingの逆伝播
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//global average pooling の逆伝播を実施する (入出力に同一バッファを指定できる)
		//関数引数：逆伝播出力バッファー、　lossバッファー、入力時形状
		global_average_pooling2D_backward(pPropagationInfo->pInputBuffer, pPropagationInfo->pInputBuffer, inHeight, inWidth, inChannel);
		break;
	default:
		break;
	}
	//---------------------------------------------------------------------------------
	//出力（入力方向）データサイズ形状
	//---------------------------------------------------------------------------------
	DataShape_construct(&pPropagationInfo->dataShape, inHeight, inWidth, inChannel);
	return TRUE;
}

//=====================================================================================
//  パラメタ更新
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_update(handle_t hLayer) {
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	flt32_t*	pParameter;
	uint32_t*	pLayerParam;
	OptimizerFunctionTable	optimizerInterface;
	NeuralNetOptimizer_getInterface(pSelfAttentionLayer->hOptimizer, &optimizerInterface);
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	pParameter = (flt32_t*)pLayerParam;
	//---------------------------------------------------------------------------------
	//層パラメタ更新
	//---------------------------------------------------------------------------------
	//　WQ,WK,WV
	optimizerInterface.pUpdate(pSelfAttentionLayer->hOptimizer, pParameter);
	return TRUE;
}

//=====================================================================================
//  パラメタ初期化
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_initializeParameters(handle_t hLayer, handle_t hRandomValueGenerator) {
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)hLayer;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	num_heads = pSelfAttentionNeuralNetHeader->num_heads;
	uint32_t	dim_k = pSelfAttentionNeuralNetHeader->dim_k;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	uint32_t	inHeight = pNeuralNetHeader->inHeight;
	uint32_t	inWidth=pNeuralNetHeader->inWidth;
	uint32_t	inChannel=pNeuralNetHeader->inChannel;
	uint32_t	seq_len;
	uint32_t	dim;
	flt32_t*	pWQ;
	flt32_t*	pWK;
	flt32_t*	pWV;
	flt32_t*	pWO;
	uint32_t*	pLayerParam;
	uint32_t	paramSize;
	uint32_t	normSize;
	//---------------------------------------------------------------------------------
	//アテンション領域の選択
	//---------------------------------------------------------------------------------
	if (FALSE == NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, NULL)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam = (uint32_t*)pSelfAttentionNeuralNetHeader;
	pLayerParam += size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	pWQ = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWK = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	pWV = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim * dim_v, uint32_t);
	pWO = (flt32_t*)pLayerParam;
	pLayerParam += size_in_type(sizeof(flt32_t) * dim_v * dim_v, uint32_t);
	//---------------------------------------------------------------------------------
	//層パラメタ初期化
	//---------------------------------------------------------------------------------
	//WQ
	paramSize = dim * dim_k;
	normSize = dim;
	set_random_initial_values_by_he(hRandomValueGenerator, pWQ, paramSize, normSize);
	//WK
	paramSize = dim * dim_k;
	normSize = dim;
	set_random_initial_values_by_he(hRandomValueGenerator, pWK, paramSize, normSize);
	//WV
	paramSize = dim * dim_v;
	normSize = dim;
	set_random_initial_values_by_he(hRandomValueGenerator, pWV, paramSize, normSize);
	//WO
	if (num_heads > 1) {
		paramSize = dim_v * dim_v;
		normSize = dim_v;
		set_random_initial_values_by_he(hRandomValueGenerator, pWO, paramSize, normSize);
	}
	return TRUE;
}

//=====================================================================================
//  層情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_getLayerInformation(
	uint32_t*	pLayerData,						// in:image data
	bool_t		fEnableLearning,
	uint32_t*	pLayerObjectSizeIn32BitWord,
	uint32_t*	pNumberOfLearningParameters,
	uint32_t*	pTempWorkAreaSizeIn32BitWord,
	DataShape*	pInputShape,
	DataShape*	pOutputShape) {
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	num_heads = pSelfAttentionNeuralNetHeader->num_heads;
	uint32_t	dim_k = pSelfAttentionNeuralNetHeader->dim_k;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	uint32_t	inHeight = pNeuralNetHeader->inHeight;
	uint32_t	inWidth = pNeuralNetHeader->inWidth;
	uint32_t	inChannel = pNeuralNetHeader->inChannel;
	uint32_t	seq_len;
	uint32_t	dim;
	DataShape	outputShape;
	//---------------------------------------------------------------------------------
	//アテンション領域の選択/出力形状の取得
	//---------------------------------------------------------------------------------
	if (FALSE == NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, &outputShape)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (num_heads == 0 || dim_k == 0 || dim_v == 0) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//multi-head attention moduleの分割次元数
	//---------------------------------------------------------------------------------
	uint32_t	dim_k_h = dim_k / num_heads;
	uint32_t	dim_v_h = dim_v / num_heads;
	//---------------------------------------------------------------------------------
	//アテンションモジュールのサイズ取得
	//---------------------------------------------------------------------------------
	uint32_t	moduleSizeIn32BitWord = 0;
	uint32_t	moduleTempWorkAreaSizeIn32BitWord = 0;
	NeuralNetAttentionModule_getSize(seq_len, dim_k_h, dim_v_h, fEnableLearning, &moduleSizeIn32BitWord, &moduleTempWorkAreaSizeIn32BitWord);
	//---------------------------------------------------------------------------------
	//学習パラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfLearningParameters != NULL) {
		*pNumberOfLearningParameters =  dim * dim_k;	// WQ
		*pNumberOfLearningParameters += dim * dim_k;	// WK
		*pNumberOfLearningParameters += dim * dim_v;	// WV
		if (num_heads > 1) {
			*pNumberOfLearningParameters += dim_v * dim_v;	// WO
		}
	}
	//---------------------------------------------------------------------------------
	//オブジェクトサイズ&入力データ
	//---------------------------------------------------------------------------------
	if (pLayerObjectSizeIn32BitWord != NULL) {
		*pLayerObjectSizeIn32BitWord = size_in_type(sizeof(SelfAttentionNeuralNetLayer), uint32_t);
		*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(handle_t) * num_heads, uint32_t);
		*pLayerObjectSizeIn32BitWord += moduleSizeIn32BitWord * num_heads;
		if (fEnableLearning == TRUE) {
			*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim, uint32_t);	//　X
			if (num_heads > 1) {
				*pLayerObjectSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);	//　Vout
			}
		}
	}
	//---------------------------------------------------------------------------------
	//層内部の計算バッファーサイズ
	//---------------------------------------------------------------------------------
	if (pTempWorkAreaSizeIn32BitWord != NULL) {
		*pTempWorkAreaSizeIn32BitWord =  size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　Q
		*pTempWorkAreaSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_k, uint32_t);		//　K
		*pTempWorkAreaSizeIn32BitWord += size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);		//　V
		*pTempWorkAreaSizeIn32BitWord += moduleTempWorkAreaSizeIn32BitWord;	//	attention module共用
		if (num_heads > 1) {
			uint32_t dim_max = max(dim_k, dim_v);
			*pTempWorkAreaSizeIn32BitWord += seq_len * dim_max;
		}
	}
	//---------------------------------------------------------------------------------
	//入出力形状
	//---------------------------------------------------------------------------------
	if (pInputShape != NULL) {
		DataShape_construct(pInputShape, inHeight, inWidth, inChannel);
	}
	if (pOutputShape != NULL) {
		*pOutputShape = outputShape;
	}
	return TRUE;
}

//=====================================================================================
//  学習パラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_getParameters(handle_t hLayer, flt32_t** ppParameters, uint32_t* pNumberOfParameters) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	uint32_t* pLayerParam = (uint32_t*)pNeuralNetLayer->pLayerData;
	//---------------------------------------------------------------------------------
	//層パラメタ
	//---------------------------------------------------------------------------------
	pLayerParam += size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	if (ppParameters != NULL) {
		*ppParameters = (flt32_t*)pLayerParam;
	}
	if (pNumberOfParameters != NULL) {
		NeuralNetLayerSelfAttention_getLayerInformation(pNeuralNetLayer->pLayerData, TRUE, NULL, pNumberOfParameters, NULL, NULL, NULL);
	}
	return TRUE;
}

//=====================================================================================
//  ハイパーパラメタ情報取得
//=====================================================================================
static
bool_t
NeuralNetLayerSelfAttention_getHyperParameters(handle_t hLayer, flt32_t* pParameterArray, uint32_t* pNumberOfParameters, uint32_t parameterArraySize) {
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)hLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pNeuralNetLayer->pLayerData;
	if (pSelfAttentionNeuralNetHeader == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//ハイパーパラメタ数
	//---------------------------------------------------------------------------------
	if (pNumberOfParameters != NULL) {
		*pNumberOfParameters = 4;
	}
	//---------------------------------------------------------------------------------
	//ハイパーパラメタ内容
	//---------------------------------------------------------------------------------
	if (pParameterArray != NULL) {
		if (parameterArraySize >= 4) {
			pParameterArray[0] = (flt32_t)pSelfAttentionNeuralNetHeader->mode;
			pParameterArray[1] = (flt32_t)pSelfAttentionNeuralNetHeader->num_heads;
			pParameterArray[2] = (flt32_t)pSelfAttentionNeuralNetHeader->dim_k;
			pParameterArray[3] = (flt32_t)pSelfAttentionNeuralNetHeader->dim_v;
		}
		else {
			return FALSE;
		}
	}
	return TRUE;
}

//=====================================================================================
//  層構築
//=====================================================================================
static
handle_t
NeuralNetLayerSelfAttention_construct(
	uint32_t*	pLayerData,
	uint32_t*	pObjectWork,
	uint32_t	sizeObjectIn32BitWord,
	bool_t		fEnableLearning,
	handle_t	hOptimizer) {
	SelfAttentionNeuralNetLayer* pSelfAttentionLayer = (SelfAttentionNeuralNetLayer*)pObjectWork;
	NeuralNetLayer* pNeuralNetLayer = (NeuralNetLayer*)pSelfAttentionLayer;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pLayerData;
	NeuralNetHeader* pNeuralNetHeader = (NeuralNetHeader*)pSelfAttentionNeuralNetHeader;
	LayerFuncTable	funcTable;
	uint32_t	seq_len = 0;
	uint32_t	dim = 0;
	uint32_t	num_heads = pSelfAttentionNeuralNetHeader->num_heads;
	uint32_t	dim_k = pSelfAttentionNeuralNetHeader->dim_k;
	uint32_t	dim_v = pSelfAttentionNeuralNetHeader->dim_v;
	SelfAttentionMode	mode = pSelfAttentionNeuralNetHeader->mode;
	uint32_t	inHeight = pNeuralNetHeader->inHeight;
	uint32_t	inWidth = pNeuralNetHeader->inWidth;
	uint32_t	inChannel = pNeuralNetHeader->inChannel;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (num_heads == 0 || dim_k == 0 || dim_v == 0) {
		return NULL;
	}
	//---------------------------------------------------------------------------------
	//multi-head attention moduleの分割次元数
	//---------------------------------------------------------------------------------
	uint32_t	dim_k_h = dim_k / num_heads;
	uint32_t	dim_v_h = dim_v / num_heads;
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	uint32_t	requiredSize = 0;
	uint32_t	numberOfLearningParameters = 0;
	uint32_t	parameterSize;
	NeuralNetLayerSelfAttention_getLayerInformation(pLayerData, fEnableLearning, &requiredSize, &numberOfLearningParameters, NULL, NULL, NULL);
	if (pObjectWork == NULL || sizeObjectIn32BitWord < requiredSize) {
		return NULL;
	}
	//---------------------------------------------------------------------------------
	//アテンション領域の選択
	//---------------------------------------------------------------------------------
	NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, NULL);
	//---------------------------------------------------------------------------------
	//層インターフェイス取得
	//---------------------------------------------------------------------------------
	NeuralNetLayerSelfAttention_getInterface(&funcTable);
	//---------------------------------------------------------------------------------
	//層構造体構築
	//---------------------------------------------------------------------------------
	NeuralNetLayer_construct(pNeuralNetLayer, pLayerData, fEnableLearning, funcTable, 0);
	//---------------------------------------------------------------------------------
	// アテンションモジュールサイズ取得
	//---------------------------------------------------------------------------------
	uint32_t	moduleSizeIn32BitWord = 0;
	uint32_t	moduleTempWorkAreaSizeIn32BitWord = 0;
	NeuralNetAttentionModule_getSize(seq_len, dim_k_h, dim_v_h, fEnableLearning, &moduleSizeIn32BitWord, &moduleTempWorkAreaSizeIn32BitWord);
	//---------------------------------------------------------------------------------
	//学習用データおよび最適化アルゴリズムオブジェクトハンドル
	//---------------------------------------------------------------------------------
	//module
	pObjectWork += size_in_type(sizeof(SelfAttentionNeuralNetLayer), uint32_t);
	sizeObjectIn32BitWord -= size_in_type(sizeof(SelfAttentionNeuralNetLayer), uint32_t);
	//学習用パラメタ
	if (fEnableLearning == TRUE) {
		// 学習パラメタサイズチェック
		OptimizerFunctionTable	optimizerInterface;
		NeuralNetOptimizer_getInterface(hOptimizer, &optimizerInterface);
		parameterSize = optimizerInterface.pGetParameterSize(hOptimizer);
		if (parameterSize < numberOfLearningParameters) {
			return NULL;
		}
		//逆伝播の際に必要な値を保持するバッファ
		//X
		pSelfAttentionLayer->pX = (flt32_t*)pObjectWork;
		pObjectWork				+= size_in_type(sizeof(flt32_t) * seq_len * dim, uint32_t);
		sizeObjectIn32BitWord	-= size_in_type(sizeof(flt32_t) * seq_len * dim, uint32_t);
		//Vout
		if (num_heads > 1) {
			pSelfAttentionLayer->pVxWO_in = (flt32_t*)pObjectWork;
			pObjectWork				+= size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);
			sizeObjectIn32BitWord	-= size_in_type(sizeof(flt32_t) * seq_len * dim_v, uint32_t);
		}
		//optimizer
		pSelfAttentionLayer->hOptimizer = hOptimizer;
	}
	else {
		pSelfAttentionLayer->pX			= NULL;
		pSelfAttentionLayer->pVxWO_in	= NULL;
		pSelfAttentionLayer->hOptimizer = NULL;
	}
	//---------------------------------------------------------------------------------
	//attention module構築（次元はマルチヘッドの次元）
	//---------------------------------------------------------------------------------
	pSelfAttentionLayer->phAttention = (handle_t*)pObjectWork;
	pObjectWork				+= size_in_type(sizeof(handle_t) * num_heads, uint32_t);
	sizeObjectIn32BitWord	-= size_in_type(sizeof(handle_t) * num_heads, uint32_t);
	uint32_t  i = 0;
	for (i = 0; i < num_heads; i++) {
		pSelfAttentionLayer->phAttention[i] = NeuralNetAttentionModule_construct(seq_len, dim_k_h, dim_v_h, fEnableLearning, pObjectWork, sizeObjectIn32BitWord);
		if (pSelfAttentionLayer->phAttention[i] == NULL) {
			return NULL;
		}
		pObjectWork				+= moduleSizeIn32BitWord;
		sizeObjectIn32BitWord	-= moduleSizeIn32BitWord;
	}
	return (handle_t)pSelfAttentionLayer;
}

//=====================================================================================
//  インターフェース取得
//=====================================================================================
void
NeuralNetLayerSelfAttention_getInterface(LayerFuncTable* pInterface) {
	pInterface->pGetLayerInformation = NeuralNetLayerSelfAttention_getLayerInformation;
	pInterface->pConstruct = NeuralNetLayerSelfAttention_construct;
	pInterface->pGetShape = NeuralNetLayerSelfAttention_getShape;
	pInterface->pForward = NeuralNetLayerSelfAttention_forward;
	pInterface->pBackward = NeuralNetLayerSelfAttention_backward;
	pInterface->pUpdate = NeuralNetLayerSelfAttention_update;
	pInterface->pInitializeParameters = NeuralNetLayerSelfAttention_initializeParameters;
	pInterface->pGetParameters = NeuralNetLayerSelfAttention_getParameters;
	pInterface->pGetHyperParameters = NeuralNetLayerSelfAttention_getHyperParameters;
}

//=====================================================================================
//  層作成
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
	uint32_t*			pSizeOfLayerIn32BitWord)
{
	uint32_t	seq_len = 0;
	uint32_t	dim		= 0;
	DataShape	outputShape;
	SelfAttentionNeuralNetHeader* pSelfAttentionNeuralNetHeader;
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (pInputHeight == NULL || pInputWidth == NULL || pInputChannel == NULL) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//パラメタ
	//---------------------------------------------------------------------------------
	uint32_t	inHeight	= *pInputHeight;
	uint32_t	inWidth		= *pInputWidth;
	uint32_t	inChannel	= *pInputChannel;
	//---------------------------------------------------------------------------------
	//アテンション形状関連パラメタ
	//---------------------------------------------------------------------------------
	switch (mode) {
	case SELF_ATTENTION_MODE_CHANNEL:		// チャネル方向 (C) を対象としたチャンネルアテンション
		//dim_vに0を指定した場合、出力形状を入力形状と同一にする
		if (dim_v == 0) {
			dim_v = inHeight * inWidth;
		}
		//dim_kパラメタチェック
		if (dim_k == 0) {
			return FALSE;
		}
		break;
	default:
		//dim_kパラメタチェック
		if (dim_k == 0) {
			return FALSE;
		}
		break;
	}
	//---------------------------------------------------------------------------------
	//パラメタチェック
	//---------------------------------------------------------------------------------
	if (num_heads == 0) {
		return FALSE;
	}
	if ((dim_k % num_heads) != 0) {	//マルチヘッドの場合、num_headsで割り切れる必要がある
		return FALSE;
	}
	if ((dim_v % num_heads) != 0) {	//マルチヘッドの場合、num_headsで割り切れる必要がある
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//アテンション領域の選択/出力形状の取得
	//---------------------------------------------------------------------------------
	if (FALSE == NeuralNetLayerSelfAttention_getAttentionMatrixShape(mode, inHeight, inWidth, inChannel, dim_v, &seq_len, &dim, &outputShape)) {
		return FALSE;
	}
	//---------------------------------------------------------------------------------
	//層サイズ
	//---------------------------------------------------------------------------------
	uint32_t	sizeHeader = size_in_type(sizeof(SelfAttentionNeuralNetHeader), uint32_t);
	uint32_t	sizeOfParamWQ = size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	uint32_t	sizeOfParamWK = size_in_type(sizeof(flt32_t) * dim * dim_k, uint32_t);
	uint32_t	sizeOfParamWV = size_in_type(sizeof(flt32_t) * dim * dim_v, uint32_t);
	uint32_t	sizeOfParamWO = 0;
	if (num_heads > 1) {
		sizeOfParamWO = size_in_type(sizeof(flt32_t) * dim_v * dim_v, uint32_t);
	}
	uint32_t	sizeLayer = sizeHeader + sizeOfParamWQ + sizeOfParamWK + sizeOfParamWV + sizeOfParamWO;
	if (pSizeOfLayerIn32BitWord != NULL) {
		*pSizeOfLayerIn32BitWord = sizeLayer;
	}
	//---------------------------------------------------------------------------------
	//層データ構築
	//---------------------------------------------------------------------------------
	if (pBuffer != NULL) {
		//サイズチェック
		if (sizeOfBufferIn32BitWord < sizeLayer) {
			return FALSE;
		}
		//バッファーの先頭をセット
		uint32_t* pLayer = pBuffer;
		//header
		pSelfAttentionNeuralNetHeader = (SelfAttentionNeuralNetHeader*)pLayer;
		NeuralNetHeader_construct(&pSelfAttentionNeuralNetHeader->super, NET_LAYER_SELF_ATTENTION, inHeight, inWidth, inChannel, sizeLayer);
		pSelfAttentionNeuralNetHeader->mode			= mode;
		pSelfAttentionNeuralNetHeader->num_heads	= num_heads;
		pSelfAttentionNeuralNetHeader->dim_k		= dim_k;
		pSelfAttentionNeuralNetHeader->dim_v		= dim_v;
		pLayer += sizeHeader;
		//WQ
		pLayer += sizeOfParamWQ;
		//WK
		pLayer += sizeOfParamWK;
		//WV
		pLayer += sizeOfParamWV;
		//WO
		pLayer += sizeOfParamWO;
	}
	//---------------------------------------------------------------------------------
	//出力次元
	//---------------------------------------------------------------------------------
	*pInputHeight	= outputShape.height;
	*pInputWidth	= outputShape.width;
	*pInputChannel	= outputShape.channel;
	return TRUE;
}
