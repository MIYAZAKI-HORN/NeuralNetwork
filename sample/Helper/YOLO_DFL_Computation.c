#include "NeuralNetLayerFunction.h"
#include "YOLO_DFL_Computation.h"

#define EPSILON		(1.0e-7f)
#define M_PI		(3.14159265358979323846f)
#define M_PI_2		(1.57079632679489661923f)
#define LN2_F 		(0.6931471805599453f)

#ifndef max
	#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
	#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

//==========================================================================
// memcpy
//==========================================================================
static
void
self_memcpy(void* pd, void* ps, uint32_t size) {
	unsigned char* d	= (unsigned char*)pd;
	unsigned char* s	= (unsigned char*)ps;
	while ( size > 0) {
		*d++ = *s++;
		size--;
	}
}

//==========================================================================
//logf近似計算
//==========================================================================
static
flt32_t 
fast_logf(flt32_t x) {
	//-----------------------------------------------------
	//エラーハンドリング
	//-----------------------------------------------------
	if (x <= EPSILON) {
		x = EPSILON;
	}
	//-----------------------------------------------------
	// ビット抽出による指数部 (e) と仮数部 (m) の分解
	//-----------------------------------------------------
	uint32_t ix;
	self_memcpy(&ix, &x, sizeof(flt32_t));
	int32_t e = (int32_t)((ix >> 23) & 0xFF) - 127;
	ix = (ix & 0x007FFFFF) | 0x3F800000; // 仮数部を [1.0, 2.0] にスケーリング
	flt32_t m;
	self_memcpy(&m, &ix, sizeof(flt32_t));
	//-----------------------------------------------------
	// 範囲を [1/sqrt(2), sqrt(2)] 付近に調整
	//-----------------------------------------------------
	if (m > 1.41421356237f) {
		m *= 0.5f;
		e += 1;
	}
	//-----------------------------------------------------
	// f = m - 1.0 における ln(1 + f) の Remez 5次多項式近似
	//-----------------------------------------------------
	flt32_t f = m - 1.0f;
	//-----------------------------------------------------
	// ホーナー法による多項式評価: f * (p0 + f * (p1 + f * (p2 + f * (p3 + f * p4))))
	// minimax 係数
	//-----------------------------------------------------
	flt32_t p0 = 0.99999994f;
	flt32_t p1 = -0.49999653f;
	flt32_t p2 = 0.33329022f;
	flt32_t p3 = -0.24867162f;
	flt32_t p4 = 0.19448152f;
	flt32_t poly = f * (p0 + f * (p1 + f * (p2 + f * (p3 + f * p4))));
	//-----------------------------------------------------
	// ln(x) = poly + e * ln(2)
	//-----------------------------------------------------
	return poly + ((flt32_t)e * LN2_F);
}

//==========================================================================
//atan近似計算
//==========================================================================
static
flt32_t 
fast_atan2f(flt32_t y, flt32_t x) {
	//-----------------------------------------------------
	//エラーハンドリング
	//-----------------------------------------------------
	if (x == 0.0f && y == 0.0f) {
		return 0.0f;
	}
	//-----------------------------------------------------
	//絶対値をとる
	//-----------------------------------------------------
	flt32_t abs_y = (y > 0.0f) ? y : -y;
	flt32_t abs_x = (x > 0.0f) ? x : -x;
	//-----------------------------------------------------
	// t が [0, 1] の範囲に入るようにスケーリング
	//-----------------------------------------------------
	flt32_t t = (abs_x > abs_y) ? (abs_y / abs_x) : (abs_x / abs_y);
	//-----------------------------------------------------
	// それほどの精度は必要ないと思われるので、簡単な多項式: atan(t) ≈ t * (0.972394f - 0.191947f * t * t)を利用する
	//-----------------------------------------------------
	flt32_t t2 = t * t;
	flt32_t angle = t * (0.972394f - 0.191947f * t2);
	//-----------------------------------------------------
	// |y| > |x| の場合は pi/2 から引く
	//-----------------------------------------------------
	if (abs_y > abs_x) {
		angle = M_PI_2 - angle;
	}
	//-----------------------------------------------------
	// 象限に応じて角度を補正
	//-----------------------------------------------------
	if (x < 0.0f) {
		angle = M_PI - angle;
	}
	if (y < 0.0f) {
		angle = -angle;
	}
	return angle;
}

//==========================================================================
// Focal Loss とその勾配を返す関数 one hot想定
//==========================================================================
static
void
compute_focal_loss(
	uint32_t	dim,				//入力次元
	flt32_t*	pTeacherArray,		//教師データ配列
	flt32_t*	pPredictedArray,	//予測値配列
	flt32_t		alpha,				//α係数
	flt32_t		lamda,				//損失関数に乗算するハイパーパラメタ
	flt32_t*	pFL,				//損失値
	flt32_t*	pDFLArray)			//損失微分値
{		
	uint32_t	i;
	//-------------------------------------
	// Sigmoid で個別確率を計算
	//-------------------------------------
	sigmoid_forward(pPredictedArray, pPredictedArray, dim);
	//-------------------------------------
	// 数値安定性のための eps クリップ
	//-------------------------------------
	for (i = 0; i < dim; i++) {
		if (pPredictedArray[i] < EPSILON) {
			pPredictedArray[i] = EPSILON;
		}
		if (pPredictedArray[i] > (1.0f - EPSILON)) {
			pPredictedArray[i] = 1.0f - EPSILON;
		}
	}
	//-------------------------------------
	// FL計算
	// FL微分値計算(sigmoidも含まれている)
	// 組み込みを考慮して　gamma = 2.0f　で固定
	//-------------------------------------
	flt32_t FL = 0.0f;
	for (i = 0; i < dim; i++) {
		//-----------------------------------------------------
		// 損失値と損失値微分
		//-----------------------------------------------------
		flt32_t dFL;
		flt32_t p = pPredictedArray[i];
		if (pTeacherArray[i] == 1.0f) {
			flt32_t log_p = fast_logf(p);					// logf(p)
			flt32_t pow_1minusp = (1.0f - p) * (1.0f - p);	// powf(1.0f - p, gamma)
			FL += -1.0f * alpha * pow_1minusp * log_p;
			dFL = alpha * pow_1minusp * (2.0f * p * log_p + p - 1.0f);
		}
		else {
			flt32_t log_1minusp = fast_logf(1.0f - p);
			flt32_t pow_p = p * p;	// powf(p, gamma)
			FL += -1.0f * (1.0f - alpha) * pow_p * log_1minusp;
			dFL = (1.0f - alpha) * pow_p * (1.0f - p - 2.0f * (1.0f - p) * log_1minusp);
		}
		//-----------------------------------------------------
		// 損失微分値の引き渡し
		//-----------------------------------------------------
		if (pDFLArray != NULL) {
			pDFLArray[i] = lamda * dFL;
		}
	}
	//-------------------------------------
	// 損失値引き渡し
	//-------------------------------------
	*pFL = FL * lamda;
}

//==========================================================================
// Distribution Focal Loss (DFL) の損失値と微分値を算出する関数
//==========================================================================
static
void
compute_dfl_loss(	flt32_t*	probs,					//確率配列
					flt32_t		target,					//学習正解値（バウンディングボックス一辺への距離：グリッドサイズベース）
					int32_t		distributionMapSize,	//
					flt32_t		lamda_dfl,
					flt32_t*	pDFLLoss,
					flt32_t*	pDDFLossArray) {
	//-------------------------------------
	// 正解値がある(y_i) とその隣(y_i1) のインデックスを取得
	// インデックス0は、中心があるセルの隣接したセルとする
	// 中心があるセルの側面は、targetが0.5となる
	//-------------------------------------
	// ターゲット座標の範囲チェック
	// 上記したようにtargetの最小値は必ず0.5
	// 中心があるセルのエッジをマップの0起点にする
	//-------------------------------------
	target -= 0.5f;
	if (target < 0.0f) {
		target = 0.0f;
	}
	int32_t y_i = (int32_t)target;
	//隣接した二点で確率計算するため以下の制限を与える
	if (y_i > (distributionMapSize - 2)) {
		y_i = distributionMapSize - 2;
	}
	int32_t y_i1 = y_i + 1;
	//-------------------------------------
	// 教師の計算 
	// これがteacher確率　つまりdistributionMapSize個の確率分布の内から近傍2個の確率を拾って、これと教師2個の確率のクロスエントロピーを計算する
	// logitsは上下左右4次元文あり、それぞれ原点がグリッドセルと重なっていると考えればよい
	// box上下左右に向かうベクトル
	//-------------------------------------
	flt32_t t_prob_i	= 1.0f - (target - (flt32_t)y_i);	// 距離が小さいほど確率が高い
	flt32_t t_prob_i1	= 1.0f - t_prob_i;					// 2個なので、確率の残り
	//-------------------------------------
	// Cross Entropy計算
	//-------------------------------------
	flt32_t loss_i	= t_prob_i  * fast_logf(probs[y_i] + EPSILON);		// logf(probs[y_i] + EPSILON);
	flt32_t loss_i1 = t_prob_i1 * fast_logf(probs[y_i1] + EPSILON);		// logf(probs[y_i1] + EPSILON);
	flt32_t loss	= -1.0f * (loss_i + loss_i1);
	//-------------------------------------
	// 損失値
	//-------------------------------------
	*pDFLLoss = loss * lamda_dfl;
	//-------------------------------------
	// soft max & cross entropyの微分
	//-------------------------------------
	for (int32_t i = 0; i < distributionMapSize; i++) {
		flt32_t target_prob;
		if (i == y_i) {
			//box端
			target_prob = t_prob_i;
		}
		else if (i == y_i1) {
			//box端から一つ離れた位置
			target_prob = t_prob_i1;
		}
		else {
			//それ以外
			target_prob = 0.0f;
		}
		pDDFLossArray[i] = lamda_dfl * (probs[i]  - target_prob);
	}
}

//==========================================================================
// IoU値を取得する
//==========================================================================
static
void
compute_ciou_loss(	flt32_t*	pTrueBox, 
					flt32_t*	pPredectedBox,
					int32_t		gridSize,
					int32_t		distributionMapSize,
					flt32_t		lamda_ciou,
					flt32_t*	pIoU,
					flt32_t*	pLossBox,
					flt32_t*	pDLossBox) {
	int32_t i;
	flt32_t IoU;
	flt32_t tltcx = pTrueBox[0];	//ltcx差分　グリッドサイズベース
	flt32_t tltcy = pTrueBox[1];	//ltcy差分
	flt32_t trbcx = pTrueBox[2];	//rbcx差分
	flt32_t trbcy = pTrueBox[3];	//rbcy差分
	//-------------------------------------
	//予測box位置　確率計算で得る
	// ボックスの中心があるセルの隣をインデックス0とする
	// すなわち位置は、i=0　が0.5とする
	//-------------------------------------
	flt32_t pltcx = 0.0f;
	flt32_t pltcy = 0.0f;
	flt32_t prbcx = 0.0f;
	flt32_t prbcy = 0.0f;
	flt32_t* pProbs = pPredectedBox;
	// ltcx 期待値: Σ (p_i * (i+0.5))
	for (i = 0; i < distributionMapSize; i++) {
		pltcx += (*pProbs++ * ((flt32_t)i + 0.5f));
	}
	// ltcy 期待値: Σ (p_i * (i+0.5))
	for (i = 0; i < distributionMapSize; i++) {
		pltcy += (*pProbs++ * ((flt32_t)i + 0.5f));
	}
	// rbcx 期待値: Σ (p_i * (i+0.5))
	for (i = 0; i < distributionMapSize; i++) {
		prbcx += (*pProbs++ * ((flt32_t)i + 0.5f));
	}
	// rbcy 期待値: Σ (p_i * (i+0.5))
	for (i = 0; i < distributionMapSize; i++) {
		prbcy += (*pProbs++ * ((flt32_t)i + 0.5f));
	}
	//-------------------------------------
	//box位置座標変換:　left corner（tltc,pltc）は中心から離れる方向
	//-------------------------------------
	flt32_t t_ltx = -1.0f * tltcx;
	flt32_t t_lty = -1.0f * tltcy;
	flt32_t t_rbx = trbcx;
	flt32_t t_rby = trbcy;
	flt32_t p_ltx = -1.0f * pltcx;
	flt32_t p_lty = -1.0f * pltcy;
	flt32_t p_rbx = prbcx;
	flt32_t p_rby = prbcy;
	//-------------------------------------
	//中止位置
	//-------------------------------------
	flt32_t	tx = (t_ltx + t_rbx) * 0.5f;
	flt32_t	ty = (t_lty + t_rby) * 0.5f;
	flt32_t	px = (p_ltx + p_rbx) * 0.5f;
	flt32_t	py = (p_lty + p_rby) * 0.5f;
	//-------------------------------------
	//幅と高さ
	//-------------------------------------
	flt32_t	tw = (t_rbx - t_ltx);
	flt32_t	th = (t_rby - t_lty);
	flt32_t	pw = (p_rbx - p_ltx);
	flt32_t	ph = (p_rby - p_lty);
	if (pw < 0.0f) {
		pw = 0.0f;
	}
	if (ph < 0.0f) {
		ph = 0.0f;
	}
	//========================================================================================
	// 
	//　IoU計算
	// 
	//========================================================================================
	//union　インターセクション矩形
	flt32_t cross_ltcx = max(t_ltx, p_ltx);
	flt32_t cross_ltcy = max(t_lty, p_lty);
	flt32_t cross_rbcx = min(t_rbx, p_rbx);
	flt32_t cross_rbcy = min(t_rby, p_rby);
	//-------------------------------------
	//　IoU計算
	//-------------------------------------
	flt32_t cross_width		= cross_rbcx - cross_ltcx;	//マイナスだと交わってない
	flt32_t cross_height	= cross_rbcy - cross_ltcy;	//マイナスだと交わってない
	if (cross_width > 0.0f && cross_height > 0.0f) {
		flt32_t	intersection = cross_width * cross_height;
		flt32_t	unionRegion = (tw * th + pw * ph) - intersection;
		IoU = intersection / unionRegion;
	}
	else {
		IoU = 0.0f;
	}
	if (IoU < 0.0f) {
		IoU = 0.0f;
	}
	//========================================================================================
	// 
	// 損失計算
	// 
	//========================================================================================
	//-------------------------------------
	//box差分
	//-------------------------------------
	flt32_t diff_x = px - tx;
	flt32_t diff_y = py - ty;
	//-------------------------------------
	//最小外接矩形
	//-------------------------------------
	flt32_t C_ltcx = min(t_ltx, p_ltx);
	flt32_t C_ltcy = min(t_lty, p_lty);
	flt32_t C_rbcx = max(t_rbx, p_rbx);
	flt32_t C_rbcy = max(t_rby, p_rby);
	//-------------------------------------
	//最小外接矩形サイズ
	//-------------------------------------
	flt32_t Cw = C_rbcx - C_ltcx;
	flt32_t Ch = C_rbcy - C_ltcy;
	//-------------------------------------
	//  rho^2, c^2 を計算
	//-------------------------------------
	flt32_t rho2 = diff_x * diff_x + diff_y * diff_y;
	flt32_t c2 = Cw * Cw + Ch * Ch + EPSILON; // ゼロ割防止を考慮
	//-------------------------------------
	// DIoU Loss
	//-------------------------------------
	flt32_t loss_dIoU = 1.0f - IoU + (rho2 / c2);
	//---------------------------------------------------
	// CIoU:DIoUに追加された部分を計算
	//---------------------------------------------------
	//---------------------------------------------------
	// v と alpha の計算
	//---------------------------------------------------
	flt32_t arctan_gt	= fast_atan2f(tw, th);	// atan2f(tw, th);
	flt32_t arctan_pred = fast_atan2f(pw, ph);	// atan2f(pw, ph);
	flt32_t diff = arctan_gt - arctan_pred;
	flt32_t v = (4.0f / (M_PI * M_PI)) * diff * diff;
	flt32_t alpha = v / ((1.0f - IoU) + v + EPSILON);
	//---------------------------------------------------
	// DIoUに追加された部分の損失
	//---------------------------------------------------
	flt32_t loss_CIoU_extra = alpha * v;
	//---------------------------------------------------
	// CIoU loss
	//---------------------------------------------------
	flt32_t loss_CIoU = loss_dIoU + loss_CIoU_extra;
	//========================================================================================
	// 
	// 損失の微分計算
	// 
	//========================================================================================
	// ボックス座標(x, y, w, h)に対する勾配 delta の計算
	// IoU 部分(1 - IoU)の各パラメータに対する微分は、直感的な誤差ベクトルで近似
	// 中心位置(x, y)のIoU 勾配:重なりを増やす方向へ動かすため、単純な差分に IoU のスケールを掛ける
	//---------------------------------------------------
	// DIoU Lossの微分値を計算 
	// pltcxとpltcyには-1をかけて利用している
	//---------------------------------------------------
	//pltcxの微分
	flt32_t delta_ltx_IoU	= -1.0f * 0.5f * diff_x;				// IoU側の引き戻し
	flt32_t delta_ltx_dist	= -1.0f * 0.5f * 2.0f * diff_x / c2;	// DIoUの中心距離ペナルティ
	flt32_t dLoss_dltx		= delta_ltx_IoU + delta_ltx_dist;
	//pltcyの微分
	flt32_t delta_lty_IoU	= -1.0f * 0.5f * diff_y;
	flt32_t delta_lty_dist	= -1.0f * 0.5f * 2.0f * diff_y / c2;
	flt32_t dLoss_dlty		= delta_lty_IoU + delta_lty_dist;
	//prbcxの微分
	flt32_t delta_rbx_IoU	= 0.5f * diff_x;				// IoU側の引き戻し
	flt32_t delta_rbx_dist	= 0.5f * 2.0f * diff_x / c2;	// DIoUの中心距離ペナルティ
	flt32_t dLoss_drbx		= delta_rbx_IoU + delta_rbx_dist;
	//prbcyの微分
	flt32_t delta_rby_IoU	= 0.5f * diff_y;
	flt32_t delta_rby_dist	= 0.5f * 2.0f * diff_y / c2;
	flt32_t dLoss_drby		= delta_rby_IoU + delta_rby_dist;
	//---------------------------------------------------
	// CIoU:DIoUに追加された部分
	//---------------------------------------------------
	flt32_t dLoss_dltx_ciou_extra;
	flt32_t dLoss_dlty_ciou_extra;
	flt32_t dLoss_drbx_ciou_extra;
	flt32_t dLoss_drby_ciou_extra;
	//---------------------------------------------------
	// DIoUの勾配の追加分計算
	//---------------------------------------------------
	flt32_t v_grad_factor = (8.0f / (M_PI * M_PI)) * (arctan_pred - arctan_gt) / (pw * pw + ph * ph + EPSILON);
	flt32_t dLoss_dw_ciou_extra = alpha * v_grad_factor * ph;
	flt32_t dLoss_dh_ciou_extra = alpha * (-v_grad_factor) * pw;
	//
	dLoss_dltx_ciou_extra	= dLoss_dw_ciou_extra;	// -1.0かけなくてよい　幅の計算では、結果的に、ltc（-x方向）とrbcの足し算になるため
	dLoss_dlty_ciou_extra	= dLoss_dh_ciou_extra;
	dLoss_drbx_ciou_extra	= dLoss_dw_ciou_extra;
	dLoss_drby_ciou_extra	= dLoss_dh_ciou_extra;
	//---------------------------------------------------
	// CIoU w, h の勾配を追加
	//---------------------------------------------------
	dLoss_dltx += dLoss_dltx_ciou_extra;
	dLoss_dlty += dLoss_dlty_ciou_extra;
	dLoss_drbx += dLoss_drbx_ciou_extra;
	dLoss_drby += dLoss_drby_ciou_extra;
	//-------------------------------------
	//ファクターを乗算する
	//-------------------------------------
	//教師boxサイズ由来のファクターを考慮する
	flt32_t tw_g = tw / (flt32_t)gridSize;
	flt32_t th_g = th / (flt32_t)gridSize;
	flt32_t lamda_coord_weighted = lamda_ciou * (2.0f - tw_g * th_g);
	//損失に係数を重畳
	loss_CIoU	*= lamda_coord_weighted;
	dLoss_dltx	*= lamda_coord_weighted;
	dLoss_dlty	*= lamda_coord_weighted;
	dLoss_drbx	*= lamda_coord_weighted;
	dLoss_drby	*= lamda_coord_weighted;
	//========================================================================================
	// 
	//　計算値の引き渡し
	// 
	//========================================================================================
	//-------------------------------------
	//IoUをセット
	//-------------------------------------
	if (pIoU != NULL) {
		*pIoU = IoU;
	}
	//-------------------------------------
	//損失をセット
	//-------------------------------------
	if (pLossBox != NULL) {
		*pLossBox = loss_CIoU;
	}
	//-------------------------------------
	//最後にpPredectedBoxの微分を考慮する
	//-------------------------------------
	if (pDLossBox != NULL) {
		//-------------------------------------
		//確率マップからのbox形状予測値の逆伝搬
		//-------------------------------------
		flt32_t* pDLoss = pDLossBox;
		//
		for (i = 0; i < distributionMapSize; i++) {
			*pDLoss++ = dLoss_dltx * ((flt32_t)i + 0.5f);
		}
		for (i = 0; i < distributionMapSize; i++) {
			*pDLoss++ = dLoss_dlty * ((flt32_t)i + 0.5f);
		}
		for (i = 0; i < distributionMapSize; i++) {
			*pDLoss++ = dLoss_drbx * ((flt32_t)i + 0.5f);
		}
		for (i = 0; i < distributionMapSize; i++) {
			*pDLoss++ = dLoss_drby * ((flt32_t)i + 0.5f);
		}
		//-------------------------------------
		//softmax逆伝搬
		//-------------------------------------
		flt32_t* pBoxProb = pPredectedBox;
		pDLoss = pDLossBox;
		for (i = 0; i < 4; i++) {
			softmax_backward(pBoxProb, pDLoss, pDLoss, distributionMapSize);
			pBoxProb += distributionMapSize;
			pDLoss += distributionMapSize;
		}
	}
}

//==========================================================================
// YOLO anchor free DFL 損失値と逆伝搬値を作成する
//==========================================================================
#define MAX_DISTIBUTION_MAP	(16)

bool_t
YORO_DFL_Training(
	int32_t		gridSize,
	int32_t		distributionMapSize,
	int32_t		nClasses,
	flt32_t*	pTeacherArray,
	flt32_t*	pPredictedValueArray,
	flt32_t*	pDLossArray,
	flt32_t		lamda_ciou,
	flt32_t		lamda_dfl,
	flt32_t		lamda_class,
	DFL_Monitor* pMeasuredValues) {
	flt32_t*	pGridTeacher	= pTeacherArray;
	flt32_t*	pGridPrediction = pPredictedValueArray;
	flt32_t*	pGridDLoss		= pDLossArray;
	flt32_t		lossDF			= 0.0f;
	flt32_t		lossCIoU		= 0.0f;
	flt32_t		lossClassProb	= 0.0f;
	flt32_t		lossNoObj		= 0.0f;
	int32_t		lossArraySize = gridSize * gridSize * (distributionMapSize * 4 + nClasses);
	uint32_t	box_count					= 0;
	uint32_t	box_class_corrrect_count	= 0;
	flt32_t		box_accumulated_IoU			= 0.0f;
	static flt32_t	boxProbabilityArray[MAX_DISTIBUTION_MAP*4];
	static flt32_t	boxProbabilityDlossArray[MAX_DISTIBUTION_MAP*4];
	//-------------------------------------------------------
	// 損失微分値初期化
	//-------------------------------------------------------
	if (pDLossArray != NULL) {
		for (int32_t i = 0; i < lossArraySize; i++) {
			pDLossArray[i] = 0.0f;
		}
	}
	//-------------------------------------------------------
	// エラー
	//-------------------------------------------------------
	if (distributionMapSize > MAX_DISTIBUTION_MAP) {
		return FALSE;
	}
	//-------------------------------------------------------
	// 損失と損失の微分値を計算
	//-------------------------------------------------------
	pGridTeacher = pTeacherArray;
	pGridPrediction = pPredictedValueArray;
	pGridDLoss = pDLossArray;
	for (int32_t iGridY = 0; iGridY < gridSize; iGridY++) {
		for (int32_t iGridX = 0; iGridX < gridSize; iGridX++) {
			// グリッド損失値
			flt32_t		lossGridDF		= 0.0f;
			flt32_t		lossGridCIoU	= 0.0f;
			flt32_t		lossGridProb	= 0.0f;
			flt32_t		lossGridNoObj	= 0.0f;
			// 教師データ　セル中心からバウンディングボックス各辺への距離
			flt32_t		ltcx = pGridTeacher[0];	//ltcxへの距離
			//----------------------------------
			// ltcxへの距離が0でない場合は、boxセンターが存在するとする
			//----------------------------------
			if (ltcx != 0.0f) {
				//------------------------------------------
				//このグリッドにオブジェクトのセンターが存在する
				//------------------------------------------
				flt32_t*	pBoxLogits			= pGridPrediction;		//ネットワーク出力生データ
				flt32_t*	pBoxProbabilities	= boxProbabilityArray;	//確率に変換された値を保持する
				//--------------------------------------------
				//activation:box位置（グリッドセンターからのltc(x,y)およびrbc(x,y)の確率マップ）
				//--------------------------------------------
				for (int32_t j = 0; j < 4; j++) {	//グリッド中心からの、上下左右位置差分
					softmax_forward(pBoxLogits, pBoxProbabilities, distributionMapSize);
					pBoxLogits			+= distributionMapSize;
					pBoxProbabilities	+= distributionMapSize;
				}
				//--------------------------------------------
				//boxの損失を計算する
				//--------------------------------------------
				//----------------------------------
				//DFLを計算/DFLに基づく損失を計算
				//損失は、クロスエントロピー＋softmaxこみの損失が計算されている
				//----------------------------------
				pBoxProbabilities = boxProbabilityArray;
				flt32_t* pDLoss	= pGridDLoss;
				lossGridDF = 0.0f;
				for (int32_t j = 0; j < 4; j++) {
					flt32_t		lossBoxDF = 0.0f;
					compute_dfl_loss(pBoxProbabilities, pGridTeacher[j],  distributionMapSize, lamda_dfl, &lossBoxDF,pDLoss);
					pBoxProbabilities	+= distributionMapSize;
					pDLoss				+= distributionMapSize;
					//累積損失
					lossGridDF += lossBoxDF;
				}
				//----------------------------------
				//CIoUに基づく損失を計算
				//----------------------------------
				flt32_t		IoU = 0.0f;
				flt32_t		lossBoxCIoU = 0.0f;
				compute_ciou_loss(pGridTeacher,boxProbabilityArray, gridSize, distributionMapSize, lamda_ciou , &IoU, &lossBoxCIoU, boxProbabilityDlossArray);
				//----------------------------------
				// CIoU loss
				//----------------------------------
				lossGridCIoU = lossBoxCIoU;
				//----------------------------------
				//DIoUDFLとDFLの損失の微分を加算する
				//----------------------------------
				for (int32_t j = 0; j < distributionMapSize * 4; j++) {
					pGridDLoss[j] += boxProbabilityDlossArray[j];
				}
				//----------------------------------
				//クラス分類の損失および勾配計算
				//----------------------------------
				flt32_t* pTeacherProbability	= pGridTeacher + 4;
				flt32_t* pClassProbability		= pGridPrediction + distributionMapSize * 4;
				pDLoss							= pGridDLoss + distributionMapSize * 4;
				//分類確率計算にfocal lossを利用
				flt32_t lossProbability = 0.0f;
				compute_focal_loss(nClasses, pTeacherProbability, pClassProbability, 0.25f, lamda_class, &lossProbability, pDLoss);
				lossGridProb = lossProbability;
				//----------------------------------
				// 認識率のモニター
				//----------------------------------
				uint32_t	p_best_class = 0;
				uint32_t	t_best_class = 0;
				flt32_t		p_max_prob = -1.e10f;
				flt32_t		t_max_prob = -1.e10f;
				for (int32_t iClass = 0; iClass < nClasses; iClass++) {
					if (p_max_prob < pClassProbability[iClass]) {
						p_max_prob = pClassProbability[iClass];
						p_best_class = iClass;
					}
					if (t_max_prob < pTeacherProbability[iClass]) {
						t_max_prob = pTeacherProbability[iClass];
						t_best_class = iClass;
					}
				}
				if (p_best_class == t_best_class) {
					box_class_corrrect_count++;
				}
				//----------------------------------
				//IoU値モニター
				//----------------------------------
				box_accumulated_IoU += IoU;
				//----------------------------------
				// 測定数をカウントする
				//----------------------------------
				box_count++;
			}
			else {
				//------------------------------------------
				//このグリッドにはオブジェクトのセンターが存在しない
				//------------------------------------------
				//--------------------------------------------
				//クラス分類の損失および勾配計算で背景を学習：背景の学習
				//--------------------------------------------
				flt32_t* pTeacherProbability	= pGridTeacher + 4;
				flt32_t* pClassProbability		= pGridPrediction + distributionMapSize * 4;
				flt32_t* pDLoss					= pGridDLoss + distributionMapSize * 4;
				//確率ロス計算タイプで異なる
				flt32_t lossProbability = 0.0f;
				compute_focal_loss(nClasses, pTeacherProbability, pClassProbability, 0.25f, lamda_class, &lossProbability, pDLoss);
				lossGridNoObj = lossProbability;
			}
			//------------------------------------------
			//次のグリッドに移動
			//------------------------------------------
			pGridTeacher	+= (4 + nClasses);
			pGridPrediction += (distributionMapSize * 4 + nClasses);
			if (pGridDLoss != NULL) {
				pGridDLoss += (distributionMapSize * 4 + nClasses);
			}
			//損失を加算
			lossDF			+= lossGridDF;
			lossCIoU		+= lossGridCIoU;
			lossClassProb	+= lossGridProb;
			lossNoObj		+= lossGridNoObj;
		}
	}
	//-------------------------------------------------------
	// 学習測定値
	//-------------------------------------------------------
	if (pMeasuredValues != NULL) {
		//----------------------------------
		// 損失の測定値
		//----------------------------------
		pMeasuredValues->lossTotal		= lossCIoU + lossDF + lossClassProb + lossNoObj;
		pMeasuredValues->lossCIoU		= lossCIoU;
		pMeasuredValues->lossDF			= lossDF;
		pMeasuredValues->lossClassProb	= lossClassProb;
		pMeasuredValues->lossNoObj		= lossNoObj;
		//----------------------------------
		// 認識率と平均IoUの測定値
		//----------------------------------
		if (box_count > 0) {
			pMeasuredValues->classAccuracy	= 100.0f * (flt32_t)box_class_corrrect_count / (flt32_t)box_count;
			pMeasuredValues->averageIoU		= 100.0f * box_accumulated_IoU / (flt32_t)box_count;
		}
		else {
			pMeasuredValues->classAccuracy	= 0.0f;
			pMeasuredValues->averageIoU		= 0.0f;
		}
	}
	return TRUE;
}

