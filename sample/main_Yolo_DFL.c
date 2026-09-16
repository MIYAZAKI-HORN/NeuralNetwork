//**********************************************************************************
//
//	自作ニューラルネットワークプログラム利用サンプルプログラム（YOLO）
//　**** YOLO : Distribution Focal Loss (DFL) type ***
// 
//	【内容】
//	C言語記述の自作ニューラルネットワークと同じくC言語で記述した、
//  YOLO（You Only Look Once）アルゴリズムにより物体検出の学習機能を検証するためのサンプルプログラムです。
// 
//　【利用しているYOLOの損失計算アルゴリズム】
//　このサンプルプログラムでは、これまでYOLO技術の改良のなかで種々提案されてきた技術のうち、
//　主に以下の技術を利用しています。
//　１．DFL (Distribution Focal Loss)
//　　　バウンディングボックスの各境界位置を連続値ではなく確率分布として扱い、
//　　　正解位置を確率的に予測します。
//　２．CIoU（Complete IoU）Loss
//　　　中⼼距離やアスペクト⽐の⼀致度を考慮することで、ボックスの収束速度と精度が向上しています。
//　３．クラス分類の確率計算にFocal Loss
//　　　圧倒的に多い、背景グリッドセルの学習安定性に定評があります。
//　４．背景学習に分類確率を利用
//　　　YOLOの初期バージョンで利用されていた、信頼度は利用せず、バウンディングボックスの存在しないグリッドは、
//　　　クラス分類の確率を利用して、閾値で背景と判断します。
//
//　上記実装は、YORO_DFL_Training関数内に実装されています。
// 
//　【学習データ】
//　１．https://universe.roboflow.com/browse　からダウンロードした、二種類の学習データを利用しています。
//　　1.1 Oxford-IIIT-Pet.v1-1
//　　　犬と猫のイメージで、一枚のイメージに、一つの種類（犬か猫）だけが存在し、その顔の部分を検出し、分類するためのデータです。
//　　　比較的検出しやすいのではと思い採用しました。
//　　1.2 Thermal Dogs and People.v6-raw-images_autoorient
//　　　赤外線イメージで、一枚の画像に一人以上の人間か（と）犬があり、それぞれのボディーとその種類を検出します。
//　　　赤外線イメージなので輪郭がぼやっとしており、DFLのアルゴリズムを検証するには、最適と思いました。
//　２．学習データ形状と総数は、本サンプルプログラムがアルゴリズムの検証用途でもあり、また、GitHubにコピーするデータのサイズも考慮して、
//　　　通常の学習データとしては小さくなっています。
//　　　入力画像：128x12x3
//　　　グリッドサイズ：7x7
//　３．学習データ構造
//　　　このサンプルでは、アップロードするサイズを小さくするためもあり、以下の仕様となっています。
//　　　入力マトリクス：128x128x3 チャネルラスト
//　　　グリッド：7x7
//　　　教師データ：6 = 4（bounding box: left top corner(x,y):2 , right bottom corner(x,y):2）+ 2(class: one hot)　
// 　　　　「犬と猫イメージ」「赤外線イメージ」も分類数は2です。サイズが2の配列のどちらかに1.0、もう一方に0.0が入っています。
//　　　データはすべてfloatデータです。
//      学習データファイルには、[入力データ（128x128x3=49152）+ 教師データ（7x7x6=294）] x 50イメージが格納されています。
//　　　　サイズ：2,472,300 x 4バイト
// 　
//　【ニューラルネットワーク構造】
//　　　簡単で、比較的小さな畳み込みニューラルネットワークが設定されています。最終層を含めて、conv2dが5段。
//　　　グリッサイズドを7x7しているため、バウンディングボックス形状推定用の確率マップサイズは4としています。（4グリッドセル）
//　　　このソースコードの中で定義されています。
// 
//　【付加機能】
//　　学習中の、損失関数の各損失項の値を取得することができます。画面とlogファイルに出力しています。
//　　グラフ化すると、どのように収束するかを視覚的に見ることができます。
//..................................................................................
//
//	学習データ前処理：
//	オリジナルの画像データの値は0～255までの数値となっていますので。255で除算して0.0～1.0の浮動小数点データにしています。
// 
//**********************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include "SequentialNet.h"
#include "NeuralNetModelConstructor.h"
#include "LOG_Function.h"
#include "YOLO_DFL_Computation.h"

//-----------------------------------------------------------------------------
//テスト対象定義
//-----------------------------------------------------------------------------
typedef enum tagEvaluationModelType {
	YOLO_DOG_AND_CAT_DFL			= 1,	//「犬と猫の画像からそれぞれの顔を検出」
	YOLO_THERMO_DOG_AND_PERSON_DFL	= 2,	//「赤外線画像の犬と人間を検出」
} EvaluationModelType;

//-----------------------------------------------------------------------------
//データファイル定義
//-----------------------------------------------------------------------------
#define DATA_FOLDER ("..\\Data\\YOLO\\")

#define PET_TRAIN_IMAGE_DATA_FILE_NAME		("PET_YOLO_AF_DFL_I128_G7_C2_sample50.flt")
#define PET_MODEL_FILE_NAME					("PET_YOLO_AF_DFL_I128_G7_M4_C2.nnm")
#define PET_LOG_FILE_NAME					("PET_YOLO_AF_DFL_I128_G7_M4_C2.txt")

#define TDP_TRAIN_IMAGE_DATA_FILE_NAME		("TDP_YOLO_AF_DFL_I128_G7_C2_sample50.flt")
#define TDP_MODEL_FILE_NAME					("TDP_YOLO_AF_DFL_I128_G7_M4_C2.nnm")
#define TDP_LOG_FILE_NAME					("TDP_YOLO_AF_DFL_I128_G7_M4_C2.txt")

//-----------------------------------------------------------------------------
//データ形状 YOLO DFL
//-----------------------------------------------------------------------------
#define INPUT_IMAGE_SIZE				(128)
#define GRID_SIZE						(7)
#define DISTRIBUTIONAL_MAP_SIZE			(4)
#define CLASS_SIZE						(2)
#define PREDICTED_GRID_DATA_SIZE		(DISTRIBUTIONAL_MAP_SIZE * 4 + CLASS_SIZE)
#define OUTPUT_SIZE						(GRID_SIZE * GRID_SIZE * PREDICTED_GRID_DATA_SIZE)
#define SINGLE_IMAGE_DATA_SIZE			(INPUT_IMAGE_SIZE * INPUT_IMAGE_SIZE * 3)
#define SINGLE_TEACHER_DATA_SIZE		(GRID_SIZE * GRID_SIZE * (4 + CLASS_SIZE))	// ltc(x,y) rbc(x,y) + class size
#define SINGLE_TRAIN_DATA_SIZE			(SINGLE_IMAGE_DATA_SIZE + SINGLE_TEACHER_DATA_SIZE)

//-----------------------------------------------------------------------------
//学習
//-----------------------------------------------------------------------------
#define EPOCHS							(100)
#define BATCH_SIZE						(50)

//-----------------------------------------------------------------------------
//その他定義
//-----------------------------------------------------------------------------
#define MAX_FILE_NAME_LENGTH			(300)

//-----------------------------------------------------------------------------
//ニューラルネットワークモデルを定義するための構造体
//-----------------------------------------------------------------------------
typedef struct tagModelLearning {
	ModelInformation		modelInformation;	//ニューラルネットワークモデル定義
	uint32_t				batchSize;			//
	NeuralNetOptimizerType	optimizer;
	char					trainFileName[200];
	char					modelFileName[200];
	char					logFileName[200];
} ModelLearning;

//-----------------------------------------------------------------------------
//シーケンシャルモデル情報設定関数
//-----------------------------------------------------------------------------
bool_t
setModelInformation(EvaluationModelType modelType, ModelLearning* pModelInfo)
{
	ModelInformation* pModel = &pModelInfo->modelInformation;
	//---------------------------------------------
	//バッチサイズ
	//---------------------------------------------
	pModelInfo->batchSize	= BATCH_SIZE;
	//---------------------------------------------
	//optimizer
	//---------------------------------------------
	pModelInfo->optimizer = NEURAL_NET_OPTIMIZER_ADAM;
	//---------------------------------------------
	//学習ファイル、モデル保存ファイル
	//---------------------------------------------
	switch (modelType) {
	case YOLO_DOG_AND_CAT_DFL:
		//データファイル
		strcpy(pModelInfo->trainFileName, PET_TRAIN_IMAGE_DATA_FILE_NAME);
		//モデルファイル
		strcpy(pModelInfo->modelFileName, PET_MODEL_FILE_NAME);
		//ログファイル
		strcpy(pModelInfo->logFileName, PET_LOG_FILE_NAME);
		break;
	case YOLO_THERMO_DOG_AND_PERSON_DFL:
		//データファイル
		strcpy(pModelInfo->trainFileName, TDP_TRAIN_IMAGE_DATA_FILE_NAME);
		//モデルファイル
		strcpy(pModelInfo->modelFileName, TDP_MODEL_FILE_NAME);
		//ログファイル
		strcpy(pModelInfo->logFileName, TDP_LOG_FILE_NAME);
		break;
	}
	//---------------------------------------------
	//ニューラルネットワークモデル定義
	//100層まで定義可能
	//「NeuralNetModelConstructor.h」のMAX_LAYERSで定義されています
	//---------------------------------------------
	switch (modelType) {
	case YOLO_DOG_AND_CAT_DFL:
	case YOLO_THERMO_DOG_AND_PERSON_DFL:
		//入力データ次元
		sequential_model_header(pModel, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3);
		//1
		conv2d(pModel, 64, 8, 8, 2, 2, FALSE);
		batch_normalization(pModel);
		activationReLU(pModel, 0.01f);
		//2
		conv2d(pModel, 64, 6, 6, 2, 2, FALSE);
		batch_normalization(pModel);
		activationReLU(pModel, 0.01f);
		//3
		conv2d(pModel, 128, 5, 5, 2, 2, FALSE);
		batch_normalization(pModel);
		activationReLU(pModel, 0.01f);
		//4
		conv2d(pModel, 256, 4, 4, 1, 1, FALSE);
		batch_normalization(pModel);
		activationReLU(pModel, 0.01f);
		//last 7x7x1x(4x4+2)
		conv2d(pModel, 18, 3, 3, 1, 1, FALSE);
		break;	
	}
	return TRUE;
}

//-----------------------------------------------------------------------------
//メイン学習関数
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	uint32_t				i,j,iData;
	bool_t					fStatus;
	uint32_t*				pModelImage = NULL;
	// best model保持
	uint32_t*				pBestModelImage = NULL;
	uint32_t				epochBest;
	uint32_t				epochBestSaved;
	flt32_t					lossBest;
	// ニューラルネットワークモデル
	uint32_t				sizeOfModelImageIn32BitWord;
	static ModelLearning	modelLearning;
	EvaluationModelType		evaluationModelType;
	uint32_t*				pSequentialModelWorkArea = NULL;
	uint32_t				sequentialModelWorkAreaSizeIn32BitWord = 0;
	handle_t				hModel;
	// 学習関連
	bool_t					fEnableLearning;
	uint32_t				numberOfBackPropagationLayers;
	uint32_t				dataSizeIn32BitWord;
	flt32_t*				pTrainingData = NULL;
	flt32_t*				pTrainDataHead;
	flt32_t*				pTrainImageHead;
	flt32_t*				pTrainTeacherHead;
	flt32_t*				pTestData = NULL;
	uint32_t				inputHeight;
	uint32_t				inputWidth;
	uint32_t				inputChannel;
	uint32_t				inputDim;
	uint32_t				outputHeight;
	uint32_t				outputWidth;
	uint32_t				outputChannel;
	uint32_t				outputDim;
	uint32_t				nTrainData;
	uint32_t				epoch;
	uint32_t				numberOfEpochs;
	uint32_t				trainCounter;
	flt32_t					predectedValue;
	flt32_t*				pPredictedValueArray = NULL;
	flt32_t*				pDLossArray = NULL;
	// YOLO学習ハイパーパラメタ
	flt32_t					lamda_ciou			= 1.0f;
	flt32_t					lamda_dfl			= 1.0f;
	flt32_t					lamda_class			= 1.0f;
	// YOLO損失関数モニター
	DFL_Monitor				averages;
	DFL_Monitor				trainingMonitor;
	//情報表示
	char					strInformation[200];
	//ファイル
	FILE*					pf_trainData;
	FILE*					pf_model;
	char					trainDataFileName[MAX_FILE_NAME_LENGTH];
	//=============================================================================
	// 
	// YOLOモデル
	// 
	//=============================================================================
	//-----------------------------------------------------------------------------
	//モデル情報のセット
	//-----------------------------------------------------------------------------
	evaluationModelType = YOLO_THERMO_DOG_AND_PERSON_DFL;	//「犬と猫の画像からそれぞれの顔を検出」を選択　犬と猫の物体検出には、YOLO_DOG_AND_CAT_DFLをセット
	setModelInformation(evaluationModelType,&modelLearning);
	//-----------------------------------------------------------------------------
	//ログファイル
	//-----------------------------------------------------------------------------
	fStatus = OPEN_LOG_FILE(modelLearning.logFileName);
	//-----------------------------------------------------------------------------
	//モデル情報セット
	//-----------------------------------------------------------------------------
	pModelImage = NULL;
	sizeOfModelImageIn32BitWord = 0;
	fStatus = constructNeuralNetModel(&modelLearning.modelInformation,&pModelImage,&sizeOfModelImageIn32BitWord);
	if (fStatus == FALSE) {
		return 1;
	}
	sprintf(strInformation,"network image size = %d (byte)\n", sizeOfModelImageIn32BitWord*sizeof(uint32_t));
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//-----------------------------------------------------------------------------
	//オプティマイザーなど
	//-----------------------------------------------------------------------------
	fEnableLearning					= TRUE;	//逆伝播可能条件でモデルを構築する
	numberOfBackPropagationLayers	= 0;	//全てのレイヤーを対象とする
	switch (modelLearning.optimizer) {
	case NEURAL_NET_OPTIMIZER_SGD:
		SAVE_LOG("optimizer : SGD");
		break;
	case NEURAL_NET_OPTIMIZER_RMSPROP:
		SAVE_LOG("optimizer : RMSPROP");
		break;
	case NEURAL_NET_OPTIMIZER_ADAM:
		SAVE_LOG("optimizer : ADAM");
		break;
	}
	SAVE_LOG_WITH_INT("batch size : ",modelLearning.batchSize);
	//-----------------------------------------------------------------------------
	//損失関数係数
	//-----------------------------------------------------------------------------
	SAVE_LOG_WITH_FLT("lamda  CIoU       : ", lamda_ciou);
	SAVE_LOG_WITH_FLT("lamda　DFL        : ", lamda_dfl);
	SAVE_LOG_WITH_FLT("lamda　class      : ", lamda_class);
	//-----------------------------------------------------------------------------
	//モデル構築
	//-----------------------------------------------------------------------------
	sequentialModelWorkAreaSizeIn32BitWord = SequentialNet_getSizeIn32BitWord(pModelImage, fEnableLearning, modelLearning.optimizer, numberOfBackPropagationLayers);
	if (sequentialModelWorkAreaSizeIn32BitWord == 0) {
		printf("error, obtaining work area size\n");
		return 1;
	}
	pSequentialModelWorkArea = (uint32_t*)malloc(sizeof(uint32_t) * sequentialModelWorkAreaSizeIn32BitWord);
	hModel = SequentialNet_construct(pModelImage, fEnableLearning, modelLearning.batchSize, modelLearning.optimizer, numberOfBackPropagationLayers, pSequentialModelWorkArea, sequentialModelWorkAreaSizeIn32BitWord);
	sprintf(strInformation,"work area (byte) : %d", sequentialModelWorkAreaSizeIn32BitWord * sizeof(uint32_t));
	SAVE_LOG(strInformation);
	if (hModel == NULL) {
		printf("error, constructing sequential model\n");
		return 1;
	}
	//-----------------------------------------------------------------------------
	//パラメタ初期化
	//-----------------------------------------------------------------------------
	fStatus = SequentialNet_initializeParameter(hModel);
	if (fStatus == FALSE) {
		printf("error, initializing parameters\n");
		return 1;
	}
	//-----------------------------------------------------------------------------
	//モデルデータの入力次元を取得
	//-----------------------------------------------------------------------------
	fStatus = SequentialNet_getInputShape(hModel, &inputHeight, &inputWidth, &inputChannel);
	if (fStatus == FALSE) {
		printf("error, obtaining input shape\n");
		return 1;
	}
	inputDim = inputHeight * inputWidth * inputChannel;
	//-----------------------------------------------------------------------------
	//モデルデータの入出力次元を取得
	//-----------------------------------------------------------------------------
	fStatus = SequentialNet_getOutputShape(hModel, &outputHeight, &outputWidth, &outputChannel);
	if (fStatus == FALSE) {
		printf("error, obtaining output shape\n");
		return 1;
	}
	outputDim = outputHeight * outputWidth * outputChannel;
	//=============================================================================
	// 
	//学習
	// 
	//=============================================================================
	//-----------------------------------------------------------------------------
	//誤差逆伝播計算バッファ確保
	//-----------------------------------------------------------------------------
	pPredictedValueArray = (flt32_t*)malloc(sizeof(flt32_t) * outputDim);
	pDLossArray = (flt32_t*)malloc(sizeof(flt32_t) * outputDim);
	numberOfEpochs = EPOCHS;
	//-----------------------------------------------------------------------------
	// best model
	//-----------------------------------------------------------------------------
	pBestModelImage = (uint32_t*)malloc(sizeof(uint32_t)* sizeOfModelImageIn32BitWord);
	epochBestSaved = -1;
	epochBest	= 1;
	lossBest = 1.0e10;
	//-----------------------------------------------------------------------------
	//教師データファイルオープン
	//-----------------------------------------------------------------------------
	sprintf(trainDataFileName, "%s", DATA_FOLDER);
	sprintf(trainDataFileName + strlen(DATA_FOLDER), "%s", modelLearning.trainFileName);
	pf_trainData = fopen(trainDataFileName, "rb");
	if (pf_trainData == NULL) {
		printf("file open error : %s\n", trainDataFileName);
		return FALSE;
	}
	sprintf(strInformation, "train data file : %s\n", trainDataFileName);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//-----------------------------------------------------------------------------
	//教師データバッファ確保
	//-----------------------------------------------------------------------------
	dataSizeIn32BitWord = SINGLE_TRAIN_DATA_SIZE;
	pTrainingData = (flt32_t*)malloc(sizeof(uint32_t) * dataSizeIn32BitWord);
	if (pTrainingData == NULL) {
		return 1;
	}
	//-----------------------------------------------------------------------------
	//教師データ個数カウント
	//-----------------------------------------------------------------------------
	nTrainData = 0;
	while(1) {
		size_t sizeRead = fread(pTrainingData, sizeof(uint32_t), SINGLE_TRAIN_DATA_SIZE, pf_trainData);
		if (sizeRead != SINGLE_TRAIN_DATA_SIZE) {
			//ファイル終端
			break;
		}
		nTrainData++;
	}
	//ファイルポインタを先頭に戻す
	fseek(pf_trainData, 0, SEEK_SET);
	//学習データ数
	sprintf(strInformation, "train data : %d\n", nTrainData);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//-----------------------------------------------------------------------------
	//学習と進捗状況表示
	//-----------------------------------------------------------------------------
	sprintf(strInformation, "epoch\tloss<total>\tloss<CIoU>\tloss<DF>\tloss<class>\tloss<no obj>\taccuracy(%%)\tIoU(%%)\n");
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	for (epoch = 1; epoch <= numberOfEpochs; epoch++) {
		//-------------------------------------------------------------------------
		//学習測定値初期化
		//-------------------------------------------------------------------------
		trainCounter = 0;
		averages.lossTotal		= 0.0f;
		averages.lossCIoU		= 0.0f;
		averages.lossDF			= 0.0f;
		averages.lossClassProb	= 0.0f;
		averages.lossNoObj		= 0.0f;
		averages.classAccuracy	= 0.0f;
		averages.averageIoU		= 0.0f;
		//-------------------------------------------------------------------------
		//学習データ全てを利用して学習
		//-------------------------------------------------------------------------
		fseek(pf_trainData, 0, SEEK_SET);
		for(iData=0;iData<nTrainData;iData++) {
			size_t sizeRead = fread(pTrainingData, sizeof(uint32_t), SINGLE_TRAIN_DATA_SIZE, pf_trainData);
			if (sizeRead != SINGLE_TRAIN_DATA_SIZE) {
				//ファイル終端で終了　次のepochに移動
				sprintf(strInformation, "epoch %u終了\t%u\t", epoch, trainCounter);
				SAVE_LOG(strInformation);
				break;
			}
			//-------------------------------------------------------------------------
			//多次元出力
			//-------------------------------------------------------------------------
			pTrainDataHead = pTrainingData;
			pTrainImageHead = pTrainDataHead;
			pTrainTeacherHead = pTrainDataHead + SINGLE_IMAGE_DATA_SIZE;
			//-------------------------------------------------------------------------
			//学習数カウント
			//-------------------------------------------------------------------------
			trainCounter++;
			//-------------------------------------------------------------------------
			// 
			//本体モデル順伝搬
			// 
			//-------------------------------------------------------------------------
			fStatus = SequentialNet_predict(hModel, pTrainImageHead, inputDim);
			if (fStatus == FALSE) {
				printf("error, performing prediction\n");
				return 1;
			}
			//-------------------------------------------------------------------------
			//本体モデルから予測値を得る
			//-------------------------------------------------------------------------
			for (i = 0; i < outputDim; i++) {
				fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
				if (fStatus == FALSE) {
					printf("error, getting prediction value\n");
					return 1;
				}
				pPredictedValueArray[i] = predectedValue;
			}
			//-------------------------------------------------------------------------
			// 
			//YOLO損失値と逆伝搬値を得る
			// 
			//-------------------------------------------------------------------------
			YORO_DFL_Training(
				GRID_SIZE,					//グリッドサイズ
				DISTRIBUTIONAL_MAP_SIZE,	//DFL確率マップサイズ
				CLASS_SIZE,					//分類数
				pTrainTeacherHead,			//学習イメージ先頭（RGB 3チャンネル）
				pPredictedValueArray,		//ネットワーク出力値
				pDLossArray,				//損失微分値配列
				lamda_ciou,					//CIoU損失値計算に乗算する係数
				lamda_dfl,					//DFL損失値計算に乗算する係数
				lamda_class,				//分類確率損失計算に乗算する係数
				&trainingMonitor			//学習測定値構造体
			);
			//---------------------------------------------
			//学習測定値の累積
			//---------------------------------------------
			averages.lossTotal		+= trainingMonitor.lossTotal;
			averages.lossCIoU		+= trainingMonitor.lossCIoU;
			averages.lossDF			+= trainingMonitor.lossDF;
			averages.lossClassProb	+= trainingMonitor.lossClassProb;
			averages.lossNoObj		+= trainingMonitor.lossNoObj;
			averages.classAccuracy	+= trainingMonitor.classAccuracy;
			averages.averageIoU		+= trainingMonitor.averageIoU;
			//-------------------------------------------------------------------------
			// 
			//誤差逆伝搬
			// 
			//-------------------------------------------------------------------------
			fStatus = SequentialNet_fit(hModel, pDLossArray, outputDim);
			if (fStatus == FALSE) {
				printf("error, fitting\n");
				return 1;
			}
		}
		//-----------------------------------------------------------------------------
		//学習進行状況表示
		//-----------------------------------------------------------------------------
		if (trainCounter > 0) {
			averages.lossTotal		/= (flt32_t)trainCounter;
			averages.lossCIoU		/= (flt32_t)trainCounter;
			averages.lossDF			/= (flt32_t)trainCounter;
			averages.lossClassProb	/= (flt32_t)trainCounter;
			averages.lossNoObj		/= (flt32_t)trainCounter;
			averages.classAccuracy	/= (flt32_t)trainCounter;
			averages.averageIoU		/= (flt32_t)trainCounter;
			sprintf(strInformation, "%u\t%6.3e\t%6.3e\t%6.3e\t%6.3e\t%6.3e\t%6.2f\t%6.2f\n", epoch, averages.lossTotal, averages.lossCIoU, averages.lossDF, averages.lossClassProb, averages.lossNoObj, averages.classAccuracy, averages.averageIoU);
			SAVE_LOG_WITHOUT_RETURN(strInformation);
			//---------------------------------------------
			// best model
			//---------------------------------------------
			if (lossBest > averages.lossTotal) {
				lossBest = averages.lossTotal;
				epochBest = epoch;
				memcpy(pBestModelImage, pModelImage, sizeof(uint32_t) * sizeOfModelImageIn32BitWord);
			}
		}
	}
	fclose(pf_trainData);
	//=============================================================================
	//モデルの保存とログファイルクローズ
	//=============================================================================
	pf_model = fopen(modelLearning.modelFileName, "wb");
	if (pf_model != NULL) {
		fwrite(pBestModelImage, sizeof(uint32_t), sizeOfModelImageIn32BitWord, pf_model);
		fclose(pf_model);
	}
	CLOSE_LOG_FILE();
	//=============================================================================
	//メモリ解放
	//=============================================================================
	free(pModelImage);
	free(pBestModelImage);
	free(pSequentialModelWorkArea);
	free(pTrainingData);
	free(pTestData);
	free(pPredictedValueArray);
	free(pDLossArray);
	return 0;
}
