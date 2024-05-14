//**********************************************************************************
//
//	自作ニューラルネットワークプログラム利用サンプルプログラム（回帰問題）
// 
//	内容：
//	C言語記述の自作ニューラルネットワーク（シーケンシャル）を利用して回帰問題に対応します。
//	全結合型ニューラルネットワークを構築し、
//	ニューラルネットワークのパラメタに初期値を与え、指定したepoch数学習を行い、
//	学習データと検証データでの、平均二乗誤差を逐次表示します。
// 
//	検証データ：
//	Boston House Prices　CSV形式　データ系列は以下の様な並びを想定しています 
//CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
//6.71772, 0, 18.1, 0, 0.713, 6.749, 92.6, 2.3236, 24, 666, 20.2, 0.32, 17.44, 13.4
//5.44114, 0, 18.1, 0, 0.713, 6.655, 98.2, 2.3552, 24, 666, 20.2, 355.29, 17.73, 15.2
//..................................................................................
//
//	前処理：
//	特になし。
//
//**********************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "SequentialNet.h"
#include "RandomValueGenerator.h"
#include "NeuralNetModelConstructor.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "LOG_Function.h"

//-----------------------------------------------------------------------------------------------------
//シーケンシャルニューラルネットワークモデル情報拡張
//-----------------------------------------------------------------------------------------------------
typedef struct tagModelInformationEx {
	ModelInformation		modelInformation;				//シーケンシャルニューラルネットワークモデル情報
	uint32_t				batchSize;						//学習バッチサイズ
	NeuralNetOptimizerType	optimizer;						//学習オプティマイザー種類
	flt32_t					batchNormalizationMomentum;		//バッチ正規化モメンタム
	flt32_t					reluActivationNegativeSlope;	//Leaky ReLU活性化関数の負値勾配
	char					trainFileName[200];				//学習教師データファイル名
	char					testFileName[200];				//学習テストデータファイル名
	char					modelFileName[200];				//生成されたシーケンシャルニューラルネットワークモデルの保存ファイル名
} ModelInformationEx;

//-----------------------------------------------------------------------------------------------------
//学習パラメタ
//-----------------------------------------------------------------------------------------------------
#define EPOCHS							(500)
#define BATCH_SIZE						(20)
#define BATCH_NORMALIZATION_MOMENTUM	(0.99f)

//-----------------------------------------------------------------------------------------------------
//Leaky ReLU活性化関数の負値勾配
//-----------------------------------------------------------------------------------------------------
#define RELU_ACTIVATION_NEGATIVE_SLOOP	(0.0f)

//-----------------------------------------------------------------------------------------------------
//データファイル定義
//-----------------------------------------------------------------------------------------------------
#define DATA_FOLDER (".\\Data\\")

#define TRAIN_IMAGE_DATA_FILE_NAME		("boston\\boston_train.csv")
#define TEST_IMAGE_DATA_FILE_NAME		("boston\\boston_test.csv")
#define MODEL_FILE_NAME					("boston.nnm")

#define SINGLE_DATA_SIZE	(14)

#define MAX_FILE_NAME_LENGTH	(500)

//-----------------------------------------------------------------------------------------------------
//データ読み込み関数
//-----------------------------------------------------------------------------------------------------
bool_t
readSingleLine(FILE* fp, flt32_t* pData,uint32_t* pDataCount) {
	static char line[100000];	//一行100000まで
	char* pValue;
	char* pHead;
	int dataCounter = 0;
	if (fgets(line, sizeof(line), fp) == NULL) {
		return FALSE;
	}
	if (pData == NULL) {
		return TRUE;
	}
	pValue = line;	//値データ先頭
	pHead = line;	//データ先頭
	while (*pHead != '\0' ) {
		if (*pHead == ',' || *pHead == '\n') {
			//カンマ検出
			*pHead = '\0';
			*pData = atof(pValue);
			pData++;
			pHead++;
			pValue = pHead;
			dataCounter++;
		}
		else {
			pHead++;
		}
	}
	if (pDataCount != NULL) {
		*pDataCount = dataCounter;
	}
	return TRUE;
}

//-----------------------------------------------------------------------------------------------------
//シーケンシャルモデル情報設定関数
//-----------------------------------------------------------------------------------------------------
bool_t
setModelInformation(ModelInformationEx* pNetworkInfo)
{
	ModelInformation* pModel = &pNetworkInfo->modelInformation;
	//バッチサイズ
	pNetworkInfo->batchSize = BATCH_SIZE;
	//batch normalization momenttum
	pNetworkInfo->batchNormalizationMomentum = BATCH_NORMALIZATION_MOMENTUM;
	//relu activation negative sloop
	pNetworkInfo->reluActivationNegativeSlope = RELU_ACTIVATION_NEGATIVE_SLOOP;
	//default optimizer
	pNetworkInfo->optimizer = NEURAL_NET_OPTIMIZER_ADAM;
	//
	//---------------------------------------
	//データファイル
	//---------------------------------------
	strcpy(pNetworkInfo->trainFileName, TRAIN_IMAGE_DATA_FILE_NAME);
	strcpy(pNetworkInfo->testFileName, TEST_IMAGE_DATA_FILE_NAME);
	//---------------------------------------
	//保存モデルファイル名
	//---------------------------------------
	strcpy(pNetworkInfo->modelFileName, MODEL_FILE_NAME);
	//---------------------------------------
	//relu activation negative sloop
	//---------------------------------------
	pNetworkInfo->reluActivationNegativeSlope = RELU_ACTIVATION_NEGATIVE_SLOOP;
	//---------------------------------------
	//モデル定義
	//---------------------------------------
	sequential_model_header(pModel, 1, 13, 1);	//入力　1x13x1　Channel last
	batch_normalization(pModel);
	dense(pModel, 80);
	activation(pModel, NEURAL_NET_ACTIVATION_RELU);
	dense(pModel, 50);
	activation(pModel, NEURAL_NET_ACTIVATION_RELU);
	dense(pModel, 20);
	batch_normalization(pModel);
	activation(pModel, NEURAL_NET_ACTIVATION_RELU);
	dense(pModel, 1);
	return TRUE;
}

int main(int argc, char* argv[])
{
	uint32_t				i,j;
	bool_t					fStatus;
	uint32_t*				pModelImage = NULL;
	uint32_t				sizeOfModelImageIn32BitWord;
	ModelInformationEx		extModelInfo;
	uint32_t*				pSequentialModelWorkArea = NULL;
	uint32_t				sequentialModelWorkAreaSizeIn32BitWord = 0;
	handle_t				hModel;
	bool_t					fEnableLearning;
	uint32_t				numberOfBackPropagationLayers;
	flt32_t*				pTrainingData = NULL;
	flt32_t*				pTrainDataHead;
	flt32_t*				pTestData = NULL;
	flt32_t*				pTestDataHead;
	uint32_t				inputDim;
	uint32_t				outputHeight;
	uint32_t				outputWidth;
	uint32_t				outputChannel;
	uint32_t				outputDim;
	uint32_t				nTrainData;
	uint32_t				nTestData;
	uint32_t				randomValueGegratorWorkAreaIn32BitWord;
	uint32_t*				pRundumValueGenerator;
	handle_t				hRandomValueGenerator;
	uint32_t				epoch;
	uint32_t				numberOfEpochs;
	uint32_t				trainCounter;
	uint32_t				testCounter;
	uint32_t				dataCounter;
	uint32_t				random;
	uint32_t				dataIndex;
	flt32_t					predectedValue;
	flt32_t*				pPredictedValueArray = NULL;
	flt32_t*				pTeacherValueArray = NULL;
	flt32_t*				pDLossArray = NULL;
	flt32_t					meanAbsoluteError;
	flt32_t					trainAverageMeanAbsoluteError;
	flt32_t					testAverageMeanAbsoluteError;
	uint32_t				numOfLayers;
	NetLayerType			layerType;
	handle_t				hLayer;
	char					strInformation[200];
	flt32_t					predictedValueRatio;
	flt32_t					totalTrainPredictedValueRatio;
	flt32_t					totalTestPredictedValueRatio;
	FILE*					pf_model;
	FILE*					fp_train;
	FILE*					fp_test;
	NeuralNetActivationType activationType;
	char					dataFileName[MAX_FILE_NAME_LENGTH];
	//================================================================
	//ログファイル
	//================================================================
	fStatus = OPEN_LOG_FILE(".\\log\\log.txt");
	//================================================================
	//モデル情報のセット
	//================================================================
	setModelInformation(&extModelInfo);
	//================================================================
	//入力次元
	//================================================================
	inputDim = extModelInfo.modelInformation.inHeight * extModelInfo.modelInformation.inWidth * extModelInfo.modelInformation.inChannel;
	//================================================================
	//シーケンシャルニューラルネットワークモデルイメージ作成
	//================================================================
	pModelImage = NULL;
	sizeOfModelImageIn32BitWord = 0;
	fStatus = constructNeuralNetModel(&extModelInfo.modelInformation,&pModelImage,&sizeOfModelImageIn32BitWord);
	if (fStatus == FALSE) {
		return 1;
	}
	sprintf(strInformation,"network image size = %d (byte)\n", sizeOfModelImageIn32BitWord*sizeof(uint32_t));
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//================================================================
	//オプティマイザーなど
	//================================================================
	fEnableLearning					= TRUE;	//逆伝播可能条件でモデルを構築する
	numberOfBackPropagationLayers	= 0;	//最終層からの学習対象数　0：すべての層を対象　それ以外；学習層数（追加学習など）
	switch (extModelInfo.optimizer) {
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
	SAVE_LOG_WITH_INT("batch size : ",extModelInfo.batchSize);
	SAVE_LOG_WITH_FLT("batch normalization momentum : ", extModelInfo.batchNormalizationMomentum);
	SAVE_LOG_WITH_FLT("relu activation negative sloop : ", extModelInfo.reluActivationNegativeSlope);
	//================================================================
	//シーケンシャルニューラルネットワーク構築
	//================================================================
	sequentialModelWorkAreaSizeIn32BitWord = SequentialNet_getSizeIn32BitWord(pModelImage, fEnableLearning, extModelInfo.optimizer, numberOfBackPropagationLayers);
	if (sequentialModelWorkAreaSizeIn32BitWord == 0) {
		printf("error, obtaining work area size\n");
		return 1;
	}
	pSequentialModelWorkArea = (uint32_t*)malloc(sizeof(uint32_t) * sequentialModelWorkAreaSizeIn32BitWord);
	hModel = SequentialNet_construct(pModelImage, fEnableLearning, extModelInfo.batchSize, extModelInfo.optimizer, numberOfBackPropagationLayers, pSequentialModelWorkArea, sequentialModelWorkAreaSizeIn32BitWord);
	sprintf(strInformation,"work area (byte) : %d", sequentialModelWorkAreaSizeIn32BitWord * sizeof(uint32_t));
	SAVE_LOG(strInformation);
	if (hModel == NULL) {
		printf("error, constructing sequential model\n");
		return 1;
	}
	//================================================================
	//レイヤーパラメタ設定
	//================================================================
	SequentialNet_getNumberOfLayers(hModel, &numOfLayers);
	for (i = 0; i < numOfLayers; i++) {
		SequentialNet_getLayerType(hModel, i, &layerType);
		SequentialNet_getLayerHandle(hModel, i, &hLayer);
		switch (layerType) {
		case NET_LAYER_BATCH_NORMALIZATION:
			sprintf(strInformation,"set batch normalization momentum : %10.6f", extModelInfo.batchNormalizationMomentum);
			SAVE_LOG(strInformation);
			fStatus = NeuralNetLayerBatchNormalization_setMomentum(hLayer, extModelInfo.batchNormalizationMomentum);
			if (fStatus == FALSE) {
				printf("error, setting batch normalization momentum\n");
				return 1;
			}
			break;
		case NET_LAYER_ACTIVATION:
			fStatus = NeuralNetLayerActivation_getType(hLayer, &activationType);
			if (fStatus == FALSE) {
				printf("error, obtaining activation type\n");
				return 1;
			}
			switch (activationType) {
			case NEURAL_NET_ACTIVATION_RELU:
				sprintf(strInformation, "set relu activation negative sloop %d : %10.6f", i, extModelInfo.reluActivationNegativeSlope);
				SAVE_LOG(strInformation);
				fStatus = NeuralNetLayerReluActivation_setParameter(hLayer, extModelInfo.reluActivationNegativeSlope);
				if (fStatus == FALSE) {
					printf("error, setting relu activation parameter\n");
					return 1;
				}
				break;
			}
			break;
		default:
			break;
		}
	}
	//================================================================
	//パラメタ初期化
	//================================================================
	fStatus = SequentialNet_initializeParameter(hModel);
	if (fStatus == FALSE) {
		printf("error, initializing parameters\n");
		return 1;
	}
	//================================================================
	//optimizerパラメタ設定：以下はデフォルト値なので設定しなくても良いが、設定方法を示しています
	//================================================================
	SequentialNet_getNumberOfLayers(hModel, &numOfLayers);
	for (i = 0; i < numOfLayers; i++) {
		handle_t hOptimizer = SequentialNet_getOptimizer(hModel,i);
		if (hOptimizer != NULL) {
			OptimizerFunctionTable	optimizerInterface;
			NeuralNetOptimizer_getInterface(hOptimizer,&optimizerInterface);
			switch (optimizerInterface.pGetType(hOptimizer)) {
			case NEURAL_NET_OPTIMIZER_UNDEFINED:
				break;
			case NEURAL_NET_OPTIMIZER_SGD:
				NeuralNetworkOptimizerSGD_setParameters(hOptimizer,0.9f, 0.01f);
				break;
			case NEURAL_NET_OPTIMIZER_RMSPROP:
				NeuralNetworkOptimizerRMSprop_setParameters(hOptimizer,0.9f,0.001f);
				break;
			case NEURAL_NET_OPTIMIZER_ADAM:
				NeuralNetworkOptimizerAdam_setParameters(hOptimizer,0.9f,0.999f, 0.001f);
				break;
			}
		}
	}
	//-----------------------------------------------------------------------------------------------------
	//モデルデータの入出力次元を取得
	//-----------------------------------------------------------------------------------------------------
	fStatus = SequentialNet_getOutputShape(hModel, &outputHeight, &outputWidth, &outputChannel);
	if (fStatus == FALSE) {
		printf("error, obtaining output shape\n");
		return 1;
	}
	outputDim = outputHeight * outputWidth * outputChannel;
	//================================================================
	//学習教師データ読み込み
	//================================================================
	sprintf(strInformation,"train data file : %s\n", extModelInfo.trainFileName);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	sprintf(dataFileName, "%s", DATA_FOLDER);
	sprintf(dataFileName + strlen(DATA_FOLDER), "%s", extModelInfo.trainFileName);
	fp_train = fopen(dataFileName, "r");
	if (fp_train == NULL) {
		printf("train file open error\n");
		return 1;
	}
	nTrainData = 500;
	pTrainingData = (float*)malloc(nTrainData * sizeof(float) * SINGLE_DATA_SIZE);
	//1行目は項目名なので読み飛ばす
	readSingleLine(fp_train, NULL,NULL);
	trainCounter = 0;
	pTrainDataHead = pTrainingData;
	while (readSingleLine(fp_train, pTrainDataHead,&dataCounter)) {
		if (++trainCounter == nTrainData) {
			break;
		}
		if (dataCounter != SINGLE_DATA_SIZE) {
			continue;
		}
		pTrainDataHead += SINGLE_DATA_SIZE;
	}
	nTrainData = trainCounter;
	fclose(fp_train);
	sprintf(strInformation,"train data : %d\n", nTrainData);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//================================================================
	//テストデータ読み込み
	//================================================================
	if (strlen(extModelInfo.testFileName) > 0) {
		sprintf(strInformation,"test data file : %s\n", extModelInfo.testFileName);
		SAVE_LOG_WITHOUT_RETURN(strInformation);
		sprintf(dataFileName, "%s", DATA_FOLDER);
		sprintf(dataFileName + strlen(DATA_FOLDER), "%s", extModelInfo.testFileName);
		fp_test = fopen(dataFileName, "r");
		if (fp_test == NULL) {
			printf("test file open error\n");
			return 1;
		}
		nTestData = 200;
		pTestData = (float*)malloc(nTestData * sizeof(float) * SINGLE_DATA_SIZE);
		//1行目は項目名なので読み飛ばす
		readSingleLine(fp_test, NULL,NULL);
		testCounter = 0;
		pTestDataHead = pTestData;
		while (readSingleLine(fp_test, pTestDataHead, &dataCounter)) {
			if (++testCounter == nTestData) {
				break;
			}
			if (dataCounter != SINGLE_DATA_SIZE) {
				continue;
			}
			pTestDataHead += SINGLE_DATA_SIZE;
		}
		nTestData = testCounter;
		fclose(fp_test);
		sprintf(strInformation,"test data : %d\n", nTestData);
		SAVE_LOG_WITHOUT_RETURN(strInformation);
	}
	else {
		nTestData = 0;
		fp_test = NULL;
	}
	//================================================================
	//誤差逆伝播計算バッファ確保
	//================================================================
	pPredictedValueArray = (flt32_t*)malloc(sizeof(flt32_t) * outputDim);
	pTeacherValueArray = (flt32_t*)malloc(sizeof(flt32_t) * outputDim);
	pDLossArray = (flt32_t*)malloc(sizeof(flt32_t) * outputDim);
	//================================================================
	//乱数発生モデル構築
	//================================================================
	randomValueGegratorWorkAreaIn32BitWord = RandomValueGenerator_getSizeIn32BitWord();
	pRundumValueGenerator = (uint32_t*)malloc(sizeof(uint32_t) * randomValueGegratorWorkAreaIn32BitWord);
	hRandomValueGenerator = RandomValueGenerator_construct(0, pRundumValueGenerator, randomValueGegratorWorkAreaIn32BitWord);
	//================================================================
	//学習
	//================================================================
	sprintf(strInformation,"epoch\tAME (train)\tAME (test)\n");
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	numberOfEpochs = EPOCHS;
	for (epoch = 1; epoch <= numberOfEpochs; epoch++) {
		//-------------------------------------------------------------------------
		//学習データ全てを利用して学習
		//-------------------------------------------------------------------------
		trainCounter = 0;
		trainAverageMeanAbsoluteError = 0.0f;
		for (j = 0; j < nTrainData; j++) {
			//-------------------------------------------------------------------------
			//　乱数により学習データを選択
			//-------------------------------------------------------------------------
			random = RandomValueGenerator_getIntegerValue(hRandomValueGenerator);
			dataIndex = random % nTrainData;
			//dataIndex = j;
			//SAVE_LOG_WITH_INT("", dataIndex);
			//-------------------------------------------------------------------------
			//多次元出力
			//-------------------------------------------------------------------------
			pTrainDataHead = pTrainingData + dataIndex * (outputDim + inputDim);
			//-------------------------------------------------------------------------
			//教師データ作成
			//-------------------------------------------------------------------------
			memcpy(pTeacherValueArray, pTrainDataHead+inputDim, sizeof(flt32_t) * outputDim);
			//-------------------------------------------------------------------------
			//学習数カウント
			//-------------------------------------------------------------------------
			trainCounter++;
			//-------------------------------------------------------------------------
			//順伝搬
			//-------------------------------------------------------------------------
			fStatus = SequentialNet_predict(hModel, pTrainDataHead, inputDim);
			if (fStatus == FALSE) {
				printf("error, performing prediction\n");
				return 1;
			}
			for (i = 0; i < outputDim; i++) {
				fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
				if (fStatus == FALSE) {
					printf("error, getting prediction value\n");
					return 1;
				}
				pPredictedValueArray[i] = predectedValue;
			}
			//---------------------------------------------
			//mean absolute error
			//---------------------------------------------
			meanAbsoluteError = 0.0f;
			for (i = 0; i < outputDim; i++) {
				flt32_t mse = (pPredictedValueArray[i] - pTeacherValueArray[i]);
				meanAbsoluteError += mse * mse;
			}
			meanAbsoluteError /= (flt32_t)outputDim;
			trainAverageMeanAbsoluteError += meanAbsoluteError;
			//---------------------------------------------
			// mean absolute error微分値
			//---------------------------------------------
			for (i = 0; i < outputDim; i++) {
				flt32_t diff = pPredictedValueArray[i] - pTeacherValueArray[i];
				pDLossArray[i] = meanAbsoluteError * diff * 2.0f / (flt32_t)outputDim;
			}
			//-------------------------------------------------------------------------
			//逆伝搬
			//-------------------------------------------------------------------------
			fStatus = SequentialNet_fit(hModel, pDLossArray, outputDim);
			if (fStatus == FALSE) {
				printf("error, fitting\n");
				return 1;
			}
		}
		//-------------------------------------------------------------------------
		//テストデータ評価
		//-------------------------------------------------------------------------
		testCounter = 0;
		testAverageMeanAbsoluteError = 0.0f;
		for (j = 0; j < nTestData; j++) {
			//-------------------------------------------------------------------------
			//評価データ
			//-------------------------------------------------------------------------
			dataIndex = j;
			pTestDataHead = pTestData + dataIndex * (outputDim + inputDim);
			//-------------------------------------------------------------------------
			//テスト数カウント
			//-------------------------------------------------------------------------
			testCounter++;
			//-------------------------------------------------------------------------
			//予測
			//-------------------------------------------------------------------------
			fStatus = SequentialNet_predict(hModel, pTestDataHead, inputDim);
			if (fStatus == FALSE) {
				return 1;
			}
			for (i = 0; i < outputDim; i++) {
				fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
				if (fStatus == FALSE) {
					printf("error, getting prediction value\n");
					return 1;
				}
				pPredictedValueArray[i] = predectedValue;
			}
			//---------------------------------------------
			//mean absolute error
			//---------------------------------------------
			meanAbsoluteError = 0.0f;
			for (i = 0; i < outputDim; i++) {
				flt32_t mse = (pPredictedValueArray[i] - pTestDataHead[i+inputDim]);
				meanAbsoluteError += mse * mse;
			}
			meanAbsoluteError /= (flt32_t)outputDim;
			testAverageMeanAbsoluteError += meanAbsoluteError;
		}
		//-----------------------------------------------------------------------------
		//学習進行状況
		//-----------------------------------------------------------------------------
		if (trainCounter > 0) {
			trainAverageMeanAbsoluteError = trainAverageMeanAbsoluteError / (flt32_t)trainCounter;
			if (nTestData > 0) {
				testAverageMeanAbsoluteError = testAverageMeanAbsoluteError / (flt32_t)testCounter;
				sprintf(strInformation, "%u\t%6.3e\t%6.3e\n", epoch, trainAverageMeanAbsoluteError, testAverageMeanAbsoluteError);
			}
			else {
				sprintf(strInformation, "%u\t%6.3e\t-\n", epoch, trainAverageMeanAbsoluteError);
			}
			SAVE_LOG_WITHOUT_RETURN(strInformation);
		}
	}
	//================================================================
	//モデルの保存
	//================================================================
	pf_model = fopen(extModelInfo.modelFileName, "wb");
	if (pf_model != NULL) {
		fwrite(pModelImage, sizeof(uint32_t),sizeOfModelImageIn32BitWord, pf_model);
		fclose(pf_model);
	}
	//================================================================
	//教師データ評価
	//================================================================
	trainCounter = 0;
	totalTrainPredictedValueRatio = 0.0f;
	for (j = 0; j < nTrainData; j++) {
		//-------------------------------------------------------------------------
		//評価データ
		//-------------------------------------------------------------------------
		dataIndex = j;
		pTrainDataHead = pTrainingData + dataIndex * (outputDim + inputDim);
		//-------------------------------------------------------------------------
		//テスト数カウント
		//-------------------------------------------------------------------------
		trainCounter++;
		//-------------------------------------------------------------------------
		//予測
		//-------------------------------------------------------------------------
		fStatus = SequentialNet_predict(hModel, pTrainDataHead, inputDim);
		if (fStatus == FALSE) {
			return 1;
		}
		for (i = 0; i < outputDim; i++) {
			fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
			if (fStatus == FALSE) {
				printf("error, getting prediction value\n");
				return 1;
			}
			predictedValueRatio = fabs(pTrainDataHead[inputDim]-predectedValue) / pTrainDataHead[inputDim] * 100.0f;
			totalTrainPredictedValueRatio += predictedValueRatio;
		}
	}
	totalTrainPredictedValueRatio /= (flt32_t)trainCounter;
	//================================================================
	//テストデータ評価
	//================================================================
	testCounter = 0;
	totalTestPredictedValueRatio = 0.0f;
	for (j = 0; j < nTestData; j++) {
		//-------------------------------------------------------------------------
		//評価データ
		//-------------------------------------------------------------------------
		dataIndex = j;
		pTestDataHead = pTestData + dataIndex * (outputDim + inputDim);
		//-------------------------------------------------------------------------
		//テスト数カウント
		//-------------------------------------------------------------------------
		testCounter++;
		//-------------------------------------------------------------------------
		//予測
		//-------------------------------------------------------------------------
		fStatus = SequentialNet_predict(hModel, pTestDataHead, inputDim);
		if (fStatus == FALSE) {
			return 1;
		}
		for (i = 0; i < outputDim; i++) {
			fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
			if (fStatus == FALSE) {
				printf("error, getting prediction value\n");
				return 1;
			}
			predictedValueRatio = fabs(pTestDataHead[inputDim]-predectedValue) / pTestDataHead[inputDim] * 100.0f;
			totalTestPredictedValueRatio += predictedValueRatio;
		}
	}
	totalTestPredictedValueRatio /= (flt32_t)testCounter;
	sprintf(strInformation, "average difference between prediction and real> train : %5.1f(%%)\ttest : %5.1f(%%)\n", totalTrainPredictedValueRatio,totalTestPredictedValueRatio);
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	//LOGファイルクローズ
	CLOSE_LOG_FILE();
	//================================================================
	//メモリ解放
	//================================================================
	free(pModelImage);
	free(pSequentialModelWorkArea);
	free(pTrainingData);
	free(pTestData);
	free(pPredictedValueArray);
	free(pTeacherValueArray);
	free(pDLossArray);
	free(pRundumValueGenerator);
	return 0;
}
