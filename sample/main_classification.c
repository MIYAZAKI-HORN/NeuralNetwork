//**********************************************************************************
//
//	自作ニューラルネットワークプログラム利用サンプルプログラム（分類問題）
// 
//	内容：
//	C言語記述の自作ニューラルネットワーク（シーケンシャル）を利用して分類問題に対応します。
//	全結合型ニューラルネットワークまたは、畳み込み型ニューラルネットワークを構築し、
//	ニューラルネットワークのパラメタに初期値を与え、指定したepoch数学習を行い、
//	学習データと検証データでの、クロスエントロピー誤差と識別率を逐次表示します。
// 
//	検証データ：
//	MNIST　CSV形式　データ系列は以下の様な並びを想定しています 
// label,pixel1,pixel2,pixel3,pixel4,pixel5,pixel6,pixel7,pixel8,pixel9,pixel10,pixel11,pixel12,pixel13,pixel14,pixel15,pixel16,pixel17,pixel18,pixel19,pixel20,pixel21,pixel22,pixel23,pixel24,pixel25,pixel26,pixel27,pixel28,pixel29,pixel30,pixel31,pixel32,pixel33,pixel34,pixel35,pixel36,pixel37,pixel38,pixel39,pixel40,pixel41,pixel42,pixel43,pixel44,pixel45,pixel46,pixel47,pixel48,pixel49,pixel50,pixel51,pixel52,pixel53,pixel54,pixel55,pixel56,pixel57,pixel58,pixel59,pixel60,pixel61,pixel62,pixel63,pixel64,pixel65,pixel66,pixel67,pixel68,pixel69,pixel70,pixel71,pixel72,pixel73,pixel74,pixel75,pixel76,pixel77,pixel78,pixel79,pixel80,pixel81,pixel82,pixel83,pixel84,pixel85,pixel86,pixel87,pixel88,pixel89,pixel90,pixel91,pixel92,pixel93,pixel94,pixel95,pixel96,pixel97,pixel98,pixel99,pixel100,pixel101,pixel102,pixel103,pixel104,pixel105,pixel106,pixel107,pixel108,pixel109,pixel110,pixel111,pixel112,pixel113,pixel114,pixel115,pixel116,pixel117,pixel118,pixel119,pixel120,pixel121,pixel122,pixel123,pixel124,pixel125,pixel126,pixel127,pixel128,pixel129,pixel130,pixel131,pixel132,pixel133,pixel134,pixel135,pixel136,pixel137,pixel138,pixel139,pixel140,pixel141,pixel142,pixel143,pixel144,pixel145,pixel146,pixel147,pixel148,pixel149,pixel150,pixel151,pixel152,pixel153,pixel154,pixel155,pixel156,pixel157,pixel158,pixel159,pixel160,pixel161,pixel162,pixel163,pixel164,pixel165,pixel166,pixel167,pixel168,pixel169,pixel170,pixel171,pixel172,pixel173,pixel174,pixel175,pixel176,pixel177,pixel178,pixel179,pixel180,pixel181,pixel182,pixel183,pixel184,pixel185,pixel186,pixel187,pixel188,pixel189,pixel190,pixel191,pixel192,pixel193,pixel194,pixel195,pixel196,pixel197,pixel198,pixel199,pixel200,pixel201,pixel202,pixel203,pixel204,pixel205,pixel206,pixel207,pixel208,pixel209,pixel210,pixel211,pixel212,pixel213,pixel214,pixel215,pixel216,pixel217,pixel218,pixel219,pixel220,pixel221,pixel222,pixel223,pixel224,pixel225,pixel226,pixel227,pixel228,pixel229,pixel230,pixel231,pixel232,pixel233,pixel234,pixel235,pixel236,pixel237,pixel238,pixel239,pixel240,pixel241,pixel242,pixel243,pixel244,pixel245,pixel246,pixel247,pixel248,pixel249,pixel250,pixel251,pixel252,pixel253,pixel254,pixel255,pixel256,pixel257,pixel258,pixel259,pixel260,pixel261,pixel262,pixel263,pixel264,pixel265,pixel266,pixel267,pixel268,pixel269,pixel270,pixel271,pixel272,pixel273,pixel274,pixel275,pixel276,pixel277,pixel278,pixel279,pixel280,pixel281,pixel282,pixel283,pixel284,pixel285,pixel286,pixel287,pixel288,pixel289,pixel290,pixel291,pixel292,pixel293,pixel294,pixel295,pixel296,pixel297,pixel298,pixel299,pixel300,pixel301,pixel302,pixel303,pixel304,pixel305,pixel306,pixel307,pixel308,pixel309,pixel310,pixel311,pixel312,pixel313,pixel314,pixel315,pixel316,pixel317,pixel318,pixel319,pixel320,pixel321,pixel322,pixel323,pixel324,pixel325,pixel326,pixel327,pixel328,pixel329,pixel330,pixel331,pixel332,pixel333,pixel334,pixel335,pixel336,pixel337,pixel338,pixel339,pixel340,pixel341,pixel342,pixel343,pixel344,pixel345,pixel346,pixel347,pixel348,pixel349,pixel350,pixel351,pixel352,pixel353,pixel354,pixel355,pixel356,pixel357,pixel358,pixel359,pixel360,pixel361,pixel362,pixel363,pixel364,pixel365,pixel366,pixel367,pixel368,pixel369,pixel370,pixel371,pixel372,pixel373,pixel374,pixel375,pixel376,pixel377,pixel378,pixel379,pixel380,pixel381,pixel382,pixel383,pixel384,pixel385,pixel386,pixel387,pixel388,pixel389,pixel390,pixel391,pixel392,pixel393,pixel394,pixel395,pixel396,pixel397,pixel398,pixel399,pixel400,pixel401,pixel402,pixel403,pixel404,pixel405,pixel406,pixel407,pixel408,pixel409,pixel410,pixel411,pixel412,pixel413,pixel414,pixel415,pixel416,pixel417,pixel418,pixel419,pixel420,pixel421,pixel422,pixel423,pixel424,pixel425,pixel426,pixel427,pixel428,pixel429,pixel430,pixel431,pixel432,pixel433,pixel434,pixel435,pixel436,pixel437,pixel438,pixel439,pixel440,pixel441,pixel442,pixel443,pixel444,pixel445,pixel446,pixel447,pixel448,pixel449,pixel450,pixel451,pixel452,pixel453,pixel454,pixel455,pixel456,pixel457,pixel458,pixel459,pixel460,pixel461,pixel462,pixel463,pixel464,pixel465,pixel466,pixel467,pixel468,pixel469,pixel470,pixel471,pixel472,pixel473,pixel474,pixel475,pixel476,pixel477,pixel478,pixel479,pixel480,pixel481,pixel482,pixel483,pixel484,pixel485,pixel486,pixel487,pixel488,pixel489,pixel490,pixel491,pixel492,pixel493,pixel494,pixel495,pixel496,pixel497,pixel498,pixel499,pixel500,pixel501,pixel502,pixel503,pixel504,pixel505,pixel506,pixel507,pixel508,pixel509,pixel510,pixel511,pixel512,pixel513,pixel514,pixel515,pixel516,pixel517,pixel518,pixel519,pixel520,pixel521,pixel522,pixel523,pixel524,pixel525,pixel526,pixel527,pixel528,pixel529,pixel530,pixel531,pixel532,pixel533,pixel534,pixel535,pixel536,pixel537,pixel538,pixel539,pixel540,pixel541,pixel542,pixel543,pixel544,pixel545,pixel546,pixel547,pixel548,pixel549,pixel550,pixel551,pixel552,pixel553,pixel554,pixel555,pixel556,pixel557,pixel558,pixel559,pixel560,pixel561,pixel562,pixel563,pixel564,pixel565,pixel566,pixel567,pixel568,pixel569,pixel570,pixel571,pixel572,pixel573,pixel574,pixel575,pixel576,pixel577,pixel578,pixel579,pixel580,pixel581,pixel582,pixel583,pixel584,pixel585,pixel586,pixel587,pixel588,pixel589,pixel590,pixel591,pixel592,pixel593,pixel594,pixel595,pixel596,pixel597,pixel598,pixel599,pixel600,pixel601,pixel602,pixel603,pixel604,pixel605,pixel606,pixel607,pixel608,pixel609,pixel610,pixel611,pixel612,pixel613,pixel614,pixel615,pixel616,pixel617,pixel618,pixel619,pixel620,pixel621,pixel622,pixel623,pixel624,pixel625,pixel626,pixel627,pixel628,pixel629,pixel630,pixel631,pixel632,pixel633,pixel634,pixel635,pixel636,pixel637,pixel638,pixel639,pixel640,pixel641,pixel642,pixel643,pixel644,pixel645,pixel646,pixel647,pixel648,pixel649,pixel650,pixel651,pixel652,pixel653,pixel654,pixel655,pixel656,pixel657,pixel658,pixel659,pixel660,pixel661,pixel662,pixel663,pixel664,pixel665,pixel666,pixel667,pixel668,pixel669,pixel670,pixel671,pixel672,pixel673,pixel674,pixel675,pixel676,pixel677,pixel678,pixel679,pixel680,pixel681,pixel682,pixel683,pixel684,pixel685,pixel686,pixel687,pixel688,pixel689,pixel690,pixel691,pixel692,pixel693,pixel694,pixel695,pixel696,pixel697,pixel698,pixel699,pixel700,pixel701,pixel702,pixel703,pixel704,pixel705,pixel706,pixel707,pixel708,pixel709,pixel710,pixel711,pixel712,pixel713,pixel714,pixel715,pixel716,pixel717,pixel718,pixel719,pixel720,pixel721,pixel722,pixel723,pixel724,pixel725,pixel726,pixel727,pixel728,pixel729,pixel730,pixel731,pixel732,pixel733,pixel734,pixel735,pixel736,pixel737,pixel738,pixel739,pixel740,pixel741,pixel742,pixel743,pixel744,pixel745,pixel746,pixel747,pixel748,pixel749,pixel750,pixel751,pixel752,pixel753,pixel754,pixel755,pixel756,pixel757,pixel758,pixel759,pixel760,pixel761,pixel762,pixel763,pixel764,pixel765,pixel766,pixel767,pixel768,pixel769,pixel770,pixel771,pixel772,pixel773,pixel774,pixel775,pixel776,pixel777,pixel778,pixel779,pixel780,pixel781,pixel782,pixel783,pixel784
//5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
//0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 159, 253, 159, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 252, 237, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 227, 253, 252, 239, 233, 252, 57, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 60, 224, 252, 253, 252, 202, 84, 252, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 252, 252, 252, 253, 252, 252, 96, 189, 253, 167, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 238, 253, 253, 190, 114, 253, 228, 47, 79, 255, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 179, 12, 75, 121, 21, 0, 0, 253, 243, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 165, 253, 233, 208, 84, 0, 0, 0, 0, 0, 0, 253, 252, 165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 178, 252, 240, 71, 19, 28, 0, 0, 0, 0, 0, 0, 253, 252, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57, 252, 252, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 253, 196, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 246, 252, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 148, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 230, 25, 0, 0, 0, 0, 0, 0, 0, 0, 7, 135, 253, 186, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 223, 0, 0, 0, 0, 0, 0, 0, 0, 7, 131, 252, 225, 71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 145, 0, 0, 0, 0, 0, 0, 0, 48, 165, 252, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 253, 225, 0, 0, 0, 0, 0, 0, 114, 238, 253, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 249, 146, 48, 29, 85, 178, 225, 253, 223, 167, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 252, 252, 252, 229, 215, 252, 252, 252, 196, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 199, 252, 252, 253, 252, 252, 233, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 128, 252, 253, 252, 141, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
//..................................................................................
//
//	前処理：
//	データの値が0～255までの数値となっていますので。255で除算して0.0～1.0の浮動小数点データにしています。
// 
//	注意：
//	シーケンシャルニューラルネットワークの最終層の活性化関数をsoftmaxとしています。
//	誤差逆伝搬の際に与える分類問題のロス計算は、softmax＋クロスエントロピーとした場合、（正解値-予測値）と簡単になるため、
//	ニューラルネットワークの誤差逆伝搬時には、最終層のsoftmaxはスキップする仕様となっています。
//
//**********************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "SequentialNet.h"
#include "RandomValueGenerator.h"
#include "NeuralNetModelConstructor.h"
#include "NeuralNetLayerActivation.h"
#include "NeuralNetLayerBatchNormalization.h"
#include "LOG_Function.h"

//-----------------------------------------------------------------------------------------------------
//テストシーケンシャルモデルタイプ
//-----------------------------------------------------------------------------------------------------
typedef enum tagEvaluationModelType {
	EVALUATION_MODEL_TYPE_DENSE		= 1,
	EVALUATION_MODEL_TYPE_CONV2D	= 2,
} EvaluationModelType;

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
//学習関連パラメタ
//-----------------------------------------------------------------------------------------------------
#define EPOCHS							(100)
#define BATCH_SIZE						(100)
#define BATCH_NORMALIZATION_MOMENTUM	(0.999f)

//-----------------------------------------------------------------------------------------------------
//Leaky ReLU活性化関数の負値勾配
//-----------------------------------------------------------------------------------------------------
#define RELU_ACTIVATION_NEGATIVE_SLOOP	(0.0f)

//-----------------------------------------------------------------------------------------------------
//データファイル定義
//-----------------------------------------------------------------------------------------------------
#define DATA_FOLDER (".\\Data\\")

#define TRAIN_IMAGE_DATA_FILE_NAME		("MNIST\\MNIST_trainHalf.csv")
#define TEST_IMAGE_DATA_FILE_NAME		("MNIST\\MNIST_test.csv")
#define MODEL_FILE_NAME					("mnist.nnm")

#define SINGLE_DATA_SIZE	(785)

#define MAX_FILE_NAME_LENGTH	(500)

//-----------------------------------------------------------------------------------------------------
//画像データ読み込み関数
//-----------------------------------------------------------------------------------------------------
bool_t
readSingleImage(FILE* fp, flt32_t* pData,uint32_t* pDataCount) {
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
setModelInformation(EvaluationModelType modelType, ModelInformationEx* pNetworkInfo)
{
	int i;
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
	switch (modelType) {
	case EVALUATION_MODEL_TYPE_DENSE:
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
		//モデル定義
		//---------------------------------------
		sequential_model_header(pModel, 28, 28, 1);	//入力　28x28x1　Channel last
		dense(pModel, 30);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		dense(pModel, 40);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		dense(pModel, 10);
		activation(pModel, NEURAL_NET_ACTIVATION_SOFTMAX);
		break;
	case EVALUATION_MODEL_TYPE_CONV2D:
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
		//モデル定義
		//---------------------------------------
		sequential_model_header(pModel, 28, 28, 1);	//入力　28x28x1　Channel last
		conv2d(pModel, 10, 3, 3, 1, 1, FALSE);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		max_pooling2d(pModel, 3, 3, 3, 3);
		conv2d(pModel, 4, 3, 3, 1, 1, TRUE);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		conv2d(pModel, 4, 3, 3, 1, 1, TRUE);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		dense(pModel, 40);
		activation(pModel, NEURAL_NET_ACTIVATION_RELU);
		dense(pModel, 10);
		activation(pModel, NEURAL_NET_ACTIVATION_SOFTMAX);		
		break;
	}
	return TRUE;
}

int main(int argc, char* argv[])
{
	uint32_t				i,j;
	bool_t					fStatus;
	uint32_t*				pModelImage = NULL;
	uint32_t				sizeOfModelImageIn32BitWord;
	ModelInformationEx		extModelInfo;
	EvaluationModelType		evaluationModelType;
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
	uint32_t				trainCorrectCounter;
	uint32_t				testCorrectCounter;
	uint32_t				dataCounter;
	uint32_t				random;
	uint32_t				dataIndex;
	uint32_t				labelNumber;
	flt32_t					predectedValue;
	flt32_t					maxProbabilty;
	uint32_t				maxIndex;
	flt32_t*				pPredictedValueArray = NULL;
	flt32_t*				pTeacherValueArray = NULL;
	flt32_t*				pDLossArray = NULL;
	flt32_t					crossEntropyError;
	flt32_t					trainAverageCrossEntropyError;
	flt32_t					trainAccuracy;
	flt32_t					testAverageCrossEntropyError;
	flt32_t					testAccuracy;
	uint32_t				numOfLayers;
	NetLayerType			layerType;
	handle_t				hLayer;
	char					strInformation[200];
	flt32_t					maxTeacherValue;
	flt32_t					minPredictedValue;
	flt32_t					maxPredictedValue;
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
	//evaluationModelType = EVALUATION_MODEL_TYPE_DENSE;
	evaluationModelType = EVALUATION_MODEL_TYPE_CONV2D;
	setModelInformation(evaluationModelType,&extModelInfo);
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
	//================================================================
	//モデルデータの入出力次元を取得
	//================================================================
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
	nTrainData = 60000;
	pTrainingData = (float*)malloc(nTrainData * sizeof(float) * SINGLE_DATA_SIZE);
	pTrainDataHead = pTrainingData;
	//1行目は項目名なので読み飛ばす
	readSingleImage(fp_train, NULL,NULL);
	trainCounter = 0;
	while (readSingleImage(fp_train, pTrainDataHead,&dataCounter)) {
		if (++trainCounter == nTrainData) {
			break;
		}
		if (dataCounter != SINGLE_DATA_SIZE) {
			continue;
		}
		//最初のデータはラベルデータなのでそのまま
		//画像データ部分を0～1.0の値に正規化する
		for (i = 1; i < SINGLE_DATA_SIZE; i++) {
			pTrainDataHead[i] /= 255.0f;
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
		nTestData = 10000;
		pTestData = (float*)malloc(nTestData * sizeof(float) * SINGLE_DATA_SIZE);
		pTestDataHead = pTestData;
		//1行目は項目名なので読み飛ばす
		readSingleImage(fp_test, NULL,NULL);
		testCounter = 0;
		while (readSingleImage(fp_test, pTestDataHead, &dataCounter)) {
			if (++testCounter == nTestData) {
				break;
			}
			if (dataCounter != SINGLE_DATA_SIZE) {
				continue;
			}
			//最初のデータはラベルデータなのでそのまま
			//画像データ部分を0～1.0の値に正規化する
			for (i = 1; i < SINGLE_DATA_SIZE; i++) {
				pTestDataHead[i] /= 255.0f;
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
	sprintf(strInformation,"epoch\tcross entropy\taccuracy\tcross entropy(test)\taccuracy(test)\n");
	SAVE_LOG_WITHOUT_RETURN(strInformation);
	numberOfEpochs = EPOCHS;
	for (epoch = 1; epoch <= numberOfEpochs; epoch++) {
		maxTeacherValue = -1.e30;
		minPredictedValue = 1.e30;
		maxPredictedValue = -1.e30;
		//-------------------------------------------------------------------------
		//学習データ全てを利用して学習
		//-------------------------------------------------------------------------
		trainCounter = 0;
		trainCorrectCounter = 0;
		trainAverageCrossEntropyError = 0.0f;
		for (j = 0; j < nTrainData; j++) {
			//-------------------------------------------------------------------------
			//　乱数により学習データを選択
			//-------------------------------------------------------------------------
			random = RandomValueGenerator_getIntegerValue(hRandomValueGenerator);
			dataIndex = random% nTrainData;
			//-------------------------------------------------------------------------
			//one hot出力
			//-------------------------------------------------------------------------
			pTrainDataHead = pTrainingData + dataIndex * (1 + inputDim);
			//-------------------------------------------------------------------------
			//クラス番号
			//-------------------------------------------------------------------------
			labelNumber = (uint32_t)(*pTrainDataHead);
			pTrainDataHead += 1;
			if (labelNumber >= outputDim) {
				printf("error, bad label number\n");
				return 1;
			}
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
			//-------------------------------------------------------------------------
			//確率値および最大確率出力のクラスインデックスの取得
			//-------------------------------------------------------------------------
			maxProbabilty = 0.0f;
			maxIndex = 0;
			for (i = 0; i < outputDim; i++) {
				fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
				if (fStatus == FALSE) {
					printf("error, getting prediction value\n");
					return 1;
				}
				pPredictedValueArray[i] = predectedValue;
				if (maxProbabilty < predectedValue) {
					maxProbabilty = predectedValue;
					maxIndex = i;
				}
			}
			//-------------------------------------------------------------------------
			//正解個数
			//-------------------------------------------------------------------------
			if (labelNumber == maxIndex) {
				trainCorrectCounter++;
			}
			//-------------------------------------------------------------------------
			//教師データ作成（one hot表現）
			//-------------------------------------------------------------------------
			for (i = 0; i < outputDim; i++) {
				pTeacherValueArray[i] = 0.0f;
			}
			pTeacherValueArray[labelNumber] = 1.0f;
			//-------------------------------------------------------------------------
			//cross entropy error計算
			//-------------------------------------------------------------------------
			crossEntropyError = 0.0f;
			for (i = 0; i < outputDim; i++) {
				crossEntropyError += pTeacherValueArray[i] * (flt32_t)log(pPredictedValueArray[i] + 1.0e-8f);
			}
			crossEntropyError *= -1.0f;
			trainAverageCrossEntropyError += crossEntropyError;
			//-------------------------------------------------------------------------
			//最終層がsoftmaxとした場合(softmax with cross entropy error)のLOSS値の微分
			//-------------------------------------------------------------------------
			for (i = 0; i < outputDim; i++) {
				pDLossArray[i] = pPredictedValueArray[i] - pTeacherValueArray[i];
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
		//-----------------------------------------------------------------------------
		//学習進行状況
		//-----------------------------------------------------------------------------
		if (trainCounter > 0) {
			trainAverageCrossEntropyError = trainAverageCrossEntropyError / (flt32_t)trainCounter;
			trainAccuracy = (flt32_t)trainCorrectCounter * 100.0f / (flt32_t)trainCounter;
			if (nTestData > 0) {
				sprintf(strInformation,"%u\t%10.3f\t%5.1f\t", epoch, trainAverageCrossEntropyError, trainAccuracy);
			}
			else {
				sprintf(strInformation, "%u\t%10.3f\t%5.1f\n", epoch, trainAverageCrossEntropyError, trainAccuracy);
			}
			SAVE_LOG_WITHOUT_RETURN(strInformation);
		}
		//-------------------------------------------------------------------------
		//テストデータ評価
		//-------------------------------------------------------------------------
		testCounter = 0;
		testCorrectCounter = 0;
		testAverageCrossEntropyError = 0.0f;
		for (j = 0; j < nTestData; j++) {
			//-------------------------------------------------------------------------
			//評価データ
			//-------------------------------------------------------------------------
			dataIndex = j;
			pTestDataHead = pTestData + dataIndex * (1 + inputDim);
			//-------------------------------------------------------------------------
			//クラス番号
			//-------------------------------------------------------------------------
			labelNumber = (uint32_t)(*pTestDataHead++);
			if (labelNumber >= outputDim) {
				printf("error, bad label number\n");
				return 1;
			}
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
			//-------------------------------------------------------------------------
			//最大確率出力インデックスの取得
			//-------------------------------------------------------------------------
			maxProbabilty = 0.0f;
			maxIndex = 0;
			for (i = 0; i < outputDim; i++) {
				fStatus = SequentialNet_getPrediction(hModel, i, &predectedValue);
				if (fStatus == FALSE) {
					return 1;
				}
				pPredictedValueArray[i] = predectedValue;
				if (maxProbabilty < predectedValue) {
					maxProbabilty = predectedValue;
					maxIndex = i;
				}
			}
			//-------------------------------------------------------------------------
			//正解個数
			//-------------------------------------------------------------------------
			if (labelNumber == maxIndex) {
				testCorrectCounter++;
			}
			//-------------------------------------------------------------------------
			//テスト用のcross entropy error計算（参考値）
			//-------------------------------------------------------------------------
			for (i = 0; i < outputDim; i++) {
				pTeacherValueArray[i] = 0.0f;
			}
			pTeacherValueArray[labelNumber] = 1.0f;
			crossEntropyError = 0.0f;
			for (i = 0; i < outputDim; i++) {
				crossEntropyError += pTeacherValueArray[i] * (flt32_t)log(pPredictedValueArray[i] + 1.0e-8f);
			}
			crossEntropyError *= -1.0f;
			testAverageCrossEntropyError += crossEntropyError;
		}
		//-----------------------------------------------------------------------------
		//テスト結果表示
		//-----------------------------------------------------------------------------
		if (testCounter > 0) {
			testAverageCrossEntropyError = testAverageCrossEntropyError / (flt32_t)testCounter;
			testAccuracy = (flt32_t)testCorrectCounter * 100.0f / (flt32_t)testCounter;
			sprintf(strInformation,"%5.3f\t%5.1f\n", testAverageCrossEntropyError, testAccuracy);
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
