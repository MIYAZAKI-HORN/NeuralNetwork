#ifndef YOLO_DFL_COMPUTATION_H
#define YOLO_DFL_COMPUTATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include "STDTypeDefinition.h"

//=====================================================================================
// YOLO DFL 学習測定値構造体
//=====================================================================================
typedef struct tagDFL_Monitor {
	flt32_t		lossTotal;
	flt32_t		lossCIoU;
	flt32_t		lossDF;
	flt32_t		lossClassProb;
	flt32_t		lossNoObj;
	flt32_t		classAccuracy;
	flt32_t		averageIoU;
} DFL_Monitor;

//=====================================================================================
// YOLO DFLトレーニング関数
//=====================================================================================
bool_t
YORO_DFL_Training(
	int32_t		gridSize,
	int32_t		distributionMapSize,
	int32_t		nClasses,
	flt32_t*	pTeacherValueHead,
	flt32_t*	pPredictedValueHead,
	flt32_t*	pDLossArray,
	flt32_t		lamda_ciou,
	flt32_t		lamda_dfl,
	flt32_t		lamda_class,
	DFL_Monitor* pMeasuredValues);

#ifdef __cplusplus
}
#endif

#endif

