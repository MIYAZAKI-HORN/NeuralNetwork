#ifndef FFT_ENGINE_H
#define FFT_ENGINE_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"

//FFTタイプ定義
typedef enum tagFFTType {
	FFT_TYPE_REGULAR = 1,
	FFT_TYPE_INVERSE
} FFTType;

//窓関数タイプ定義
typedef enum tagFFTWindowFunctionType {
	FFT_WINDOW_FUNCTION_TYPE_NONE = 0,
	FFT_WINDOW_FUNCTION_TYPE_HANNING,
	FFT_WINDOW_FUNCTION_TYPE_HAMMING
} FFTWindowFunctionType;

//エンジンハンドル型
typedef void* FFTEngineHandle;

uint32_t				FFTEngine_getSizeIn32BitWord(uint32_t vectorSize, FFTWindowFunctionType windowFunctionType);
FFTEngineHandle			FFTEngine_construct(uint32_t vectorSize, FFTWindowFunctionType windowFunctionType, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord);
bool_t					FFTEngine_perform(FFTEngineHandle hEngine,flt32_t* pRealPartData, flt32_t* pImagPartData, uint32_t vectorSize, FFTType typeOfFFT,bool_t fApplyWIndowFunction);
uint32_t				FFTEngine_getVectrSize(FFTEngineHandle hEngine);
FFTWindowFunctionType	FFTEngine_getWindowFunctionType(FFTEngineHandle hEngine);

#ifdef __cplusplus
}
#endif

#endif

