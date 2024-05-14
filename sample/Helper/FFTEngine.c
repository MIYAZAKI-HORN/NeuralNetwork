#include <math.h>
#include "FFTEngine.h"

typedef struct tagFFTEngineD {
	uint32_t				vectorSize;
	flt32_t*				pSinTable;
	uint32_t*				pBitReverseTable;
	flt32_t				scalingFactor;
	FFTWindowFunctionType	windowFunctionType;
	flt32_t*				pWindowFunction;
} FFTEngine;

#define PAI (3.14159265358979)

static
void
FFTEngine_generateTables(FFTEngine* pEngine) {
	int i,j,k;
	//-----------------------------------------------------
	//
	//-----------------------------------------------------
	int 	n2 	= pEngine->vectorSize/2;
	int 	n4 	= pEngine->vectorSize/4;
	int 	n8 	= pEngine->vectorSize/8;
	flt32_t 	temp 	= sin(PAI/(flt32_t)(pEngine->vectorSize));
	flt32_t	dc	= 2.*temp*temp;
	flt32_t	ds	= sqrt(dc*(2.0f-dc));
	flt32_t	cosValue	= pEngine->pSinTable[n4] = 1.0f;
	flt32_t	sineValue	= pEngine->pSinTable[0]  = 0.0f;
	temp      	= 2.*dc;
	for(i=1;i<n8;i++) {
		cosValue 	-= dc;
		dc 	+= temp*cosValue;
		sineValue 	+= ds;
		ds	-= temp*sineValue;
		pEngine->pSinTable[i] 	= sineValue;
		pEngine->pSinTable[n4-i] = cosValue;
	}
	if (n8 != 0) {
		pEngine->pSinTable[n8] = sqrt(0.5f);
	}
	for (i = 0; i < n4; i++) {
		pEngine->pSinTable[n2 - i] = pEngine->pSinTable[i];
	}
	for (i = 0; i < n2 + n4; i++) {
		pEngine->pSinTable[i + n2] = -pEngine->pSinTable[i];
	}
	//-----------------------------------------------------
	//
	//-----------------------------------------------------
	i = 0;
	j = 0;
	while(1) {
		pEngine->pBitReverseTable[i]	= j;
		if( ++i >= pEngine->vectorSize) {
			break;
		}
		k= n2;
		while( k <= j ) {
			j -= k;
			k /= 2;
		}
		j += k;
    }
	pEngine->scalingFactor = 1.0f / sqrt((flt32_t)(pEngine->vectorSize));
}

//-----------------------------------------------------------------------------------------
// Generate window function
//-----------------------------------------------------------------------------------------
static
void
FFTEngine_generateWindowFunctionTables(FFTEngine* pEngine) {
	uint32_t	i;
	flt32_t	alpha;
	flt32_t	beta;
	flt32_t	deltaAngle;
	flt32_t	angle;
	flt32_t	factor;
	switch (pEngine->windowFunctionType) {
	case FFT_WINDOW_FUNCTION_TYPE_NONE:
		pEngine->pWindowFunction = NULL;
		return;
	case FFT_WINDOW_FUNCTION_TYPE_HANNING:
		alpha = 0.5f;
		break;
	case FFT_WINDOW_FUNCTION_TYPE_HAMMING:
		alpha = 0.54f;
		break;
	default:
		// error
		alpha = 1.0f;
		break;
	}
	beta = 1.0f - alpha;
	deltaAngle = 2.0f * 3.14159265 / (flt32_t)pEngine->vectorSize;
	for (i = 0; i < pEngine->vectorSize; i++) {
		angle = deltaAngle * (flt32_t)i;
		factor = alpha - beta * cos(angle);
		pEngine->pWindowFunction[i] = factor;
	}
}

uint32_t
FFTEngine_getSizeIn32BitWord(uint32_t vectorSize, FFTWindowFunctionType windowFunctionType) {
	uint32_t	sizeIn32BitWord;
	uint32_t	workAreaSizeIn32BitWord = 0;
	uint32_t	tableSize;
	uint32_t	windowTableSize;
	// engine
	sizeIn32BitWord = (sizeof(FFTEngine) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	workAreaSizeIn32BitWord += sizeIn32BitWord;
	// sine table
	tableSize = vectorSize + vectorSize / 4;
	sizeIn32BitWord = (sizeof(flt32_t) * tableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	workAreaSizeIn32BitWord += sizeIn32BitWord;
	// bit reverse table
	tableSize = vectorSize;
	sizeIn32BitWord = (sizeof(flt32_t) * tableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	workAreaSizeIn32BitWord += sizeIn32BitWord;
	// window function table
	switch (windowFunctionType) {
	case FFT_WINDOW_FUNCTION_TYPE_NONE:
		windowTableSize = 0;
		break;
	case FFT_WINDOW_FUNCTION_TYPE_HANNING:
		windowTableSize = vectorSize;
		break;
	case FFT_WINDOW_FUNCTION_TYPE_HAMMING:
		windowTableSize = vectorSize;
		break;
	default:
		windowTableSize = vectorSize;
		break;
	}
	sizeIn32BitWord = (sizeof(flt32_t) * windowTableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	workAreaSizeIn32BitWord += sizeIn32BitWord;
	return workAreaSizeIn32BitWord;
}

FFTEngineHandle	
FFTEngine_construct(uint32_t vectorSize, FFTWindowFunctionType windowFunctionType, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord) {
	uint32_t	sizeIn32BitWord;
	uint32_t	workAreaSizeIn32BitWord = 0;
	uint32_t	tableSize;
	uint32_t	windowTableSize;
	uint32_t*	pWorkAreaHead = pWorkArea;
	FFTEngine*	pEngine;
	//
	workAreaSizeIn32BitWord = FFTEngine_getSizeIn32BitWord(vectorSize, windowFunctionType);
	if (sizeOfWorkAreaIn32BitWord < workAreaSizeIn32BitWord) {
		return NULL;
	}
	// engine
	sizeIn32BitWord = (sizeof(FFTEngine) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	pEngine = (FFTEngine*)pWorkAreaHead;
	pWorkAreaHead += sizeIn32BitWord;
	// sine table
	tableSize = vectorSize + vectorSize / 4;
	sizeIn32BitWord = (sizeof(flt32_t) * tableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	pEngine->pSinTable = (flt32_t*)pWorkAreaHead;
	pWorkAreaHead += sizeIn32BitWord;
	// bit reverse table
	tableSize = vectorSize;
	sizeIn32BitWord = (sizeof(flt32_t) * tableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	pEngine->pBitReverseTable = (uint32_t*)pWorkAreaHead;
	pWorkAreaHead += sizeIn32BitWord;
	// window function table
	switch (windowFunctionType) {
	case FFT_WINDOW_FUNCTION_TYPE_NONE:
		windowTableSize = 0;
		break;
	case FFT_WINDOW_FUNCTION_TYPE_HANNING:
		windowTableSize = vectorSize;
		break;
	case FFT_WINDOW_FUNCTION_TYPE_HAMMING:
		windowTableSize = vectorSize;
		break;
	default:
		windowTableSize = vectorSize;
		break;
	}
	sizeIn32BitWord = (sizeof(flt32_t) * windowTableSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
	pEngine->pWindowFunction = (flt32_t*)pWorkAreaHead;
	pWorkAreaHead += sizeIn32BitWord;
	// generate tables
	pEngine->vectorSize = vectorSize;
	pEngine->windowFunctionType = windowFunctionType;
	FFTEngine_generateTables(pEngine);
	FFTEngine_generateWindowFunctionTables(pEngine);
	return (FFTEngineHandle)pEngine;
}

bool_t			
FFTEngine_perform(FFTEngineHandle hEngine, flt32_t* pRealPartData, flt32_t* pImagPartData, uint32_t vectorSize, FFTType typeOfFFT, bool_t fApplyWIndowFunction) {
	FFTEngine* pEngine = (FFTEngine*)hEngine;
	uint32_t 	n4 	= pEngine->vectorSize/4;
	uint32_t 	i,j,k;
	uint32_t	h;
	uint32_t	d;
	uint32_t	k2;
	uint32_t	ik;
	flt32_t	temp;
	flt32_t	cosValue;
	flt32_t	sineValue;
	flt32_t 	dx;
	flt32_t 	dy;
	flt32_t*	pSinTable;
	uint32_t*	pBitReverseTable;
	flt32_t	scalingFactor;
	flt32_t*	pRealPart;
	flt32_t*	pImagPart;
	flt32_t*	pWindowFunction;
	flt32_t	sinValueSign;
	//-----------------------------------------------------
	//	generate sin/cos table
	//-----------------------------------------------------	
	if( pEngine->vectorSize != vectorSize ) {
		return FALSE;
    }
	//-----------------------------------------------------
	// apply window function
	//-----------------------------------------------------	
	if (fApplyWIndowFunction == TRUE) {
		switch(pEngine->windowFunctionType) {
		case FFT_WINDOW_FUNCTION_TYPE_NONE:
			return FALSE;
		case FFT_WINDOW_FUNCTION_TYPE_HANNING:
		case FFT_WINDOW_FUNCTION_TYPE_HAMMING:
			pWindowFunction	= pEngine->pWindowFunction;
			for (i = 0; i < vectorSize; i++) {
				pRealPartData[i] *= pWindowFunction[i];
				pImagPartData[i] *= pWindowFunction[i];
			}
			break;
		default:
			break;
		}
	}
	//-----------------------------------------------------
	// table
	//-----------------------------------------------------	
	pSinTable			= pEngine->pSinTable;
	pBitReverseTable	= pEngine->pBitReverseTable;
	scalingFactor		= pEngine->scalingFactor;
	//-----------------------------------------------------
	// FFT
	//-----------------------------------------------------	
	for(i=0;i<vectorSize;i++) {
		j 	= pBitReverseTable[i];
		if( i < j ) {
			// replace
			temp 		= pRealPartData[i];
			pRealPartData[i] 	= pRealPartData[j];
			pRealPartData[j]	= temp;
			temp		= pImagPartData[i];
			pImagPartData[i]	= pImagPartData[j];
			pImagPartData[j]	= temp;
		}
	}
	switch (typeOfFFT) {
	case FFT_TYPE_REGULAR:
		sinValueSign = 1.0f;
		break;
	case FFT_TYPE_INVERSE:
		sinValueSign = -1.0f;
		break;
	default:
		return FALSE;
		break;
	}
	for(k=1;k<vectorSize;k=k2) {
		h 	= 0;
		k2	= k + k;
		d	= vectorSize/k2;
		for(j=0;j<k;j++) {
			cosValue = pSinTable[h+n4];
			sineValue = sinValueSign * pSinTable[h];
			for(i=j;i<vectorSize;i+=k2) {
				ik = i + k;
				dx = sineValue*pImagPartData[ik] + cosValue*pRealPartData[ik];
				dy = cosValue*pImagPartData[ik]  - sineValue*pRealPartData[ik];
				pRealPartData[ik] 	= pRealPartData[i] - dx;
				pRealPartData[i]	+= dx;
				pImagPartData[ik]	= pImagPartData[i] - dy;
				pImagPartData[i]	+= dy;
			}
			h += d;
		}
    }
	//-----------------------------------------------------
	//	normalization
	//-----------------------------------------------------
	pRealPart	= pRealPartData;
	pImagPart	= pImagPartData;
	for(i=0;i<vectorSize;i++) {
		*pRealPart++ *= scalingFactor;
		*pImagPart++ *= scalingFactor;
	}
	return TRUE;
}

uint32_t
FFTEngine_getVectrSize(FFTEngineHandle hEngine) {
	FFTEngine* pEngine = (FFTEngine*)hEngine;
	return pEngine->vectorSize;
}

FFTWindowFunctionType
FFTEngine_getWindowFunctionType(FFTEngineHandle hEngine) {
	FFTEngine* pEngine = (FFTEngine*)hEngine;
	return pEngine->windowFunctionType;
}
