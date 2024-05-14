#include "STDTypeDefinition.h"
#include "RandomValueGenerator.h"
#include "NeuralNetLayerFunction.h"

//=====================================================================================
//  ジェネレーター構造体
//=====================================================================================
typedef struct tagRandomValueGenerator {
    uint32_t	a;
} RandomValueGenerator;

/* https://ja.wikipedia.org/wiki/Xorshift */
/* The state word must be initialized to non-zero */
inline
uint32_t 
xorshift32(RandomValueGenerator* pEngine)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = pEngine->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return pEngine->a = x;
}

//=====================================================================================
//  必要ワークエリアサイズ取得
//=====================================================================================
uint32_t
RandomValueGenerator_getSizeIn32BitWord(void) {
	uint32_t	sizeOfEngineIn32BitWord = 0;
	sizeOfEngineIn32BitWord = size_in_type(sizeof(RandomValueGenerator),uint32_t);
	return sizeOfEngineIn32BitWord;
}

//=====================================================================================
//  ジェネレーター構築
//=====================================================================================
handle_t
RandomValueGenerator_construct(uint32_t seed, uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord) {
	RandomValueGenerator* This;
    uint32_t   requiredSizeOfWorkAreaIn32BitWord;
    //----------------------------------------------------------------------------------
    //  ワークエリアをチェック
    //----------------------------------------------------------------------------------
    if (pWorkArea == NULL) {
        return NULL;
    }
    //----------------------------------------------------------------------------------
    //  作業領域サイズをチェック
    //----------------------------------------------------------------------------------
    requiredSizeOfWorkAreaIn32BitWord = RandomValueGenerator_getSizeIn32BitWord();
    if (sizeOfWorkAreaIn32BitWord < requiredSizeOfWorkAreaIn32BitWord) {
        return NULL;
    }
    This = (RandomValueGenerator*)pWorkArea;
    //----------------------------------------------------------------------------------
    //  初期化
    //----------------------------------------------------------------------------------
    RandomValueGenerator_initialize((handle_t)This,seed);
	return This;
}

//-------------------------------------------------------------------------
// 初期化
//-------------------------------------------------------------------------
bool_t
RandomValueGenerator_initialize(handle_t hGenerator,uint32_t seed) {
    RandomValueGenerator* This = (RandomValueGenerator*)hGenerator;
    if (This == NULL) {
        return FALSE;
    }
    if (seed == 0) {
        seed = 10;
    }
    This->a = seed;
    return TRUE;
}

//-------------------------------------------------------------------------
// 乱数取得：整数
//-------------------------------------------------------------------------
uint32_t
RandomValueGenerator_getIntegerValue(handle_t hGenerator) {
    RandomValueGenerator* This = (RandomValueGenerator*)hGenerator;
    if (This == NULL) {
        return 0;
    }
    return xorshift32(This);
}

//-------------------------------------------------------------------------
// 乱数取得：小数
//-------------------------------------------------------------------------
flt32_t
RandomValueGenerator_getFloatingPointValue(handle_t hGenerator,flt32_t factor) {
    RandomValueGenerator* This = (RandomValueGenerator*)hGenerator;
    uint32_t   iValue;
    flt32_t    fValue;
    if (This == NULL) {
        return 0;
    }
    iValue = xorshift32(This);
    fValue = (flt32_t)(iValue & 0x7FFF) * (1.0f / 32768.0f); // [0.0, 1.0)
    fValue *= factor;
    return fValue;
}
