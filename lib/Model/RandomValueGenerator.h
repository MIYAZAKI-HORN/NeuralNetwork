#ifndef RANDOM_VALUE_GENERATOR_H
#define RANDOM_VALUE_GENERATOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"

//-------------------------------------------------------------------------
// サイズ取得
//-------------------------------------------------------------------------
uint32_t	RandomValueGenerator_getSizeIn32BitWord(void);

//-------------------------------------------------------------------------
// 構築
//-------------------------------------------------------------------------
handle_t	RandomValueGenerator_construct(uint32_t seed,uint32_t* pWorkArea, uint32_t sizeOfWorkAreaIn32BitWord);
bool_t		RandomValueGenerator_initialize(handle_t hGenerator, uint32_t seed);

//-------------------------------------------------------------------------
// 乱数取得
//-------------------------------------------------------------------------
uint32_t	RandomValueGenerator_getIntegerValue(handle_t hGenerator);
flt32_t		RandomValueGenerator_getFloatingPointValue(handle_t hGenerator, flt32_t factor);

#ifdef __cplusplus
}
#endif


#endif
