#ifndef NEURAL_NET_LAYER_FUNCTION_H
#define NEURAL_NET_LAYER_FUNCTION_H


#ifdef __cplusplus
extern "C" {
#endif

#include "STDTypeDefinition.h"

//=====================================================================================
//  sigmoid
//=====================================================================================
void	sigmoid_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim);
void	sigmoid_backword(flt32_t* pX, flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
//  WX+B
//=====================================================================================
void	weight_matrix_with_bias_forward(flt32_t* pInputBuffer, uint32_t inputDim, flt32_t* pWeightMatrix, flt32_t* pBias, flt32_t* pOutputBuffer, uint32_t unit, bool_t fInitialize);
void	weight_matrix_with_bias_backward(flt32_t* pInputBuffer, uint32_t inputDim, flt32_t* pWeightMatrix, flt32_t* pOutputBuffer, uint32_t outputDim, flt32_t* pInputX, flt32_t* pDeltaWeightMatrix, flt32_t* pDeltaBias);

//=====================================================================================
//  relu
//=====================================================================================
void	relu_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim, flt32_t alpha);
void	relu_backword(flt32_t* pX, flt32_t* pLoss, flt32_t* pInput, uint32_t dim, flt32_t alpha);

//=====================================================================================
//  tanh
//=====================================================================================
void	tanh_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim);
void	tanh_backword(flt32_t* pY,flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
//  softmax
//=====================================================================================
void	softmax_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim);
void	softmax_backword(flt32_t* pY, flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
// sqrt
//=====================================================================================
flt32_t	low_cost_sqrt(const flt32_t x, uint32_t iterlations);

//-------------------------------------------------------------------------
// èâä˙ílê›íË
//-------------------------------------------------------------------------
void	set_random_initial_values(handle_t hRandomValueGenerator, flt32_t* pParameterArray, uint32_t arraySize, flt32_t factor);

//-------------------------------------------------------------------------
// èâä˙ílê›íË:normalization by sqrt
//-------------------------------------------------------------------------
void	set_random_initial_values_by_sqrt(handle_t hRandomValueGenerator, flt32_t* pParameterArray, uint32_t arraySize, uint32_t normSize);

//-------------------------------------------------------------------------
// èâä˙ílê›íË:constant value
//-------------------------------------------------------------------------
void	set_constant_initial_values(flt32_t* pParameterArray, uint32_t arraySize, flt32_t value);


#ifdef __cplusplus
}
#endif

#endif
