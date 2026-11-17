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
void	sigmoid_backward(flt32_t* pY, flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
//  WX+B
//=====================================================================================
void	weight_matrix_with_bias_forward(flt32_t* pInputBuffer, uint32_t inputDim, flt32_t* pWeightMatrix, flt32_t* pBias, flt32_t* pOutputBuffer, uint32_t unit, bool_t fSuperpose);
void	weight_matrix_with_bias_backward(flt32_t* pInputBuffer, uint32_t inputDim, flt32_t* pWeightMatrix, flt32_t* pOutputBuffer, uint32_t outputDim, flt32_t* pInputX, flt32_t* pDeltaWeightMatrix, flt32_t* pDeltaBias);

//=====================================================================================
//  Matrix Multiplication　C = A * B (A: M x K, B: K x N -> C: M x N)
//=====================================================================================
void	matmul_forward(const flt32_t* pA, const flt32_t* pB, flt32_t* pC, uint32_t M, uint32_t K, uint32_t N);
void	matmul_backward(const flt32_t* pA, const flt32_t* pB, const flt32_t* pLossC, flt32_t* pDeltaA, flt32_t* pDeltaB, uint32_t M, uint32_t K, uint32_t N);

//=====================================================================================
//  Matrix Multiplication Transposed C = A * B^T (A: M x K, B: N x K -> C: M x N)
//=====================================================================================
void	matmul_transposed_forward(const flt32_t* pA, const flt32_t* pB, flt32_t* pC, uint32_t M, uint32_t K, uint32_t N);
void	matmul_transposed_backward(const flt32_t* pA, const flt32_t* pB, const flt32_t* pLossC, flt32_t* pDeltaA, flt32_t* pDeltaB, uint32_t M, uint32_t K, uint32_t N);

//=====================================================================================
//  global average pooling 2D
//=====================================================================================
void	global_average_pooling2D_forward(const flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);
void	global_average_pooling2D_backward(flt32_t* pInputBuffer, const flt32_t* pOutputBuffer, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);

//=====================================================================================
//  relu
//=====================================================================================
void	relu_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim, flt32_t alpha);
void	relu_backward(flt32_t* pX, flt32_t* pLoss, flt32_t* pInput, uint32_t dim, flt32_t alpha);

//=====================================================================================
//  tanh
//=====================================================================================
void	tanh_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim);
void	tanh_backward(flt32_t* pY,flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
//  softmax
//=====================================================================================
void	softmax_forward(flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t dim);
void	softmax_backward(flt32_t* pY, flt32_t* pLoss, flt32_t* pInput, uint32_t dim);

//=====================================================================================
// sqrt
//=====================================================================================
flt32_t	low_cost_sqrt(const flt32_t x, uint32_t iterlations);

//=====================================================================================
// 初期値設定
//=====================================================================================
void	set_random_initial_values(handle_t hRandomValueGenerator, flt32_t* pParameterArray, uint32_t arraySize, flt32_t factor);

//=====================================================================================
// 初期値設定:normalization by sqrt
//=====================================================================================
void	set_random_initial_values_by_sqrt(handle_t hRandomValueGenerator, flt32_t* pParameterArray, uint32_t arraySize, uint32_t normSize);

//=====================================================================================
//初期値設定:Heの初期化（Kaiming Initialization）
//ReLU関数やLeakyReLU関数を利用する場合のデファクトスタンダード σ=√(2/parameter size)
//=====================================================================================
void	set_random_initial_values_by_he(handle_t hRandomValueGenerator, flt32_t* pParameterArray, uint32_t arraySize, uint32_t normSize);

//=====================================================================================
// 初期値設定:constant value
//=====================================================================================
void	set_constant_initial_values(flt32_t* pParameterArray, uint32_t arraySize, flt32_t value);

//=====================================================================================
// 浮動小数点配列値のコピー
//=====================================================================================
void	mem_copy_flt(flt32_t* pDest, const flt32_t* pSource, uint32_t arraySize);

//=====================================================================================
// 同一浮動小数点数値を配列にセットする
//=====================================================================================
void	mem_set_flt(flt32_t* pX, flt32_t value, uint32_t arraySize);

//=====================================================================================
// 同一浮動小数点数値を配列に乗算する
//=====================================================================================
void	mem_multiply_flt(flt32_t* pX, flt32_t value, uint32_t arraySize);

//=====================================================================================
//  C x H x W → H x W x Cに変換
//=====================================================================================
void	channel_first_2_last(const flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);

//=====================================================================================
//　H x W x C  → C x H x Wに変換
//=====================================================================================
void	channel_last_2_first(const flt32_t* pInputBuffer, flt32_t* pOutputBuffer, uint32_t inHeight, uint32_t inWidth, uint32_t inChannel);

#ifdef __cplusplus
}
#endif

#endif
