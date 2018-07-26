#ifndef __IM2COL_H__
#define __IM2COL_H__

#include "common.h"

int check_result(float* a, float* b, int size);

cudaError_t addWithCuda(float *c, float *a, float *b, unsigned int size);

cudaError_t im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col,
	const int num_kernels,
	float* data_kernel,
	float* data_ret);

cudaError_t bu_im2colWithCuda(
	const float* data_im,
	const int batch_size,
	const int channels,
	const int height, const int width, const int ksize, const int pad,
	const int stride,
	float* data_col,
	const int num_kernels,
	float* data_kernel,
	float* data_ret);
cudaError_t convWithCuda(
        float* data_out,
        float* data_in,
        float* weights,
        float* bias,
        const int weightHeight, 
        const int outSize,
        const int inSize,
        const int batch_size);

cudaError_t fcWithCuda(
        float* data_out,
        float* data_in,
        float* weights,
        float* bias,
        const int weightHeight, 
        const int outSize,
        const int inSize,
        const int batch_size);
cudaError_t convBPWithCuda(float* dx,float* dw,	float* db,float* dy,float* x,float* weights,float* bias,const int inheight,const int outSize,const int inSize, const int batch_size);
cudaError_t biasWithCuda(
	float* data_in, 
	float* data_out, 
	int count);
cudaError_t fcBPWithCuda(
	float* dx,
	float* dw,
	float* db,
	float* dy,
	float* x,
	float* w,
	float* b,
	int outSize,
	int inSize, const int batch_size);
cudaError_t maxpoolWithCuda(
        const float* data_im,
        const int batch_size,
        const int channels,
        const int height,
        const int width,
        const int ksize,
        const int pad,
        const int count,
        const int stride,
        float* data_out,
        int *mask);
cudaError_t maxpoolBPWithCuda(const float* dy, const int batch_size,const int channels,
				const int height, const int width, const int ksize, const int pad,const int count,
				const int stride, float* dx,int *mask);

cudaError_t sigmoidWithCuda(const float* data_im, float* data_out,const int count, const int batch_size);
cudaError_t reluWithCuda(const float* data_im, float* data_out,const int count, const int batch_size);
cudaError_t reluBPWithCuda(const float* in_diff, const float * data_in, float* out_diff,const int count, const int batch_size);
#endif // __IM2COL_H__
