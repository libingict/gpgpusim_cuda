#include<algorithm>
#include "common.h"
#include<float.h>
#include<cmath>


template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out)
{
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : 0;
  }
}
template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff)
{
  CUDA_KERNEL_LOOP(index, n) {
	out_diff[index] = in_diff[index] * (in_data[index] > 0);  
}
}
template <typename Dtype>
void relu_gpu(const Dtype* data_im, const int count, Dtype* data_out){
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, data_im , data_out);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
  }
}
cudaError_t reluWithCuda(const float* data_im, float* data_out,const int count, const int batch_size)
{

	float *dev_in = 0;
	float *dev_out = 0;

	cudaError_t cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&dev_out, count *batch_size * sizeof(float)));
	printf("dev out malloc success\n");
	checkCudaErrors(cudaMalloc((void**)&dev_in, count *batch_size * sizeof(float)));
	printf("dev in malloc success\n");
	checkCudaErrors(cudaMemcpy(dev_in, data_im, count*batch_size * sizeof(float), cudaMemcpyHostToDevice));
	printf("data im memcpy success\n");
	float* t_dev_image = dev_in;
	float* t_dev_ret = dev_out;
	for(int i = 0; i < batch_size; i++)
	{
		// Launch a kernel on the GPU with one thread for each element.
		relu_gpu<float>(t_dev_image, count, t_dev_ret);


        //Perform warmup operation with cublas
		t_dev_image += count;
		t_dev_ret += count;
	}

	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pool Kernel!\n", cudaStatus);
		cudaFree(dev_in);
		cudaFree(dev_out);
	}

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(data_out, dev_out, count*batch_size * sizeof(float), cudaMemcpyDeviceToHost));
		cudaFree(dev_in);
		cudaFree(dev_out);
	return cudaStatus;

}

cudaError_t reluBPWithCuda(const float* in_diff, const float * data_in, float* out_diff,const int count, const int batch_size)
{

	float *dev_data = 0;
	float *dev_in = 0;
	float *dev_out = 0;

	cudaError_t cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&dev_out, count *batch_size * sizeof(float)));
	printf("dev out malloc success\n");
	checkCudaErrors(cudaMalloc((void**)&dev_in, count *batch_size * sizeof(float)));
	printf("dev in malloc success\n");
	checkCudaErrors(cudaMemcpy(dev_in, in_diff, count*batch_size * sizeof(float), cudaMemcpyHostToDevice));
	printf("data im memcpy success\n");
	checkCudaErrors(cudaMalloc((void**)&dev_data, count *batch_size * sizeof(float)));
	printf("dev in malloc success\n");
	checkCudaErrors(cudaMemcpy(dev_data,data_in, count*batch_size * sizeof(float), cudaMemcpyHostToDevice));
	float* t_dev_image = dev_data;
	float* t_dev_indiff = dev_in;
	float* t_dev_ret = dev_out;
	for(int i = 0; i < batch_size; i++)
	{
		// Launch a kernel on the GPU with one thread for each element.
		ReLUBackward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,t_dev_indiff, t_dev_image, t_dev_ret);

        //Perform warmup operation with cublas
		t_dev_image += count;
		t_dev_indiff += count;
		t_dev_ret += count;
	}

	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pool Kernel!\n", cudaStatus);
		cudaFree(dev_in);
		cudaFree(dev_data);
		cudaFree(dev_out);
	}

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(out_diff, dev_out, count*batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(dev_in);
	cudaFree(dev_data);
	cudaFree(dev_out);
	return cudaStatus;

}

template <typename Dtype>
void sigmoid_gpu(const Dtype* data_im, const int count, Dtype* data_out){

  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, data_im , data_out);
  //CUDA_POST_KERNEL_CHECK;
}
cudaError_t sigmoidWithCuda(const float* data_im, float* data_out,const int count, const int batch_size)
{

	float *dev_in = 0;
	float *dev_out = 0;

	cudaError_t cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&dev_out, count *batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_in, count *batch_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_in, data_im, count*batch_size * sizeof(float), cudaMemcpyHostToDevice));
	float* t_dev_image = dev_in;
	float* t_dev_ret = dev_out;
	for(int i = 0; i < batch_size; i++)
	{
		// Launch a kernel on the GPU with one thread for each element.
		sigmoid_gpu<float>(t_dev_image, count, t_dev_ret);

        //Perform warmup operation with cublas
		t_dev_image += count;
		t_dev_ret += count;
	}

	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pool Kernel!\n", cudaStatus);
		cudaFree(dev_in);
		cudaFree(dev_out);
	}

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(data_out, dev_out, count *batch_size* sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(dev_in);
	cudaFree(dev_out);
	return cudaStatus;

}
