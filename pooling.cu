#include "common.h"
#include<algorithm>
#include<float.h>
//using namespace std;
template <typename Dtype>
__global__ void MaxPoolForward(const int nt,
    const Dtype* data_in, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int ksize, const int stride, const int pad, Dtype* data_out,  int* mask) {
    
	//printf("maxpool_gpu_kernel success\n");
  CUDA_KERNEL_LOOP(index, nt) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
   // printf("index %d, pw %d, ph %d, c %d\n",index, pw, ph, c);
    int hstart = ph * stride - pad;
    int wstart = pw * stride - pad;
    const int hend = min(hstart + ksize, height);
    const int wend = min(wstart + ksize, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        data_in + (n * channels + c) * height * width;
    //printf("bottom_slice %.3f\n", bottom_slice[data_in+(n*channels+c)*height*width]);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
//	printf("h %d,w %d,index %d\n", h,w,h*width+w);
	//printf("h %d,w %d,index %d, bottom_slice %.1f\n", h,w,h*width+w,bottom_slice[h*width+w]);
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    data_out[index] = maxval;
    mask[index] = maxidx;
  }
}

template <typename Dtype>
void maxpool_gpu(const Dtype* data_im, const int channels, const int height, const int width, const int ksize, const int pad,const int count, const int stride, Dtype* data_out,int *mask)
{
	int pooled_height = (height+2*pad-ksize)/stride+1;
	int pooled_width = (width+2*pad-ksize)/stride+1;
	MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, data_im, channels, height, width, pooled_height, pooled_width, ksize,
        stride, pad, data_out, mask);
}
template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const dy,
    const int* const mask, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const dx) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = dy + offset;
    const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    dx[index] = gradient;
  }
}

cudaError_t maxpoolBPWithCuda(const float* dy, const int batch_size,const int channels,
				const int height, const int width, const int ksize, const int pad,const int count,
				const int stride, float* dx,int *mask)
{
	float *dev_dy = 0;
	float *dev_dx = 0;
	int *dev_mask = 0;
	cudaError_t cudaStatus;
	
	int image_size = height*width*channels;
	int images_size = image_size*batch_size;
	

	int pooled_height = (height+2*pad-ksize)/stride+1;
	int pooled_width = (width+2*pad-ksize)/stride+1;

	int result_size = pooled_height*pooled_width*channels;

	// col 
	checkCudaErrors(cudaMalloc((void**)&dev_dy, result_size *batch_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_dy, dy, result_size*batch_size * sizeof(float), cudaMemcpyHostToDevice));
	printf("dev_out malloc success!\n");
	// image
	checkCudaErrors(cudaMalloc((void**)&dev_dx, images_size * sizeof(float)));
	printf("dev_in malloc success!\n");
	
	// kernel
	checkCudaErrors(cudaMalloc((void**)&dev_mask, result_size*batch_size * sizeof(int)));
	checkCudaErrors(cudaMemcpy(dev_mask, mask, result_size*batch_size * sizeof(int),cudaMemcpyHostToDevice));
	printf("dev_mask malloc success!\n");
	// result

	float* t_dev_dy = dev_dy;
	int* t_dev_mask = dev_mask;
	//printf("t_dev_mask success\n");
	float* t_dev_dx = dev_dx;
	//printf("t_dev_ret success\n");
	for(int i = 0; i < batch_size; i++)
	{	
		// Launch a kernel on the GPU with one thread for each element.
		MaxPoolBackward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, t_dev_dy, mask, channels, height, width, pooled_height, pooled_width, ksize,ksize,
				stride, stride, pad, pad, t_dev_dx);

        //Perform warmup operation with cublas
		t_dev_dx += image_size;
		t_dev_mask += result_size;
		t_dev_dy += result_size;
	}

	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching pool Kernel!\n", cudaStatus);
		cudaFree(dev_dy);
		cudaFree(dev_dx);
		cudaFree(dev_mask);
	}

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(dx, dev_dx, images_size* sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(dev_dy);
	cudaFree(dev_dx);
	cudaFree(dev_mask);

	return cudaStatus;
	
}
cudaError_t maxpoolWithCuda(const float* data_im, const int batch_size,const int channels,
				const int height, const int width, const int ksize, const int pad,const int count,
				const int stride, float* data_out,int *mask)
{

	float *dev_in = 0;
	float *dev_out = 0;
	int *dev_mask = 0;
	cudaError_t cudaStatus;
	
	int image_size = height*width*channels;
	int images_size = image_size*batch_size;
	

	int pooled_height = (height+2*pad-ksize)/stride+1;
	int pooled_width = (width+2*pad-ksize)/stride+1;

	int result_size = pooled_height*pooled_width*channels;

	// col 
	checkCudaErrors(cudaMalloc((void**)&dev_out, result_size *batch_size * sizeof(float)));
	printf("dev_out malloc success!\n");
	// image
	checkCudaErrors(cudaMalloc((void**)&dev_in, images_size * sizeof(float)));
	printf("dev_in malloc success!\n");
	checkCudaErrors(cudaMemcpy(dev_in, data_im, images_size * sizeof(float), cudaMemcpyHostToDevice));
	printf("dev_in memcpy success!\n");
	
	// kernel
	checkCudaErrors(cudaMalloc((void**)&dev_mask, result_size*batch_size * sizeof(int)));
	printf("dev_mask malloc success!\n");
	// result

	float* t_dev_image = dev_in;
	int* t_dev_mask = dev_mask;
	//printf("t_dev_mask success\n");
	float* t_dev_ret = dev_out;
	//printf("t_dev_ret success\n");
	for(int i = 0; i < batch_size; i++)
	{	
		// Launch a kernel on the GPU with one thread for each element.
		maxpool_gpu<float>(t_dev_image, channels, height, width, ksize, pad, count, stride, t_dev_ret,t_dev_mask);


        //Perform warmup operation with cublas
		t_dev_image += image_size;
		t_dev_mask += result_size;
		t_dev_ret += result_size;
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
		cudaFree(dev_mask);
	}

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(mask, dev_mask, result_size*batch_size* sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(data_out, dev_out, result_size *batch_size* sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_mask);

	return cudaStatus;
}

