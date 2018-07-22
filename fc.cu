#include "common.h"


__global__ void fc_gpu_kernel(float *y, float *x, float *weights, const int weightHeight,const int outSize, const int inSize){
        //printf(x);
        int row = blockIdx.x*blockDim.x+threadIdx.x;
        int col = blockIdx.y*blockDim.y+threadIdx.y;
	//printf("row %d, col %d in fc.cu \n",row,col);
        if(row < inSize && col < outSize){
                float acc = 0;
                for(int i = 0; i < weightHeight; ++i){
  	  	  y[row*outSize+col] +=x[row*weightHeight + i ]*weights[i*outSize+col];
//                  printf("x[%d] is %.1f,weight[%d] is %.1f\n", row*weightHeight+i,x[row*weightHeight+i],i*outSize+col,weights[i*outSize+col]);
                }
//                printf("acc is %3f, y %d is %3f\n",acc, row*outSize+col, y[row*outSize+col] );
        }
}

__global__ void convdw_gpu_kernel(float *dw, float *dy, float *x, const int S,const int outSize, const int inSize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(row < inSize && col < outSize){
//		printf("row %d, col %d, bias[col] %.2f\n", row, col,bias[col]);	
  		for(int i = 0; i < S; ++i){
  	  	  dw[row*outSize+col] +=x[row+S*i ]*dy[i*outSize+col];
//		  printf("x[%d] is %.1f,dy[%d] is %.1f\n", row + S*i,x[row + S*i],i*S+row,dy[i*outSize+col]);
  		}
//  		printf("conv dw %d is %3f\n",row*outSize+col, dw[row*outSize+col] );
	}
}
__global__ void convdx_gpu_kernel(float *dx, float *dy, float *weights, const int S,const int outSize, const int inSize){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(row < inSize && col < outSize){
//		printf("row %d, col %d, bias[col] %.2f\n", row, col,bias[col]);	
  		for(int i = 0; i < S; ++i){
  	  	  dx[row*outSize+col] +=dy[row* S + i ]*weights[col*S+i];
//		  printf("dy[%d] is %.1f,weight[%d] is %.1f\n", row*S+i,dy[row*S+i],col*S+i,weights[col*S+i]);
  		}
  //		printf("conv dx %d is %3f\n",row*outSize+col, dx[row*outSize+col] );
	}
}
	
template <typename Dtype>
__global__ void BiasForward(Dtype* y, Dtype* bias,int weight_heights, const int outSize, const int inSize) { 
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if(row<inSize && col < outSize){
  		for(int i = 0; i < weight_heights; ++i){
			y[i*outSize+col] += bias[col];
	//	printf("y %d is %3f\n",row*outSize+col, y[row*outSize+col] );
  }
}                                   
}
template <typename Dtype>
__global__ void BiasBackward(const int n, int inSize, const Dtype* in, Dtype* out) { 
  CUDA_KERNEL_LOOP(index, n) { 
    out[index*inSize] += in[index]; 
  }                                   
}


void fc_gpu(float* data_out, float* data_in, float *weights, float *bias,const int weight_heights, const int outSize, const int inSize)
{
        // We are going to launch channels * height_col * width_col kernels, each
        // kernel responsible for copying a single-channel grid.
        //int num_kernels = channels * height_col * width_col;
        // NOLINT_NEXT_LINE(whitespace/operators)
        //int num_kernels = inSize * outSize;
        //printf("inheight is  %d innum is %d\n",inheight, innum);

        dim3 DimGrid((inSize-1)/32+1, (outSize-1)/32+1, 1);
        dim3 DimBlock(32,32,1);
//	printf("fc_gpu function in fc.cu!\n");
	fc_gpu_kernel<<<DimGrid, DimBlock>>>(
		data_out,data_in,weights,weight_heights,outSize,inSize);
	BiasForward<<<DimGrid, DimBlock>>>(data_out,bias,weight_heights,outSize,inSize);

}


cudaError_t convBPWithCuda(float* dx,float* dw,	float* db,float* dy,float* x,float* w,	float* b,int weight_heights,int outSize,
	int inSize, const int batch_size)
{
	float *dev_dy;
	float *dev_x;
	float *dev_w;
	float *dev_b;
	float *dev_dx;
	float *dev_dw;
	float *dev_db;
	checkCudaErrors(cudaMalloc((void**)&dev_dw, weight_heights*inSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_dx, inSize*weight_heights*batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_db, outSize*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&dev_dy, inSize * outSize*batch_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_dy, dy, inSize * outSize*batch_size*sizeof(float), cudaMemcpyHostToDevice));
	// image
	checkCudaErrors(cudaMalloc((void**)&dev_x, inSize* weight_heights * batch_size* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_x, x, inSize*weight_heights* batch_size*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&dev_w, weight_heights*outSize* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_w, w,  weight_heights*outSize*sizeof(float), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&dev_b, outSize* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_b, b, outSize*sizeof(float), cudaMemcpyHostToDevice));

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	float* tmp_in = dev_x;
	float* tmp_dx = dev_dx;
	float* tmp_dy = dev_dy;

	for(int i = 0; i<batch_size; i++){
		dim3 DimGrid((weight_heights-1)/32+1, (outSize-1)/32+1, 1);
		dim3 DimBlock(32,32,1);
		convdw_gpu_kernel<<<DimGrid, DimBlock>>>(dev_dw,tmp_in,tmp_dy,inSize,outSize, weight_heights);
		dim3 dimGrid((inSize-1)/32+1, (weight_heights-1)/32+1, 1);
		dim3 dimBlock(32,32,1);
		convdx_gpu_kernel<<<dimGrid, dimBlock>>>(tmp_dx,dev_w,tmp_dy,outSize, weight_heights, inSize);
		tmp_in +=inSize;
		tmp_dx +=inSize;
		BiasBackward<float><<<CAFFE_GET_BLOCKS(outSize*inSize), CAFFE_CUDA_NUM_THREADS>>>(outSize, inSize,tmp_dy,  dev_db);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "weights backward fcWithCuda failed!");
			return cudaStatus;
		}
		tmp_dy +=outSize;
	}
	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fc Kernel!\n", cudaStatus);
		cudaFree(dev_dy);
		cudaFree(dev_w);
		cudaFree(dev_b);
		cudaFree(dev_x);
		cudaFree(dev_dw);
		cudaFree(dev_db);
		cudaFree(dev_dx);

		return cudaStatus;
	}

	checkCudaErrors(cudaMemcpy(dw,dev_dw, weight_heights* outSize * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(db,dev_db, outSize*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(dx, dev_dx, inSize*weight_heights*batch_size*sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaFree(dev_dy);
    	cudaFree(dev_w);
    	cudaFree(dev_b);
    	cudaFree(dev_x);
	cudaFree(dev_dw);
    	cudaFree(dev_db);
    	cudaFree(dev_dx);

	return cudaStatus;
}

cudaError_t fcBPWithCuda(float* dx,float* dw,float* db,	float* dy,float* x,float* w,float* b,int outSize,int inSize, const int batch_size)
{
	float *dev_dy;
	float *dev_x;
	float *dev_w;
	float *dev_b;
	float *dev_dx;
	float *dev_dw;
	float *dev_db;
	checkCudaErrors(cudaMalloc((void**)&dev_dw, outSize*inSize * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_dx, inSize*batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dev_db, outSize*sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&dev_dy, outSize*batch_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_dy, dy, outSize*batch_size*sizeof(float), cudaMemcpyHostToDevice));
	// image
	checkCudaErrors(cudaMalloc((void**)&dev_x, inSize*batch_size* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_x, x, inSize* batch_size*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&dev_w, inSize*outSize* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_w, w, inSize* outSize*sizeof(float), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&dev_b, outSize* sizeof(float)));
	checkCudaErrors(cudaMemcpy(dev_b, b, outSize*sizeof(float), cudaMemcpyHostToDevice));

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	float* tmp_in = dev_x;
	float* tmp_dx = dev_dx;
	float* tmp_dy = dev_dy;

	for(int i = 0; i<batch_size; i++){
		dim3 DimGrid((inSize-1)/8+1, (outSize-1)/8+1, 1);
		dim3 DimBlock(8,8,1);
		fc_gpu_kernel<<<DimGrid, DimBlock>>>(dev_dw,tmp_in,tmp_dy,1,outSize, inSize);
		dim3 dimGrid(1,(inSize-1)/8+1, 1);
		dim3 dimBlock(8,8,1);
		fc_gpu_kernel<<<dimGrid, dimBlock>>>(tmp_dx,dev_w,tmp_dy,outSize,1,inSize);
		tmp_in +=inSize;
		tmp_dx +=inSize;
		BiasBackward<float><<<CAFFE_GET_BLOCKS(outSize), CAFFE_CUDA_NUM_THREADS>>>(outSize, 1,tmp_dy,  dev_db);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "weights backward fcWithCuda failed!");
			return cudaStatus;
		}
		tmp_dy +=outSize;
	}
	// Check for any errors launching the kernel
	checkCudaErrors(cudaGetLastError());
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fc Kernel!\n", cudaStatus);
		cudaFree(dev_dy);
		cudaFree(dev_w);
		cudaFree(dev_b);
		cudaFree(dev_x);
		cudaFree(dev_dw);
		cudaFree(dev_db);
		cudaFree(dev_dx);

		return cudaStatus;
	}

	checkCudaErrors(cudaMemcpy(dw,dev_dw, inSize* outSize * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(db,dev_db, outSize*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(dx, dev_dx, inSize*batch_size*sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaFree(dev_dy);
    	cudaFree(dev_w);
    	cudaFree(dev_b);
    	cudaFree(dev_x);
	cudaFree(dev_dw);
    	cudaFree(dev_db);
    	cudaFree(dev_dx);

	return cudaStatus;
}

cudaError_t fcWithCuda(
        float* data_out,
        float* data_in,
        float* weights,
        float* bias,
        const int weightHeight,
	const int outSize,
        const int inSize,
        const int batch_size)
{
        float *dev_out = 0;
        float *dev_in = 0;
        float *dev_w = 0;
        float *dev_b = 0;
        cudaError_t cudaStatus;
	
	//printf("fcWithCuda\n");
        // col 
        checkCudaErrors(cudaMalloc((void**)&dev_out, outSize*weightHeight *batch_size * sizeof(float)));

	//printf("fcWithCuda devout malloc success!\n");
        // image
        checkCudaErrors(cudaMalloc((void**)&dev_in, inSize*weightHeight* sizeof(float)));
	//printf("fcWithCuda devin malloc %d success!\n",inSize*weightHeight);
        checkCudaErrors(cudaMemcpy(dev_in, data_in, inSize*weightHeight * sizeof(float), cudaMemcpyHostToDevice));
	//printf("fcWithCuda devin memcpy success!\n");

        // kernel
        checkCudaErrors(cudaMalloc((void**)&dev_w, inSize*outSize * sizeof(float)));
	//printf("fcWithCuda devw malloc success!\n");
        checkCudaErrors(cudaMemcpy(dev_w, weights, inSize*outSize * sizeof(float), cudaMemcpyHostToDevice));

	//printf("fcWithCuda devw memcpy success!\n");
        // result
        checkCudaErrors(cudaMalloc((void**)&dev_b, outSize * sizeof(float)));
	//printf("fcWithCuda devb malloc success!\n");
        checkCudaErrors(cudaMemcpy(dev_b, bias, outSize * sizeof(float), cudaMemcpyHostToDevice));
	//printf("fcWithCuda devb mempcy success!\n");

        float* t_dev_in = dev_in;
        float* t_dev_out = dev_out;
        float* t_dev_w = dev_w;
        float* t_dev_b = dev_b;
        for(int i = 0; i < batch_size; i++)
        {
                // Launch a kernel on the GPU with one thread for each element.
                fc_gpu(t_dev_out, t_dev_in, t_dev_w,t_dev_b,weightHeight,outSize,inSize);


        //Perform warmup operation with cublas
                t_dev_in += inSize*weightHeight;
                t_dev_out += outSize*weightHeight;
        }


        // Check for any errors launching the kernel
        //checkCudaErrors(cudaGetLastError());
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fc Kernel!\n", cudaStatus);
		cudaFree(dev_in);
		cudaFree(dev_out);
		cudaFree(dev_w);
		cudaFree(dev_b);
		return cudaStatus;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fc Kernel!\n", cudaStatus);
		cudaFree(dev_in);
		cudaFree(dev_out);
		cudaFree(dev_w);
		cudaFree(dev_b);
		return cudaStatus;
        }


        // Copy output vector from GPU buffer to host memory.
        checkCudaErrors(cudaMemcpy(data_out, dev_out, outSize*weightHeight *batch_size* sizeof(float), cudaMemcpyDeviceToHost));
  	checkCudaErrors(cudaFree(dev_in));
	checkCudaErrors(cudaFree(dev_out));
	checkCudaErrors(cudaFree(dev_w));
	checkCudaErrors(cudaFree(dev_b));
	
         return cudaStatus;
}
 

