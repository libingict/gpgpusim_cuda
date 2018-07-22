
#include <stdio.h>
#include <cstdlib>

#include "im2col.hpp"

void init_data(float * data, int size)
{
	for(int i = 0; i < size; i++)
	{
		data[i] = (float)(rand()%3 + 1)/(float)size; 
	
	}
}
void init_int(int * data, int size)
{
	for(int i = 0; i < size; i++)
	{
		data[i] = (int)(rand()%size + 1)/size; 
	
	}
}
void print_data(float * data, int size)
{
        for(int i = 0; i < size; i++)
        {
                printf("%f ",data[i]);
        }
        printf("\n");
}

void block_fc_forward(const int height, const int width, const int channels, const int outSize,const int batch_size){

        const int dataArraySize = height * width * channels;
        const int kernelSize = dataArraySize * outSize;
        float *data_in = new float[dataArraySize];
        float *weights = new float[kernelSize];
        float *bias = new float[outSize]();
        float *data_out = new float[outSize];
        init_data(data_in, dataArraySize);
        init_data(weights, kernelSize);
        fcWithCuda(data_out, data_in, weights, bias,dataArraySize,outSize,1,batch_size);
}


void block_conv_forward(const int height, const int width, const int channels, const int batch_size, const int ksize, const int pad, const int stride, const int num_kernels){

	const int arraySize = height * width * channels * batch_size; //each bacth have 128 image, each image have 256*256 size and 3 channels
	float *image = new float[arraySize];// = { 1, 2, 3, 4, 5 };

	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
//	printf("height_col %d, width_col %d", height_col, width_col);
	int colArraySize = height_col * width_col * channels *ksize*ksize* batch_size;
	float *col1 = new float[colArraySize]();// = { 10, 20, 30, 40, 50 };
	//float *col2 = new float[colArraySize]();// = { 0 };
	
	int kernelArraySize = num_kernels*ksize*ksize*channels;
	float *data_kernel = new float[kernelArraySize];

	int resultArraySize = num_kernels * height_col * width_col*batch_size;
	float *bias = new float[num_kernels]();
	float *yl1 = new float[resultArraySize]();
	//float *r2 = new float[resultArraySize]();

	srand(2014);
	init_data(image, arraySize);
	init_data(data_kernel, kernelArraySize);

	// Choose which GPU to run on, change this on a multi-GPU system.
        cudaError_t cudaStatus = cudaSetDevice(0);
        //float *r1 = new float[resultArraySize]();
        cudaStatus = im2colWithCuda(image, batch_size, channels, height, width, ksize, pad, stride, col1, num_kernels, data_kernel, yl1);
        if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "im2colWithCuda failed!");
                return ;
        }

        int inHeight = height_col * width_col;
        int resultWidth = num_kernels;
        int weightHeight = ksize*ksize*channels;
 
        if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "fcWithCuda failed!");
                return ;
        }
        cudaStatus=fcWithCuda(yl1, col1, data_kernel, bias, weightHeight, resultWidth, inHeight, batch_size);
	delete [] col1;
}
void block_pool_forward(const int height, const int  width, const int channels, const int batch_size, const int ksize, const int pad, const int stride){

	int indataArraySize = height * width * channels * batch_size;

	float *yl1 = new float[indataArraySize];// = { 1, 2, 3, 4, 5 };
	init_data(yl1, indataArraySize);

	int pooled_height = (height+2*pad-ksize)/stride+1;
	int pooled_width = (width+2*pad-ksize)/stride+1;
	int count = pooled_height*pooled_width*channels;
	const int result_size = pooled_height*pooled_width*channels*batch_size;
	float *yl2 = new float[result_size]();
	//printf("yl2 new success!\n");
	int *mask_y2 = new int[result_size];
	//printf("mask yl2 new success!\n");
        cudaError_t cudaStatus = cudaSetDevice(0);
	cudaStatus=maxpoolWithCuda(yl1, batch_size,channels,height,width,ksize,pad,count,stride,yl2,mask_y2);

}
void block_pool_backward(const int height, const int  width, const int channels, const int batch_size, const int ksize, const int pad, const int stride){

	int indataArraySize = height * width * channels * batch_size;

	float *dx = new float[indataArraySize];// = { 1, 2, 3, 4, 5 };

	int pooled_height = (height+2*pad-ksize)/stride+1;
	int pooled_width = (width+2*pad-ksize)/stride+1;
	int count = pooled_height*pooled_width*channels;
	const int result_size = pooled_height*pooled_width*channels*batch_size;
	float *dy = new float[result_size];
	int *mask = new int[result_size];
	init_int(mask, result_size);
	init_data(dy, result_size);
        cudaError_t cudaStatus = cudaSetDevice(0);
	cudaStatus=maxpoolBPWithCuda(dy, batch_size,channels,height,width,ksize,pad,count,stride,dx,mask);

}
void block_relu_forward(const int height, const int  width, const int channels, const int batch_size){

	cudaError_t cudaStatus;
	int indataArraySize = height * width * channels * batch_size;

	float *yl1 = new float[indataArraySize];// = { 1, 2, 3, 4, 5 };
	init_data(yl1, indataArraySize);

	float *yl2 = new float[indataArraySize]();
	cudaStatus=reluWithCuda(yl1,yl2,indataArraySize, batch_size);

}
void block_relu_backward(const int height, const int  width, const int channels, const int batch_size){

	cudaError_t cudaStatus;
	int indataArraySize = height * width * channels;

	float *dy = new float[indataArraySize];// = { 1, 2, 3, 4, 5 };
	float *data_in = new float[indataArraySize];// = { 1, 2, 3, 4, 5 };
	init_data(dy, indataArraySize);
	init_data(data_in, indataArraySize);

	float *dx = new float[indataArraySize]();
	cudaStatus=reluBPWithCuda(dy,data_in,dx,indataArraySize, batch_size);
}

void block_conv_backward(const int batch_size, const int channels, const int num_kernels,const int height, const int width, const int ksize, const int pad, const int stride){
	const int dataSize=height * width * channels;
        int height_col = (height + 2 * pad - ksize) / stride + 1;
        int width_col = (width + 2 * pad - ksize) / stride + 1;
        int colArraySize = height_col * width_col * channels *ksize*ksize ;
	const int resultArraySize = num_kernels * height_col * width_col;
	const int kernelArraySize = num_kernels * ksize * ksize * channels;
        const int inheight = height_col*width_col;
        int inwidth = ksize*ksize*channels;

	float *dy = new float[resultArraySize];
	float *weights = new float[kernelArraySize];
	float *data_in = new float[dataSize*batch_size];
	float *bias = new float[num_kernels]();
	float *col1 = new float[colArraySize*batch_size]();

	init_data(dy, resultArraySize*batch_size);
	init_data(weights,kernelArraySize);
	init_data(data_in, dataSize*batch_size);
/*
	printf("dy\n");
	print_data(dy,resultArraySize*batch_size);
	printf("weights\n");
	print_data(weights,kernelArraySize);
	printf("data_in\n");
	print_data(data_in,dataSize*batch_size);
*/
	float *dw = new float[kernelArraySize]();
	float *db = new float[num_kernels]();
	float *dx = new float[colArraySize*batch_size]();

	cudaError_t cudaStatus = cudaSetDevice(0);
	float *r1 = new float[resultArraySize]();
	cudaStatus = im2colWithCuda(data_in, batch_size, channels, height, width, ksize, pad, stride, col1, num_kernels, weights, r1);
//	printf("conv1, conl1 \n");
 //       print_data(col1,colArraySize*batch_size); 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "im2colWithCuda failed!");
		return ;
	}

        cudaStatus=convBPWithCuda(dx,dw,db,dy, col1, weights, bias, inwidth,num_kernels,inheight, batch_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convBPWithCuda failed!");
		return ;
	}
/*
	printf("dw\n");
	print_data(dw,kernelArraySize);
	printf("dx\n");
	print_data(dx,colArraySize*batch_size);
	printf("db\n");
	print_data(db,num_kernels);
*/
	return;
}
void block_fc_backward(const int batch_size, const int inSize, const int outSize){
	float *dy = new float[outSize*batch_size];
	float *weights = new float[inSize*outSize];
	float *data_in = new float[inSize*batch_size];
	float *bias = new float[outSize]();

	init_data(dy,outSize*batch_size);
	init_data(weights,outSize*inSize);
	init_data(data_in, inSize*batch_size);
/*
	printf("dy\n");
	print_data(dy,outSize*batch_size);
	printf("weights\n");
	print_data(weights,outSize*inSize);
	printf("data_in\n");
	print_data(data_in,inSize*batch_size);
*/
	float *dw = new float[outSize*inSize]();
	float *db = new float[outSize]();
	float *dx = new float[inSize*batch_size]();
//return dw: 
	cudaError_t cudaStatus;
	cudaStatus=fcBPWithCuda(dx, dw, db, dy, data_in, weights, bias, outSize, inSize, batch_size);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "weights backward fcWithCuda failed!");
	return ;
	}

/*	printf("dw\n");
	print_data(dw,outSize*inSize);
	printf("dx\n");
	print_data(dx,inSize*batch_size);
	printf("db\n");
	print_data(db,outSize);
*/	
}

int main()
{
	//cudaError_t cudaStatus;
	const int batch_size = 1 ;
        //conv1
//	block_conv_forward(28,28,1,batch_size,5,0,1,20);
	//pool1
/*	block_pool_forward(24,24,20,batch_size,2,0,2);
	//conv2
	block_conv_forward(12,12,20,batch_size,5,0,1,50);
	//pool2
	block_pool_forward(8,8,50,batch_size,2,0,2);
	//ip1
	block_fc_forward(4,4,50,500,batch_size);
	//relu
//	block_relu_forward(1,1,500,batch_size);
	//ip2
	//block_fc_forward(1,1,500,10,batch_size);
*/
/*	block_fc_backward(batch_size,500,10);
	block_relu_backward(1,1,500,batch_size);
	//block_fc_backward(batch_size,4*4*50,500);
	block_pool_backward(8,8,50,batch_size,2,0,2);
	block_conv_backward(batch_size,20,50,12,12,5,0,1);
	block_pool_backward(24,24,20,batch_size,2,0,2);
*/	block_conv_backward(batch_size,1,20,28,28,5,0,1);
	//ip1
	return 0;
}

