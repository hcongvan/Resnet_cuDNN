// CudnnTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <vector>
#include "CUException.h"
#include <opencv2/opencv.hpp>
#include <cublas.h>

enum layer_type {
	CONVOLUTION = 0,
	ACTIVATION,
	POOLING,
	BATCH_NORMALIZATION,
	DROPOUT,
	DENSE
};

typedef struct {
	uint8_t in_pos;
	uint8_t out_pos;
	uint8_t layer_pos;
	enum layer_type _layer;
}layer_t;
typedef struct {
	cudnnConvolutionFwdAlgo_t algo;
	size_t workspace;
	cudnnConvolutionDescriptor_t descriptor;
	cudnnFilterDescriptor_t kernel;
}conv_t;
typedef struct Matix {
	int n, c, h, w;
	void * data;
}matrix_t;
typedef enum {
	CONVOLUTION_SAME,
	CONVOLUTION_VAILD
}convType;
std::vector<conv_t> _conv;
std::vector<cudnnFilterDescriptor_t> _filter;
std::vector<cudnnTensorDescriptor_t> _mat;
std::vector<cudnnActivationDescriptor_t> _activation;
std::vector<cudnnPoolingDescriptor_t> _pooling;
std::vector<cudnnOpTensorDescriptor_t> _fc;
std::vector<cudnnDropoutDescriptor_t> _dropout;
std::vector<layer_type> _layer;
std::vector<matrix_t> m_mat;
std::vector<matrix_t> m_weight;
std::vector<matrix_t> m_bias;
cudnnTensorFormat_t format; //init
cudnnDataType_t dataType; //init


bool ForwardConvNd(matrix_t in,cudnnHandle_t handle, layer_t pos)
{
	


	float alph = 1.0, beta = 0;
	cudnnConvolutionForward(cudnnHandle, &alph, input, _input, f1, _f1, c1, algo, _workSpace, workspaceSize, &beta, output, _output);
	return true;
}
bool InitConvNd(matrix_t in,cudnnHandle_t handle, convType type,const int convNd,int stride ,int kernelSize, int channel_out,layer_t pos)
{
	int *_pad, *_stride, *_dilation;
	_pad = _stride = _dilation = (int *)malloc(convNd * sizeof(int));
	int _out[4];
	CUDNN_API_CALL( cudnnCreateConvolutionDescriptor(&_conv[pos.layer_pos].descriptor));
	CUDNN_API_CALL( cudnnCreateFilterDescriptor(&_filter[pos.layer_pos]));
	//CUDNN_API_CALL( cudnnCreateTensorDescriptor(&_mat[pos]));
	CUDNN_API_CALL( cudnnCreateTensorDescriptor(&_mat[pos.out_pos]));
	
	if (type == CONVOLUTION_SAME)
	{
		for (int i = 0; i < convNd; i++)
		{
			_pad[i] = (round(kernelSize / 2) - 1);
			_dilation[i] = 1;
			_stride[i] = stride;
		}
	}
	else 
	{
		for (int i = 0; i < convNd; i++)
		{
			_pad[i] = 0;
			_dilation[i] = 1;
			_stride[i] = stride;
		}
	}
	CUDNN_API_CALL( cudnnSetConvolutionNdDescriptor(_conv[pos.layer_pos], convNd, _pad, _stride, _dilation, CUDNN_CONVOLUTION, dataType));
	int sizefilter[] = { channel_out,in.c,kernelSize,kernelSize };
	CUDNN_API_CALL( cudnnSetFilterNdDescriptor(_filter[pos.layer_pos], dataType, format, 4, sizefilter));

	//CUDNN_API_CALL( cudnnSetTensor4dDescriptor(_mat[pos], format, dataType, in.n, in.c, in.h, in.w));
	CUDNN_API_CALL( cudnnGetConvolutionNdForwardOutputDim(_conv[pos.layer_pos], _mat[pos.in_pos], _filter[pos.layer_pos], 4,_out));
	if (_out[1] != channel_out)
	{
		printf("size of output and filter not match");
		return false;
	}
	CUDNN_API_CALL( cudnnSetTensor4dDescriptor(_mat[pos.out_pos], format, dataType, _out[0], _out[1], _out[2], _out[3]));
	CUDNN_API_CALL(cudnnGetConvolutionForwardAlgorithm(handle, _mat[pos.in_pos], _filter[pos.layer_pos],
		_conv[pos.layer_pos], _mat[pos.out_pos], CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, input, f1, c1, output, algo, &workspaceSize);
	return true;
}
bool InitPoolingNd(matrix_t in,int poolNd ,int stride,int poolsize, layer_t pos)
{
	int *out;
	out = (int *)malloc((poolNd + 2) * sizeof(int));
	cudnnCreatePoolingDescriptor(&_pooling[pos.layer_pos]);
	cudnnCreateTensorDescriptor(&_mat[pos.in_pos]);
	int window[2] = { 2,2 }, _stride[2] = { 2,2 }, pad[2] = { 0,0 };
	cudnnSetPoolingNdDescriptor(_pooling[pos.layer_pos], CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, poolNd, window, pad, _stride);
	cudnnGetPoolingNdForwardOutputDim(_pooling[pos.layer_pos], _mat[pos.in_pos], poolNd+2, out);
	cudnnSetTensor4dDescriptor(_mat[pos.out_pos], format, dataType, out[0], out[1], out[2], out[3]);
	return true;
}

bool InitActFunc(matrix_t in,cudnnActivationMode_t mode, double cofficient,layer_t pos)
{
	cudnnCreateActivationDescriptor(&_activation[pos.layer_pos]);
	cudnnSetActivationDescriptor(_activation[pos.layer_pos], mode, CUDNN_NOT_PROPAGATE_NAN, cofficient);
	return true;
}
bool InitDropout(matrix_t in, cudnnHandle_t handle, float dropout, matrix_t out,int seed, layer_t pos)
{
	int size = out.c * out.h * out.w * out.n;
	cudnnCreateDropoutDescriptor(&_dropout[pos.layer_pos]);
	cudnnSetDropoutDescriptor(_dropout[pos.layer_pos], handle, dropout, out.data, size, seed);
	return true;
}

void Forward(std::vector<layer_type> layer)
{
	
}
cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}
int main()
{

	cv::Mat image = load_image("E:\tensorflow.png");
	int mH = 128, mW = 128, mD = 3;
	float * _data,*_f1,*_input,*_workSpace,*_output,*_result;
	
	_data = (float *)malloc(mH*mW*mD * sizeof(float));
	//_filter = (float *)malloc(3 * 3 * 1 * 1 * sizeof(float));
	for (int i = 0; i < 1 ; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			for (int k = 0; k < 12; k++)
			{
				_data[i * 12 * 12 + j * 12 + k] = (rand() % 10);
				printf("  %.2f", _data[i * 12 * 12 + j * 12 + k]);
			}
			printf("\r\n");
		}
		printf("\r\n"); printf("\r\n");
	}
	float _filter[3][3] = {
		{ 1,  1, 1 },
		{ 1, -8, 1 },
		{ 1,  1, 1 }
	};
	//for (int i = 0; i < 1; i++)
	//{
	//	for (int j = 0; j < 1; j++)
	//	{
	//		for (int k = 0; k < 3; k++)
	//		{
	//			for (int h = 0; h < 3; h++)
	//			{
	//				_filter[i * 3 * 3 * 5 + j * 3 * 5 + k * 5 + h] = rand() % 4;
	//				printf("  %.2f", _filter[i * 3 * 3 * 5 + j * 3 * 5 + k * 5 + h]);
	//			}
	//			printf("\r\n");
	//		}
	//		printf("\r\n"); printf("\r\n");
	//	}
	//	printf("\r\n"); printf("\r\n");
	//}
	// Create model train;
	cublasHandle_t cublasHandle;
	cudaError_t error;
	cublasStatus_t err;
	cudnnHandle_t cudnnHandle;
	cudnnConvolutionDescriptor_t c1;
	cudnnFilterDescriptor_t f1;
	cudnnTensorDescriptor_t input,output;
	cudnnConvolutionFwdAlgo_t algo;
	size_t workspaceSize;
	int c, h, w, n;
	cudnnStatus_t status;
	error = cudaSetDevice(0);
	err = cublasCreate_v2(&cublasHandle);
	status = cudnnCreate(&cudnnHandle);
	//build Convolution Layer
	
	status = cudnnCreateConvolutionDescriptor(&c1);
	status = cudnnCreateFilterDescriptor(&f1);
	status = cudnnCreateTensorDescriptor(&input);
	status = cudnnCreateTensorDescriptor(&output);
	int pad[2] = { 1,1 }, stride[2] = { 1,1 }, dilation[2] = { 1,1 };
	status = cudnnSetConvolutionNdDescriptor(c1, 2,pad,stride,dilation, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT );
	int sizefilter[] = { 1,1,3,3 };
	status = cudnnSetFilterNdDescriptor(f1, CUDNN_DATA_FLOAT , CUDNN_TENSOR_NCHW, 4,sizefilter);
	
	int sizeITensor[] = { 1,1,12,12 }, strideITensor[] = { 1,1,1,1 };
	int sizeOTensor[4] , strideOTensor[] = { 1,1,1,1 };
	status = cudnnSetTensor4dDescriptor(input,CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT ,1,1,12,12);
	//status = cudnnSetTensordDescriptor(input, CUDNN_DATA_FLOAT, 4, sizeITensor,strideITensor);
	cudnnGetConvolutionNdForwardOutputDim(c1, input, f1, 4, sizeOTensor);
	
	status = cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT , sizeOTensor[0], sizeOTensor[1], sizeOTensor[2], sizeOTensor[3]);
	//status = cudnnSetTensorNdDescriptor(output, CUDNN_DATA_FLOAT, 4, sizeOTensor,strideOTensor);
	status = cudnnGetConvolutionForwardAlgorithm(cudnnHandle, input, f1, c1, output, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&algo);
	if (status == CUDNN_STATUS_BAD_PARAM)
	{
		algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
	}
	status = cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, input, f1, c1, output, algo, &workspaceSize);
	
	_result = (float *)malloc(sizeOTensor[0]* sizeOTensor[1]* sizeOTensor[2]* sizeOTensor[3] * sizeof(float));
	memset(_result, 0, sizeOTensor[0] * sizeOTensor[1] * sizeOTensor[2] * sizeOTensor[3] * sizeof(float));
	error = cudaMalloc((void**)&_workSpace,workspaceSize * sizeof(size_t));
	error = cudaMalloc((void**)&_output, 1*12*12*1 * sizeof(float));
	error = cudaMalloc((void**)&_input, 12 * 12 * 1 * sizeof(float));
	error = cudaMalloc((void**)&_f1, 1 * 3 * 3 * 1 * sizeof(float));
	error = cudaMemcpy(_input, _data, 12 * 12 * 1 * sizeof(float), cudaMemcpyHostToDevice);
	error = cudaMemcpy(_f1, _filter, 1 * 3 * 3 * 1 * sizeof(float), cudaMemcpyHostToDevice);

	float alph = 1.0, beta = 0;
	status = cudnnConvolutionForward(cudnnHandle, &alph, input, _input, f1, _f1, c1, algo, _workSpace, workspaceSize, &beta, output, _output);
	error = cudaMemcpy(_result, _output, sizeOTensor[0] * sizeOTensor[1] * sizeOTensor[2] * sizeOTensor[3] * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sizeOTensor[1]; i++)
	{
		for (int j = 0; j < sizeOTensor[2]; j++)
		{
			for (int k = 0; k < sizeOTensor[3]; k++)
			{
				//_data[i * 12 * 12 + j * 12 + k] = rand() % 10;
				printf(" %.2f", _result[i * 12 * 12 + j * 12 + k]);
			}
			printf("\r\n");
		}
		printf("\r\n"); printf("\r\n");
	}
    return 0;
}

