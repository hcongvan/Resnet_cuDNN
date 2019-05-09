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
	void* params;
	bool* forward;
	bool* backward;
	bool* option;
}template_t;
typedef enum {
	CONVOLUTION_SAME,
	CONVOLUTION_VAILD
}convType;
typedef struct {
	uint8_t inPos;
	uint8_t outPos;
	uint8_t curPos;
	enum layer_type _layer;
}layer_t;
typedef struct {
	cudnnConvolutionFwdAlgo_t algo;
	cudnnConvolutionDescriptor_t handle;
	cudnnFilterDescriptor_t kernel;
	cudnnTensorDescriptor_t in;
	cudnnTensorDescriptor_t out;
	size_t sWorkspace;
	int convNd;
	int cIn, hIn, wIn, nIn;
	int cOut, hOut, wOut, nOut;
	int stride, kernelSize;
	convType type;
}conv_t;
typedef struct {
	cudnnPoolingDescriptor_t handle;
	cudnnTensorDescriptor_t in;
	cudnnTensorDescriptor_t out;
	int cOut, hOut, wOut, nOut;
	int cIn, hIn, wIn, nIn;
	int poolNd;
	int stride;
}pooling_t;
typedef struct {
	cudnnDropoutDescriptor_t handle;
	cudnnTensorDescriptor_t in;
	cudnnTensorDescriptor_t out;
	int cOut, hOut, wOut, nOut;
	int cIn, hIn, wIn, nIn;
	float drop;
	int seed;
}dropout_t;
typedef struct {
	cudnnActivationDescriptor_t handle;
	cudnnTensorDescriptor_t in;
	cudnnActivationMode_t mode;
	cudnnTensorDescriptor_t out;
	int cOut, hOut, wOut, nOut;
	int cIn, hIn, wIn, nIn;
	double cofficient;
}act_t;
typedef struct Matix {
	int n, c, h, w;
	void * data;
}matrix_t;

std::vector<conv_t> _conv;
std::vector<cudnnFilterDescriptor_t> _filter;
std::vector<cudnnTensorDescriptor_t> _mat;
std::vector<act_t> _activation;
std::vector<pooling_t> _pooling;
std::vector<cudnnOpTensorDescriptor_t> _fc;
std::vector<dropout_t> _dropout;
std::vector<layer_type> _layer;
std::vector<matrix_t> m_mat;
std::vector<matrix_t> m_weight;
std::vector<matrix_t> m_bias;
cudnnTensorFormat_t format; //init
cudnnDataType_t dataType; //init

typedef struct {
	union {
		conv_t conv;
		pooling_t pool;
	};

};
bool ForwardConvNd(matrix_t in,cudnnHandle_t handle, int pos)
{
	float alph = 1.0, beta = 0;
	CUDNN_API_CALL(cudnnConvolutionForward(handle, &alph, _conv[pos].in, in.data, _conv[pos].kernel,
		/*weight matrix*/, _conv[pos].handle, _conv[pos].algo, /*workspace matrix*/, _conv[pos].sWorkspace, &beta, _conv[pos].out, /*out matrix*/));
	return true;
}
bool InitConvNd(cudnnHandle_t handle, convType type,const int convNd,int stride ,int kernelSize, int channel_out,int pos)
{
	int *_pad, *_stride, *_dilation;
	_pad = _stride = _dilation = (int *)malloc(convNd * sizeof(int));
	int _out[4];
	CUDNN_API_CALL( cudnnCreateConvolutionDescriptor(&_conv[pos].handle));
	CUDNN_API_CALL( cudnnCreateFilterDescriptor(&_conv[pos].kernel));
	CUDNN_API_CALL( cudnnCreateTensorDescriptor(&_conv[pos].in));
	CUDNN_API_CALL( cudnnCreateTensorDescriptor(&_conv[pos].out));
	
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
	CUDNN_API_CALL( cudnnSetConvolutionNdDescriptor(_conv[pos].handle, convNd, _pad, _stride,
		_dilation, CUDNN_CONVOLUTION, dataType));
	int sizefilter[] = { channel_out,_conv[pos].cIn,kernelSize,kernelSize };
	CUDNN_API_CALL( cudnnSetFilterNdDescriptor(_conv[pos].kernel, dataType, format, 4, sizefilter));

	//CUDNN_API_CALL( cudnnSetTensor4dDescriptor(_mat[pos], format, dataType, in.n, in.c, in.h, in.w));
	CUDNN_API_CALL( cudnnGetConvolutionNdForwardOutputDim(_conv[pos].handle, _conv[pos].in, _conv[pos].kernel, 4,_out));
	if (_out[1] != channel_out)
	{
		printf("size of output and filter not match");
		return false;
	}
	CUDNN_API_CALL( cudnnSetTensor4dDescriptor(_conv[pos].out, format, dataType, _out[0], _out[1], _out[2], _out[3]));
	CUDNN_API_CALL(cudnnGetConvolutionForwardAlgorithm(handle, _conv[pos].in, _conv[pos].kernel,
		_conv[pos].handle, _conv[pos].out, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &_conv[pos].algo));
	CUDNN_API_CALL(cudnnGetConvolutionForwardWorkspaceSize(handle, _conv[pos].in, _conv[pos].kernel,
		_conv[pos].handle, _conv[pos].out, _conv[pos].algo, &_conv[pos].sWorkspace));
	_conv[pos].type = type;
	_conv[pos].convNd = convNd;
	_conv[pos].stride = stride;
	_conv[pos].kernelSize = kernelSize;
	_conv[pos].cOut = channel_out;
	return true;
}
bool InitPoolingNd(matrix_t in,int poolNd ,int stride, int pos)
{
	
	int *out;
	out = (int *)malloc((poolNd + 2) * sizeof(int));
	CUDNN_API_CALL(cudnnCreatePoolingDescriptor(&_pooling[pos].handle));
	CUDNN_API_CALL(cudnnCreateTensorDescriptor(&_pooling[pos].in));
	int window[2] = { 2,2 }, _stride[2] = { stride,stride }, pad[2] = { 0,0 }; //fix it
	CUDNN_API_CALL(cudnnSetPoolingNdDescriptor(_pooling[pos].handle, CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN, poolNd, window, pad, _stride));
	CUDNN_API_CALL(cudnnGetPoolingNdForwardOutputDim(_pooling[pos].handle, _pooling[pos].in, poolNd+2, out));
	CUDNN_API_CALL(cudnnSetTensor4dDescriptor(_pooling[pos].out, format, dataType, out[0], out[1], out[2], out[3]));
	_pooling[pos].stride = stride;
	_pooling[pos].poolNd = poolNd;
	return true;
}

bool InitActFunc(matrix_t in,cudnnActivationMode_t mode, double cofficient,int pos)
{
	CUDNN_API_CALL(cudnnCreateActivationDescriptor(&_activation[pos].handle));
	CUDNN_API_CALL(cudnnSetActivationDescriptor(_activation[pos].handle, mode, CUDNN_NOT_PROPAGATE_NAN, cofficient));
	_activation[pos].mode = mode;
	_activation[pos].cofficient = cofficient;
	return true;
}
bool InitDropout(matrix_t in, cudnnHandle_t handle, float drop, matrix_t out,int seed, int pos)
{
	int size = out.c * out.h * out.w * out.n;
	CUDNN_API_CALL(cudnnCreateDropoutDescriptor(&_dropout[pos].handle));
	CUDNN_API_CALL(cudnnSetDropoutDescriptor(_dropout[pos].handle, handle, drop, out.data, size, seed));
	_dropout[pos].drop = drop;
	_dropout[pos].seed = seed;
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
	template_t a;
	conv_t convd;
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

