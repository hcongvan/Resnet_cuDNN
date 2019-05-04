#pragma once
#include <iostream>
#include <sstream>
#include "stdafx.h"

class CUDAException : public std::exception
{
public:
	CUDAException(const std::string& errorStr, const CUresult errorCode)
		: m_errorString(errorStr), m_errorCode(errorCode) {}
	CUDAException(const std::string& errorStr, const cudnnStatus_t errorCode)
		: m_errorString(errorStr), errorDNNCode(errorCode) {}
	CUDAException(const std::string& errorStr, const cudaError_t errorCode)
		: m_errorString(errorStr), cudaErCode(errorCode) {}

	virtual ~CUDAException() throw() {}
	virtual const char* what() const throw() { return m_errorString.c_str(); }
	CUresult  getErrorCUVIDCode() const { return m_errorCode; }
	cudnnStatus_t  getErrorDNNCode() const { return errorDNNCode; }
	cudaError_t  getCudaErrorCode() const { return cudaErCode; }
	const std::string& getErrorString() const { return m_errorString; }
	static CUDAException getCUVIDException(const std::string& errorStr, const CUresult errorCode,
		const std::string& functionName, const std::string& fileName, int lineNo);
	static CUDAException getDNNException(const std::string& errorStr, const cudnnStatus_t errorCode,
		const std::string& functionName, const std::string& fileName, int lineNo);
	static CUDAException getCUDAException(const std::string& errorStr, const cudaError_t errorCode,
		const std::string& functionName, const std::string& fileName, int lineNo);
private:
	std::string m_errorString;
	CUresult m_errorCode;
	cudaError_t cudaErCode;
	cudnnStatus_t errorDNNCode;
};

inline CUDAException CUDAException::getCUVIDException(const std::string& errorStr, const CUresult errorCode, const std::string& functionName,
	const std::string& fileName, int lineNo)
{
	std::ostringstream errorLog;
	errorLog << functionName << " : " << errorStr << " at " << fileName << ":" << lineNo << std::endl;
	CUDAException exception(errorLog.str(), errorCode);
	return exception;
}

inline CUDAException CUDAException::getDNNException(const std::string& errorStr, const cudnnStatus_t errorCode, const std::string& functionName,
	const std::string& fileName, int lineNo)
{
	std::ostringstream errorLog;
	errorLog << functionName << " : " << errorStr << " at " << fileName << ":" << lineNo << std::endl;
	CUDAException exception(errorLog.str(), errorCode);
	return exception;
}
inline CUDAException CUDAException::getCUDAException(const std::string& errorStr, const cudaError_t errorCode, const std::string& functionName,
	const std::string& fileName, int lineNo)
{
	std::ostringstream errorLog;
	errorLog << functionName << " : " << errorStr << " at " << fileName << ":" << lineNo << std::endl;
	CUDAException exception(errorLog.str(), errorCode);
	return exception;
}
#define NVDEC_API_CALL( cuvidAPI )																						\
		{																												\
			CUresult errorCode = cuvidAPI;                                                                             \
			if( errorCode != CUDA_SUCCESS)                                                                             \
			{                                                                                                          \
				std::ostringstream errorLog;                                                                           \
				errorLog << #cuvidAPI << " returned error " << errorCode;                                              \
				throw CUDAException::getCUVIDException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__);	\
			}                                                                                                          \
		}

#define CUDNN_API_CALL( cudnnAPI )																						\
		{																												\
			cudnnStatus_t errorCode = cudnnAPI;                                                                          \
			if( errorCode != CUDA_SUCCESS)                                                                             \
			{                                                                                                          \
				std::ostringstream errorLog;                                                                           \
				errorLog << #cudnnAPI << " returned error " << errorCode;                                              \
				throw CUDAException::getDNNException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__);		\
			}                                                                                                          \
		}

#define CUDA_RESULT_HANDLE( cudaAPI )																						\
		{																												\
			cudaError_t errorCode = cudaAPI;                                                                          \
			if( errorCode != CUDA_SUCCESS)                                                                             \
			{                                                                                                          \
				std::ostringstream errorLog;                                                                           \
				errorLog << #cudaAPI << " returned error " << errorCode;                                              \
				throw CUDAException::getCUDAException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__);		\
			}                                                                                                          \
		}