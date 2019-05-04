#ifndef __video_decode_h__
#define __video_decode_h__
#include "NvDecoder\nvcuvid.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <mutex>
#include <stdint.h>
#include <assert.h>

typedef struct {
	int b, t, r, l;
}Rect_t;

typedef struct {
	int w, h;
}Mat_t;

enum cvtFlags_e{
	NV12TORGB=0,
	NV12TORGBA,
	NV12TOIYUV
};

class NVDECException : public std::exception
{
public:
	NVDECException(const std::string& errorStr, const CUresult errorCode)
		: m_errorString(errorStr), m_errorCode(errorCode) {}

	virtual ~NVDECException() throw() {}
	virtual const char* what() const throw() { return m_errorString.c_str(); }
	CUresult  getErrorCode() const { return m_errorCode; }
	const std::string& getErrorString() const { return m_errorString; }
	static NVDECException makeNVDECException(const std::string& errorStr, const CUresult errorCode,
		const std::string& functionName, const std::string& fileName, int lineNo);
private:
	std::string m_errorString;
	CUresult m_errorCode;
};

inline NVDECException NVDECException::makeNVDECException(const std::string& errorStr, const CUresult errorCode, const std::string& functionName,
	const std::string& fileName, int lineNo)
{
	std::ostringstream errorLog;
	errorLog << functionName << " : " << errorStr << " at " << fileName << ":" << lineNo << std::endl;
	NVDECException exception(errorLog.str(), errorCode);
	return exception;
}

#define NVDEC_API_CALL( cuvidAPI )																						\
		{																												\
			CUresult errorCode = cuvidAPI;                                                                             \
			if( errorCode != CUDA_SUCCESS)                                                                             \
			{                                                                                                          \
				std::ostringstream errorLog;                                                                           \
				errorLog << #cuvidAPI << " returned error " << errorCode;                                              \
				throw NVDECException::makeNVDECException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__); \
			}                                                                                                          \
		}
class VideoDecode
{
private:
	
	CUdevice m_Device;
	CUvideodecoder h_Decoder;
	CUcontext m_cuContext = NULL;
	CUvideoctxlock m_ctxLock;
	cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs;
	std::mutex *m_pMutex = NULL;
	int m_Width, m_Height;
	int m_nBitDepth;
	int m_nDecodedFrame = 0, m_nDecodedFrameReturned = 0;
	bool m_bEndDecodeDone = false;
	std::vector<uint8_t *> m_vpFrame;
	std::vector<uint8_t *> m_vpFrameRet;
	std::vector<int64_t> m_vTimestamp;
	bool m_bDeviceFramePitched = false;
	size_t m_nDeviceFramePitch = 0;
	std::mutex m_mtxVPFrame;
	int m_nFrameAlloc = 0;
	// height of the mapped surface 
	int m_nSurfaceHeight = 0;
	bool m_bUseDeviceFrame;
	CUvideoparser h_Parser;
	bool bOutPlanar = false;
	CUstream m_cuvidStream = 0;
	Rect_t m_Crop;
	Mat_t m_FrameSize;
	//FFmpegDemuxer demux("e:\\Youtube\\sample_per_title\\presentation_1.mp4");
	//std::ofstream fpOut("e:\\Youtube\\sample_per_title\\presentation_2.yuv", std::ios::out | std::ios::binary);
	static int CUDAAPI HandlePictureProc(void *pDataUser, CUVIDPICPARAMS * pPictureParams) { return ((VideoDecode *)pDataUser)->HandleDecodePicture(pPictureParams); };
	static int CUDAAPI HandleVideoProc(void *pDataUser, CUVIDEOFORMAT * pVideo) { return ((VideoDecode *)pDataUser)->HandleVideoFormat(pVideo); };
	static int CUDAAPI HandleDisplayProc(void *pDataUser, CUVIDPARSERDISPINFO * pDisplayParams) { return ((VideoDecode *)pDataUser)->HandleDisplayInfo(pDisplayParams); };
	int HandleDisplayInfo(CUVIDPARSERDISPINFO * pDisplay);
	int HandleDecodePicture(CUVIDPICPARAMS * pPictureParams);
	int HandleVideoFormat(CUVIDEOFORMAT * pVideo);

public:
	VideoDecode(CUcontext &context,int height,int width,int bitdepth, cudaVideoCodec codec, std::mutex * mutex,
		bool useDeviceFrame, bool useDeviceFramePitch);
		
	~VideoDecode();
	bool Decode(uint8_t* pData, int size, uint8_t *** p3DFrameOut, int * pnFrameReturn, int64_t ** ppTimeStamp, int nTimeStamp, int nFlags, CUstream vStream);
	int GetHeight() { assert(m_Height); return m_Height; }
	int GetWidth() { assert(m_Width);  return m_Width; }
	int GetBitDepth() { return m_nBitDepth; }
	int GetSurfaceFormat() { assert(m_nBitDepth); return ((m_nBitDepth - 8) ? 1 : 2); }
	int GetFrameSize() { assert(m_Height); assert(m_Width); return m_Width*m_Height * 3 / (GetSurfaceFormat()); }
	int GetVideoByte() { assert(m_nBitDepth); return ((m_nBitDepth - 8) ? 2: 1); }
	bool ConvertTo(uint8_t **p2DSrc, void * p2DDst ,int height, int width, int bitdepth,cvtFlags_e nflag);
};

#endif