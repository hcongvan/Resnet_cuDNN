#include "stdafx.h"

#include "VideoDecode.h"

//void ConvertToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth) {
//	if (nBitDepth == 8) {
//		// nv12->iyuv
//		YuvConverter<uint8_t> converter8(nWidth, nHeight);
//		converter8.UVInterleavedToPlanar(pHostFrame);
//	}
//	else {
//		// p016->yuv420p16
//		YuvConverter<uint16_t> converter16(nWidth, nHeight);
//		converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
//	}
//}

static unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec, unsigned int nWidth, unsigned int nHeight) {
	if (eCodec == cudaVideoCodec_VP9) {
		return 12;
	}
	if (eCodec == cudaVideoCodec_H264 || eCodec == cudaVideoCodec_H264_SVC || eCodec == cudaVideoCodec_H264_MVC) {
		// assume worst-case of 20 decode surfaces for H264
		return 20;
	}
	if (eCodec == cudaVideoCodec_HEVC) {
		// ref HEVC spec: A.4.1 General tier and level limits
		// currently assuming level 6.2, 8Kx4K
		int MaxLumaPS = 35651584;
		int MaxDpbPicBuf = 6;
		int PicSizeInSamplesY = (int)(nWidth * nHeight);
		int MaxDpbSize;
		if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
			MaxDpbSize = MaxDpbPicBuf * 4;
		else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
			MaxDpbSize = MaxDpbPicBuf * 2;
		else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
			MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
		else
			MaxDpbSize = MaxDpbPicBuf;
		return std::min(MaxDpbSize, 16) + 4;
	}
	return 8;
}

int VideoDecode::HandleVideoFormat(CUVIDEOFORMAT * pVideo)
{
	int nDecodeSurface = GetNumDecodeSurfaces(pVideo->codec, pVideo->coded_width, pVideo->coded_height);
	CUVIDDECODECREATEINFO stDecodeInfo = { 0 };
	stDecodeInfo.bitDepthMinus8 = pVideo->bit_depth_chroma_minus8;
	stDecodeInfo.OutputFormat = pVideo->bit_depth_luma_minus8 ? cudaVideoSurfaceFormat_P016 : cudaVideoSurfaceFormat_NV12;
	stDecodeInfo.ChromaFormat = pVideo->chroma_format;
	stDecodeInfo.CodecType = pVideo->codec;
	stDecodeInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;  //???
	stDecodeInfo.ulNumDecodeSurfaces = nDecodeSurface; //??
	stDecodeInfo.ulNumOutputSurfaces = 2;  //??
	stDecodeInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;  //?
	stDecodeInfo.vidLock = m_ctxLock;
	stDecodeInfo.ulHeight = pVideo->coded_height;
	stDecodeInfo.ulWidth = pVideo->coded_width;
	if (!(m_Crop.b && m_Crop.r) && (!(m_FrameSize.h && m_FrameSize.w)))
	{
		m_Height = pVideo->display_area.bottom - pVideo->display_area.top;
		m_Width = pVideo->display_area.right - pVideo->display_area.left;
		stDecodeInfo.ulTargetHeight = pVideo->coded_height;
		stDecodeInfo.ulTargetWidth = pVideo->coded_width;
	}
	else
	{
		if (m_FrameSize.w && m_FrameSize.h)
		{
			stDecodeInfo.display_area.bottom = pVideo->display_area.bottom;
			stDecodeInfo.display_area.top = pVideo->display_area.top;
			stDecodeInfo.display_area.right = pVideo->display_area.right;
			stDecodeInfo.display_area.left = pVideo->display_area.left;
			m_Height = m_FrameSize.h;
			m_Width = m_FrameSize.w;
		}
		if (m_Crop.b && m_Crop.r)
		{
			stDecodeInfo.display_area.bottom = m_Crop.b;
			stDecodeInfo.display_area.top = m_Crop.t;
			stDecodeInfo.display_area.right = m_Crop.r;
			stDecodeInfo.display_area.left = m_Crop.l;
			m_Height = m_Crop.b - m_Crop.t;
			m_Width = m_Crop.r - m_Crop.l;
		}
		stDecodeInfo.ulTargetHeight = m_Height;
		stDecodeInfo.ulTargetWidth = m_Width;
	}
	m_nSurfaceHeight = stDecodeInfo.ulTargetHeight;
	NVDEC_API_CALL(cuCtxPushCurrent(m_cuContext));
	NVDEC_API_CALL(cuvidCreateDecoder(&h_Decoder, &stDecodeInfo));
	NVDEC_API_CALL(cuCtxPopCurrent(NULL));
	return  1;
}

int VideoDecode::HandleDecodePicture(CUVIDPICPARAMS * pPictureParams)
{
	if (h_Decoder == NULL)
	{
		std::cout << "Decoder not yet initalization" << std::endl;
		return -1;
	}
	NVDEC_API_CALL(cuvidDecodePicture(h_Decoder, pPictureParams));
	return 1;
}
int VideoDecode::HandleDisplayInfo(CUVIDPARSERDISPINFO * pDisplay)
{
	CUVIDPROCPARAMS pfProgress = { 0 };
	pfProgress.progressive_frame = pDisplay->progressive_frame;
	pfProgress.second_field = pDisplay->repeat_first_field + 1;
	pfProgress.top_field_first = pDisplay->top_field_first;
	pfProgress.unpaired_field = pDisplay->repeat_first_field < 0;
	pfProgress.output_stream = m_cuvidStream;

	CUdeviceptr dpSrcFrame = 0;
	unsigned int nSrcPitch = 0;

	NVDEC_API_CALL(cuvidMapVideoFrame(h_Decoder, pDisplay->picture_index, &dpSrcFrame, &nSrcPitch, &pfProgress));

	uint8_t * ptrFrame = nullptr;
	uint8_t *pDecodedFrame = nullptr;
	{
		std::lock_guard<std::mutex> lock(m_mtxVPFrame);
		if ((unsigned)++m_nDecodedFrame > m_vpFrame.size())
		{
			// Not enough frames in stock
			m_nFrameAlloc++;
			uint8_t *pFrame = nullptr;
			if (m_bUseDeviceFrame)
			{
				NVDEC_API_CALL(cuCtxPushCurrent(m_cuContext));
				if (m_bDeviceFramePitched)
				{
					NVDEC_API_CALL(cuMemAllocPitch((CUdeviceptr *)&pFrame, &m_nDeviceFramePitch, m_Width*GetVideoByte(), m_Height*3/2, 16));
				}
				else
				{
					NVDEC_API_CALL(cuMemAlloc((CUdeviceptr *)&pFrame,GetFrameSize()));
				}
				NVDEC_API_CALL(cuCtxPopCurrent(NULL));
			}
			else
			{
				pFrame = new uint8_t[GetFrameSize()];
			}
			m_vpFrame.push_back(pFrame);
		}
		pDecodedFrame = m_vpFrame[m_nDecodedFrame - 1];
	}

	NVDEC_API_CALL(cuCtxPushCurrent(m_cuContext));
	CUDA_MEMCPY2D m = { 0 };
	m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	m.srcDevice = dpSrcFrame;
	m.srcPitch = nSrcPitch;
	m.dstMemoryType = m_bUseDeviceFrame ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
	m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame);
	m.dstPitch = m_nDeviceFramePitch ? m_nDeviceFramePitch : m_Width * GetVideoByte();
	m.WidthInBytes = m_Width * GetVideoByte();
	m.Height = m_Height;
	NVDEC_API_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));
	m.srcDevice = (CUdeviceptr)((uint8_t *)dpSrcFrame + m.srcPitch * m_nSurfaceHeight);
	m.dstDevice = (CUdeviceptr)(m.dstHost = pDecodedFrame + m.dstPitch * m_Height);
	m.Height = m_Height / 2;
	NVDEC_API_CALL(cuMemcpy2DAsync(&m, m_cuvidStream));
	NVDEC_API_CALL(cuStreamSynchronize(m_cuvidStream));
	NVDEC_API_CALL(cuCtxPopCurrent(NULL));

	if ((int)m_vTimestamp.size() < m_nDecodedFrame) {
		m_vTimestamp.resize(m_vpFrame.size());
	}
	m_vTimestamp[m_nDecodedFrame - 1] = pDisplay->timestamp;
	NVDEC_API_CALL(cuvidUnmapVideoFrame(h_Decoder, dpSrcFrame));
	return 1;
}
VideoDecode::VideoDecode(CUcontext &context,int height,int width,int bitdepth,
	cudaVideoCodec codec, std::mutex * mutex,bool useDeviceFrame,bool useDeviceFramePitch)
	:m_cuContext(context),m_eCodec(codec),m_bUseDeviceFrame(useDeviceFrame),
	m_bDeviceFramePitched(useDeviceFramePitch),m_Height(height),m_Width(width),m_nBitDepth(bitdepth)
{
	cuvidCtxLockCreate(&m_ctxLock, m_cuContext);

	CUVIDPARSERPARAMS stParser = {};
	
	stParser.CodecType = m_eCodec;
	stParser.ulMaxNumDecodeSurfaces = 1;      //??
	stParser.ulMaxDisplayDelay = false ^ 0x01; //??
	stParser.pUserData = this;
	stParser.pfnSequenceCallback = HandleVideoProc;
	stParser.pfnDecodePicture = HandlePictureProc;
	stParser.pfnDisplayPicture = HandleDisplayProc;
	if (m_pMutex) m_pMutex->lock();
	NVDEC_API_CALL(cuvidCreateVideoParser(&h_Parser, &stParser));
	if (m_pMutex) m_pMutex->unlock();
}

VideoDecode::~VideoDecode()
{
	cuvidDestroyVideoParser(h_Parser);
	cuvidDestroyDecoder(h_Decoder);
	for (uint8_t * pFrame : m_vpFrame)
	{
		if (m_bUseDeviceFrame)
		{
			if (m_pMutex) m_pMutex->lock();
			NVDEC_API_CALL(cuCtxPushCurrent(m_cuContext));
			cuMemFree((CUdeviceptr)pFrame);
			NVDEC_API_CALL(cuCtxPopCurrent(NULL));
			if (m_pMutex) m_pMutex->unlock();
		}
		else
		{
			delete[] pFrame;
		}
	}
}

bool VideoDecode::Decode(uint8_t* pData,int size,uint8_t *** p3DFrameOut, int * pnFrameReturn, int64_t ** ppTimeStamp,int nTimeStamp,int nFlags,CUstream vStream)
{
	m_nDecodedFrame = 0;
	CUVIDSOURCEDATAPACKET packet = { 0 };
	packet.payload = pData;
	packet.payload_size = size;
	packet.flags = nFlags | CUVID_PKT_TIMESTAMP;
	packet.timestamp = nTimeStamp;
	if (!pData || size == 0) {
		packet.flags |= CUVID_PKT_ENDOFSTREAM;
	}
	m_cuvidStream = vStream;
	if (m_pMutex) m_pMutex->lock();
	NVDEC_API_CALL(cuvidParseVideoData(h_Parser, &packet));
	if (m_pMutex) m_pMutex->unlock();
	m_cuvidStream = 0;

	if (m_nDecodedFrame > 0)
	{
		if (p3DFrameOut)
		{
			m_vpFrameRet.clear();
			std::lock_guard<std::mutex> lock(m_mtxVPFrame);
			m_vpFrameRet.insert(m_vpFrameRet.begin(), m_vpFrame.begin(), m_vpFrame.begin() + m_nDecodedFrame);
			*p3DFrameOut = &m_vpFrameRet[0];
		}
		if (ppTimeStamp)
		{
			*ppTimeStamp = &m_vTimestamp[0];
		}
	}
	if (pnFrameReturn)
	{
		*pnFrameReturn = m_nDecodedFrame;
	}
	return true;
}