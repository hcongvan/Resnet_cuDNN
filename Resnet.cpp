// Resnet.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <sstream>
#include "stdafx.h"
#include "Utils\NvCodecUtils.h"
#include "Utils\FFmpegDemuxer.h"
#include "VideoDecode.h"
#include "cuda.h"
#include "cudnn.h"
#include "NvDecoder\nvcuvid.h"
#include "NvDecoder\cuviddec.h"
#include "dynlink_cuda.h"
#include <ctime>

#define TESTPLANAR false
#define ASSERT(x) 
#define PATH 'e:\\Youtube\\sample_per_title\\presentation.mp4'



simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

std::ofstream fpOut("e:\\Youtube\\sample_per_title\\presentation_2.yuv", std::ios::out | std::ios::binary);

int main()
{
	ck(cuInit(0));
	int nGpu = 0;
	ck(cuDeviceGetCount(&nGpu));
	std::mutex m_Mutex;
	int mH, mW;
	int mBitDepth;
	CUdevice cuDevice = 0;
	cuDeviceGet(&cuDevice, 0);
	CUcontext cuContext;
	char szDeviceName[80];
	cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice);
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	cuCtxCreate(&cuContext, 0, cuDevice);
	//int * test;
	//CUresult result;
	//result = (CUresult)cudaMalloc((void **)test, 100000);
	std::clock_t mTimer;
	double duration;
	mTimer = std::clock();
	FFmpegDemuxer demux("e:\\Youtube\\sample_per_title\\presentation_1.mp4");
	VideoDecode vDecoder(cuContext,demux.GetHeight(),demux.GetWidth(),demux.GetBitDepth() ,FFmpeg2NvCodecId(demux.GetVideoCodec())
		,&m_Mutex, false, false);
	
	int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
	uint8_t * pFrame, **ppFrame;
	mH = vDecoder.GetHeight();
	mW = vDecoder.GetWidth();
	mBitDepth = vDecoder.GetBitDepth();
	int FrameSize = mW *vDecoder.GetVideoByte()* mH * 3 /2;
	do
	{
		demux.Demux(&pFrame, &nVideoBytes);
		vDecoder.Decode(pFrame,nVideoBytes,&ppFrame, &nFrameReturned, NULL,0,0,0);
		
		for (int i=0; i < nFrameReturned; i++)
		{
			//if (fpOut.is_open()) { std::cout << "Opened" << std::endl; }
			fpOut.write(reinterpret_cast<char*>(ppFrame[i]), FrameSize);
		}
		nFrame += nFrameReturned;
	} while (nVideoBytes);
	fpOut.close();
//dataset video, processed with NVIDIA video codec sdk
	duration = (std::clock() - mTimer) / CLOCKS_PER_SEC;
	std::cout << "Time to complete: " << duration << std::endl;
	


	return 0;
}

