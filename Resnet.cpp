// Resnet.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <sstream>
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"
#include "VideoDecode.h"
#include "cuda.h"
#include "cudnn.h"
#include "NvDecoder/nvcuvid.h"
#include "NvDecoder/cuviddec.h"
//#include "dynlink_cuda.h"
#include <ctime>

#define TESTPLANAR false
#define ASSERT(x) 
#define PATH 'e:\\Youtube\\sample_per_title\\presentation.mp4'
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

std::ofstream fpOut("e:\\Youtube\\sample_per_title\\presentation_2.yuv", std::ios::out | std::ios::binary);

void ShowHelpAndExit(const char *szBadOption = NULL)
{
	bool bThrowError = false;
	std::ostringstream oss;
	if (szBadOption)
	{
		bThrowError = true;
		oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
	}
	oss << "Options:" << std::endl
		<< "-i             Input file path" << std::endl
		<< "-o             Output file path" << std::endl
		<< "-outplanar     Convert output to planar format" << std::endl
		<< "-gpu           Ordinal of GPU to use" << std::endl
		<< "-crop l,t,r,b  Crop rectangle in left,top,right,bottom (ignored for case 0)" << std::endl
		<< "-resize WxH    Resize to dimension W times H (ignored for case 0)" << std::endl
		;
	oss << std::endl;
	if (bThrowError)
	{
		throw std::invalid_argument(oss.str());
	}
	else
	{
		std::cout << oss.str();
		//ShowDecoderCapability();
		exit(0);
	}
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName,
	bool &bOutPlanar, int &iGpu, Rect_t &cropRect, Mat_t &resizeDim)
{
	std::ostringstream oss;
	int i;
	for (i = 1; i < argc; i++) {
		if (!_stricmp(argv[i], "-h")) {
			ShowHelpAndExit();
		}
		if (!_stricmp(argv[i], "-i")) {
			if (++i == argc) {
				ShowHelpAndExit("-i");
			}
			sprintf(szInputFileName, "%s", argv[i]);
			continue;
		}
		if (!_stricmp(argv[i], "-o")) {
			if (++i == argc) {
				ShowHelpAndExit("-o");
			}
			sprintf(szOutputFileName, "%s", argv[i]);
			continue;
		}
		if (!_stricmp(argv[i], "-outplanar")) {
			bOutPlanar = true;
			continue;
		}
		if (!_stricmp(argv[i], "-gpu")) {
			if (++i == argc) {
				ShowHelpAndExit("-gpu");
			}
			iGpu = atoi(argv[i]);
			continue;
		}
		if (!_stricmp(argv[i], "-crop")) {
			if (++i == argc || 4 != sscanf(
				argv[i], "%d,%d,%d,%d",
				&cropRect.l, &cropRect.t, &cropRect.r, &cropRect.b)) {
				ShowHelpAndExit("-crop");
			}
			if ((cropRect.r - cropRect.l) % 2 == 1 || (cropRect.b - cropRect.t) % 2 == 1) {
				std::cout << "Cropping rect must have width and height of even numbers" << std::endl;
				exit(1);
			}
			continue;
		}
		if (!_stricmp(argv[i], "-resize")) {
			if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &resizeDim.w, &resizeDim.h)) {
				ShowHelpAndExit("-resize");
			}
			if (resizeDim.w % 2 == 1 || resizeDim.h % 2 == 1) {
				std::cout << "Resizing rect must have width and height of even numbers" << std::endl;
				exit(1);
			}
			continue;
		}
		ShowHelpAndExit(argv[i]);
	}
}


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

