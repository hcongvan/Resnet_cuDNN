// openCVRead.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
int mHeight = 720, mWidth = 1280;
std::vector<char> cvtYUV2RGB(char *Y, char *U, char *V)
{

	char *R, *G, *B;
	R = (char *)malloc(mHeight*mWidth * sizeof(char));
	G = (char *)malloc(mHeight*mWidth * sizeof(char));
	B = (char *)malloc(mHeight*mWidth * sizeof(char));

	for (int j = 0; j < mHeight; j++) {
		for (int i = 0; i < mWidth; i++) {
			R[j*mHeight + i] = Y[j*mHeight + i] + 1.370705*V[j*mHeight / 4 + i / 2];
			G[j*mHeight + i] = Y[j*mHeight + i] - 0.698001*V[j*mHeight / 4 + i / 2] - 0.337633 *U[j*mHeight / 4 + i / 2];
			B[j*mHeight + i] = Y[j*mHeight + i] + 1.732446*U[j*mHeight / 4 + i / 2];
			//Threshold 
			if (R[j*mHeight + i] < 0) R[j*mHeight + i] = 0;
			if (G[j*mHeight + i] < 0) G[j*mHeight + i] = 0;
			if (B[j*mHeight + i] < 0) B[j*mHeight + i] = 0;
			if (R[j*mHeight + i] > 255) R[j*mHeight + i] = 255;
			if (G[j*mHeight + i] > 255) G[j*mHeight + i] = 255;
			if (B[j*mHeight + i] > 255) B[j*mHeight + i] = 255;
		}
	}
	std::vector<char> _img(R, R + mHeight*mWidth);
	for (int i = 0; i < mHeight*mWidth; i++) {
		_img.push_back(G[i]);
	}
	for (int i = 0; i < mHeight*mWidth; i++) {
		_img.push_back(B[i]);
	}
	return _img;
}

int main()
{
	cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
	char *y,*u,*v;
	std::vector<char> img;
	y = (char *)malloc(mHeight*mWidth * sizeof(char));
	v = (char *)malloc(mHeight*mWidth * sizeof(char) / 4);
	u = (char *)malloc(mHeight*mWidth * sizeof(char) / 4);
	std::ifstream fsIn;
	fsIn.open("E:\\Youtube\\sample_per_title\\frames0.yuv", std::ios::in | std::ios::binary);
	fsIn.read(y,mHeight*mWidth);

	fsIn.read(u, mHeight*mWidth / 4);
	fsIn.read(v, mHeight*mWidth / 4);
	img = cvtYUV2RGB(y, u, v);
	Mat mimage(mHeight,mWidth,CV_8UC3,img.data(),0);
	imshow("Test",mimage);
	Mat test;
	test = imread("E:\\Youtube\\no0001_001.jpg");
	imshow("Test", test);
	cvtColor(test, test, cv::COLOR_RGB2GRAY);
	imshow("Test", test);

    return 0;
}

