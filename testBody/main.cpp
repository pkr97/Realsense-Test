// This file is part of the Orbbec Astra SDK [https://orbbec3d.com]
// Copyright (c) 2015-2017 Orbbec 3D
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Be excellent to each other.
#include <astra/astra.hpp>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <key_handler.h>
#include <sstream>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;
using namespace cv;


float elapsedMillis_{.0f};

using DurationType = std::chrono::milliseconds;
using ClockType = std::chrono::high_resolution_clock;

ClockType::time_point prev_;

using buffer_ptr = std::unique_ptr<astra::RgbPixel []>;
buffer_ptr buffer_;
unsigned int lastWidth_;
unsigned int lastHeight_;

void check_fps()
{
	const float frameWeight = .2f;
	
	const ClockType::time_point now = ClockType::now();
	const float elapsedMillis = std::chrono::duration_cast<DurationType>(now - prev_).count();
	
	elapsedMillis_ = elapsedMillis * frameWeight + elapsedMillis_ * (1.f - frameWeight);
	prev_ = now;
	
	const float fps = 1000.f / elapsedMillis;
	
	const auto precision = std::cout.precision();
	
	std::cout << std::fixed
	<< std::setprecision(1)
	<< fps << " fps ("
	<< std::setprecision(1)
	<< elapsedMillis_ << " ms)"
	<< std::setprecision(precision)<<endl;
}

void processBodies(astra::Frame& frame)
{
	astra::BodyFrame bodyFrame = frame.get<astra::BodyFrame>();
	
	const auto& bodyMask = bodyFrame.body_mask();
	
	// Create black empty images
	Mat image = Mat::zeros( 480, 640, CV_8UC3 );
	
	///Mat imgr(480, 640, CV_8UC3, Scalar(255, 255, 255));
	//const auto bodymask = bodyFrame.body_mask();
	auto bb = bodyMask.data();

	int h = bodyMask.height();
	int w = bodyMask.width();



	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			
			if (bb[(x + (y * w))] > 0)
			{
				image.at<Vec3b>(y,x)[0] = 255;
				image.at<Vec3b>(y,x)[1] = 0;
				image.at<Vec3b>(y,x)[2] = 0;
			}
			else
			{
				image.at<Vec3b>(y,x)[0] = 255;
				image.at<Vec3b>(y,x)[1] = 255;
				image.at<Vec3b>(y,x)[2] = 255;
			}
	
		}
	}
	
	imshow("Image",image);
	
	waitKey(1);
}	

int main(int argc, char** argv)
{
	astra::initialize();
	
	set_key_handler();
	
	const char* licenseString = "<INSERT LICENSE KEY HERE>";
	orbbec_body_tracking_set_license(licenseString);
	
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();

	reader.stream<astra::BodyStream>().start();

	do
	{
		astra::Frame frame = reader.get_latest_frame();
		processBodies(frame);
		check_fps();
		
		std::cout << "Running"<<endl;
		
		astra_update();
		
		if( cv::waitKey(1) == 'q')
			break;
	
	} while (shouldContinue);

	astra::terminate();
	
	return(0);
}
