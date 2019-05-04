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


astra::ColorStream configure_color(astra::StreamReader& reader)
{
	auto colorStream = reader.stream<astra::ColorStream>();
	
	auto oldMode = colorStream.mode();
	
	//We don't have to set the mode to start the stream, but if you want to here is how:
	astra::ImageStreamMode colorMode;
	
	colorMode.set_width(640);
	colorMode.set_height(480);
	colorMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_RGB888);
	colorMode.set_fps(30);
	
	colorStream.set_mode(colorMode);
	
	auto newMode = colorStream.mode();
	printf("Changed color mode: %dx%d @ %d -> %dx%d @ %d\n",
	oldMode.width(), oldMode.height(), oldMode.fps(),
	newMode.width(), newMode.height(), newMode.fps());
	
	return colorStream;
}

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

int main(int argc, char** argv)
{
	astra::initialize();
	
	set_key_handler();
	
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();

	auto colorStream = configure_color(reader);

	colorStream.start();

	reader.stream<astra::ColorStream>().enable_mirroring(1);
	
	std::cout << "colorStream -- hFov: "
	<< reader.stream<astra::ColorStream>().hFov()
	<< " vFov: "
	<< reader.stream<astra::ColorStream>().vFov()
	<< std::endl;
	do
	{
		astra::Frame frame = reader.get_latest_frame();
		const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();
		
		cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
		cv::Mat cImageBGR;
		cv::cvtColor( mImageRGB, cImageBGR, CV_RGB2BGR );

		cv::imshow( "Color Image", cImageBGR ); // RGB image       
		check_fps();
		
		cv::waitKey(1);

		astra_update();
		
		if( cv::waitKey(1) == 'q')
			break;
	
	} while (shouldContinue);

	astra::terminate();
	
	return(0);
}
