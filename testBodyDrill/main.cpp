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
#include <sys/time.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/calib3d/calib3d.hpp>

#include <opencv2/opencv.hpp>
#include "cvui.h"

#define WINDOW_NAME "My Application"

using namespace std;
using namespace cv;

float elapsedMillis_{.0f};

using DurationType = std::chrono::milliseconds;
using ClockType = std::chrono::high_resolution_clock;

ClockType::time_point prev_;

cv::Mat cImageBGRLeft = Mat::zeros( 480, 640, CV_8UC3 );
cv::Mat cImageBGRRight = Mat::ones( 480, 640, CV_8UC3 );

using buffer_ptr = std::unique_ptr<astra::RgbPixel []>;
buffer_ptr buffer_;
unsigned int lastWidth_;
unsigned int lastHeight_;

// 3d vector for joints
astra::Vector3f jointHead3D, jointNeck3D;
astra::Vector3f jointLeftShoulder3D, jointLeftElbow3D, jointLeftHand3D;
astra::Vector3f jointRightShoulder3D, jointRightElbow3D, jointRightHand3D;
astra::Vector3f jointShoulderSpine3D, jointMidSpine3D, jointBaseSpine3D; 
astra::Vector3f jointLeftHip3D, jointLeftKnee3D, jointLeftFoot3D;
astra::Vector3f jointRightHip3D, jointRightKnee3D, jointRightFoot3D;
astra::Vector3f jointLeftWrist3D, jointRightWrist3D;

// 2d POINT for joints
cv::Point jointHead2D, jointNeck2D;
cv::Point jointLeftShoulder2D, jointLeftElbow2D, jointLeftHand2D;
cv::Point jointRightShoulder2D, jointRightElbow2D, jointRightHand2D;
cv::Point jointShoulderSpine2D, jointMidSpine2D, jointBaseSpine2D; 
cv::Point jointLeftHip2D, jointLeftKnee2D, jointLeftFoot2D;
cv::Point jointRightHip2D, jointRightKnee2D, jointRightFoot2D;
cv::Point jointLeftWrist2D, jointRightWrist2D;

astra::HandPose left_hand, right_hand; // palm close or open

Mat imgLeft, imgRight;
long int millisecondStart;

void MyLine( Mat img, Point start, Point end )
{
	int thickness = 2;
	int lineType = 8;
	const Scalar SCALAR_GREEN = Scalar(0, 255, 0);
	
	line( img, start, end, SCALAR_GREEN, thickness, lineType );
}

void MyFilledCircle( Mat img, Point center )
{
	int filled = -1;
	int lineType = 8;
	int radius = 5;
	const Scalar SCALAR_RED = Scalar(0, 0, 255);	
	
	circle( img, center, radius, SCALAR_RED, filled, lineType);
}

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

void output_joints_body_pose(astra::Body body, const int32_t bodyId, 
							const astra::JointList& joints, astra::Frame& frame)
{
	
	const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();
	
	cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
	cv::cvtColor( mImageRGB, cImageBGRLeft, CV_RGB2BGR );
	
	
	for (const auto& joint : joints)
    {
		// jointType is one of joints which exists for each joint type
		astra::JointType jointType = joint.type();
		astra::JointStatus jointStatus = joint.status();
		
		const auto& depthPos = joint.depth_position();
		const auto& worldPos = joint.world_position();
		
		// orientation is a 3x3 rotation matrix where the column vectors also
	    // represent the orthogonal basis vectors for the x, y, and z axes.
	    const auto& orientation = joint.orientation();
	    
	    //printf("Body orientation", orientation.x_axis());
	   
	    const auto& xAxis = orientation.x_axis(); // same as orientation->m00, m10, m20
	    const auto& yAxis = orientation.y_axis(); // same as orientation->m01, m11, m21
	    const auto& zAxis = orientation.z_axis(); // same as orientation->m02, m12, m22
		
		// jointStatus is one of:
		// ASTRA_JOINT_STATUS_NOT_TRACKED = 0,
		// ASTRA_JOINT_STATUS_LOW_CONFIDENCE = 1,
		// ASTRA_JOINT_STATUS_TRACKED = 2,
		
		if (joint.status() == astra::JointStatus::NotTracked)
		{
			continue;
		}
		else
		{
			/*printf("Body %u Joint %d status %d @ world (%.1f, %.1f, %.1f) depth (%.1f, %.1f)\n",
			bodyId,
			jointType,
			jointStatus,
			worldPos.x,
			worldPos.y,
			worldPos.z,
			depthPos.x,
			depthPos.y);
			*/
			
			//printf("Joint orientation x: [%f %f %f]\n", xAxis.x, xAxis.y, xAxis.z);
			//printf("Joint orientation y: [%f %f %f]\n", yAxis.x, yAxis.y, yAxis.z);
			//printf("Joint orientation z: [%f %f %f]\n", zAxis.x, zAxis.y, zAxis.z);
			
			
			if ((int)jointType == 0) // Head
			{
				jointHead3D.x = worldPos.x;
				jointHead3D.y = worldPos.y;
				jointHead3D.z = worldPos.z;
				
				jointHead2D.x = depthPos.x;
				jointHead2D.y = depthPos.y;
			}
			else if ((int)jointType == 18) // Neck
			{
				jointNeck3D.x = worldPos.x;
				jointNeck3D.y = worldPos.y;
				jointNeck3D.z = worldPos.z;
				
				jointNeck2D.x = depthPos.x;
				jointNeck2D.y = depthPos.y;
			}
			else if ((int)jointType == 1) // ShoulderSpine
			{
				jointShoulderSpine3D.x = worldPos.x;
				jointShoulderSpine3D.y = worldPos.y;
				jointShoulderSpine3D.z = worldPos.z;
				
				jointShoulderSpine2D.x = depthPos.x;
				jointShoulderSpine2D.y = depthPos.y;

			}
			else if ((int)jointType == 8) // MidSpine
			{
				jointMidSpine3D.x = worldPos.x;
				jointMidSpine3D.y = worldPos.y;
				jointMidSpine3D.z = worldPos.z;
				
				jointMidSpine2D.x = depthPos.x;
				jointMidSpine2D.y = depthPos.y;
			}
			else if ((int)jointType == 9) // BaseSpine
			{
				jointBaseSpine3D.x = worldPos.x;
				jointBaseSpine3D.y = worldPos.y;
				jointBaseSpine3D.z = worldPos.z;
				
				jointBaseSpine2D.x = depthPos.x;
				jointBaseSpine2D.y = depthPos.y;
			}
			else if ((int)jointType == 5) // RightShoulder
			{
				jointRightShoulder3D.x = worldPos.x;
				jointRightShoulder3D.y = worldPos.y;
				jointRightShoulder3D.z = worldPos.z;
				
				jointRightShoulder2D.x = depthPos.x;
				jointRightShoulder2D.y = depthPos.y;
			}
			else if ((int)jointType == 6) // RightElbow
			{
				jointRightElbow3D.x = worldPos.x;
				jointRightElbow3D.y = worldPos.y;
				jointRightElbow3D.z = worldPos.z;
				
				jointRightElbow2D.x = depthPos.x;
				jointRightElbow2D.y = depthPos.y;
			}
			else if ((int)jointType == 17) // RightWrist
			{
				jointRightWrist3D.x = worldPos.x;
				jointRightWrist3D.y = worldPos.y;
				jointRightWrist3D.z = worldPos.z;
				
				jointRightWrist2D.x = depthPos.x;
				jointRightWrist2D.y = depthPos.y;
			}
			else if ((int)jointType == 7) // RightHand
			{
				jointRightHand3D.x = worldPos.x;
				jointRightHand3D.y = worldPos.y;
				jointRightHand3D.z = worldPos.z;
				
				jointRightHand2D.x = depthPos.x;
				jointRightHand2D.y = depthPos.y;
			}
			else if ((int)jointType == 2) // LeftShoulder
			{
				jointLeftShoulder3D.x = worldPos.x;
				jointLeftShoulder3D.y = worldPos.y;
				jointLeftShoulder3D.z = worldPos.z;
				
				jointLeftShoulder2D.x = depthPos.x;
				jointLeftShoulder2D.y = depthPos.y;
			}
			else if ((int)jointType == 3) // LeftElbow
			{	
				jointLeftElbow3D.x = worldPos.x;
				jointLeftElbow3D.y = worldPos.y;
				jointLeftElbow3D.z = worldPos.z;
				
				jointLeftElbow2D.x = depthPos.x;
				jointLeftElbow2D.y = depthPos.y;
			}
			else if ((int)jointType == 16) // LeftWrist
			{
				jointLeftWrist3D.x = worldPos.x;
				jointLeftWrist3D.y = worldPos.y;
				jointLeftWrist3D.z = worldPos.z;
				
				jointLeftWrist2D.x = depthPos.x;
				jointLeftWrist2D.y = depthPos.y;
			}
			else if ((int)jointType == 4) // LeftHand
			{
				jointLeftHand3D.x = worldPos.x;
				jointLeftHand3D.y = worldPos.y;
				jointLeftHand3D.z = worldPos.z;
				
				jointLeftHand2D.x = depthPos.x;
				jointLeftHand2D.y = depthPos.y;
			}
			
			else if ((int)jointType == 10) // LeftHip
			{
				jointLeftHip3D.x = worldPos.x;
				jointLeftHip3D.y = worldPos.y;
				jointLeftHip3D.z = worldPos.z;
				
				jointLeftHip2D.x = depthPos.x;
				jointLeftHip2D.y = depthPos.y;
			}
			else if ((int)jointType == 11) // LeftKnee
			{
				jointLeftKnee3D.x = worldPos.x;
				jointLeftKnee3D.y = worldPos.y;
				jointLeftKnee3D.z = worldPos.z;
				
				jointLeftKnee2D.x = depthPos.x;
				jointLeftKnee2D.y = depthPos.y;
			}
			else if ((int)jointType == 12) // LeftFoot
			{
				jointLeftFoot3D.x = worldPos.x;
				jointLeftFoot3D.y = worldPos.y;
				jointLeftFoot3D.z = worldPos.z;
				
				jointLeftFoot2D.x = depthPos.x;
				jointLeftFoot2D.y = depthPos.y;
			}
			else if ((int)jointType == 13) // RightHip
			{
				jointRightHip3D.x = worldPos.x;
				jointRightHip3D.y = worldPos.y;
				jointRightHip3D.z = worldPos.z;
				
				jointRightHip2D.x = depthPos.x;
				jointRightHip2D.y = depthPos.y;
			}
			else if ((int)jointType == 14) // RightKnee
			{
				jointRightKnee3D.x = worldPos.x;
				jointRightKnee3D.y = worldPos.y;
				jointRightKnee3D.z = worldPos.z;
				
				jointRightKnee2D.x = depthPos.x;
				jointRightKnee2D.y = depthPos.y;
			}
			else if ((int)jointType == 15) // RightFoot
			{
				jointRightFoot3D.x = worldPos.x;
				jointRightFoot3D.y = worldPos.y;
				jointRightFoot3D.z = worldPos.z;
				
				jointRightFoot2D.x = depthPos.x;
				jointRightFoot2D.y = depthPos.y;
			}
		
		}
		
		/*! Hand pose is not known or unrecognized */
	    //Unknown = 0,
	    /*! Grip pose */
	    //Grip = 1,
		
		const auto& handPoses = body.hand_poses();
	
		const auto& leftHandPose = handPoses.left_hand();
		const auto& rightHandPose = handPoses.right_hand();
		
		//cout<<"Left hand pose : "<< (int)leftHandPose <<endl;
	
	    //printf("Body %d Left hand pose: %d Right hand pose: %d\n",
		//		body.id(),
		//		leftHandPose,
		//		rightHandPose);
	}
	
	astra::BodyFrame bodyFrame = frame.get<astra::BodyFrame>();
	const auto& bodyMask = bodyFrame.body_mask();
	
	auto bb = bodyMask.data();

	int h = bodyMask.height();
	int w = bodyMask.width();


	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			
			if (bb[(x + (y * w))] > 0)
			{
				cImageBGRLeft.at<Vec3b>(y,x)[0] = 255;
				cImageBGRLeft.at<Vec3b>(y,x)[1] = 0;
				cImageBGRLeft.at<Vec3b>(y,x)[2] = 0;
			}
			else
			{
				cImageBGRLeft.at<Vec3b>(y,x)[0] = cImageBGRLeft.at<Vec3b>(y,x)[0];
				cImageBGRLeft.at<Vec3b>(y,x)[1] = cImageBGRLeft.at<Vec3b>(y,x)[1];
				cImageBGRLeft.at<Vec3b>(y,x)[2] = cImageBGRLeft.at<Vec3b>(y,x)[2];
			}
	
		}
	}
	
	// Head 
	MyFilledCircle( cImageBGRRight, jointHead2D); // head

	// Spine
	MyFilledCircle( cImageBGRRight, jointNeck2D);
	MyFilledCircle( cImageBGRRight, jointShoulderSpine2D);
	MyFilledCircle( cImageBGRRight, jointMidSpine2D);
	MyFilledCircle( cImageBGRRight, jointBaseSpine2D);
	
	// Left Hand 
	MyFilledCircle( cImageBGRRight, jointLeftShoulder2D);
	MyFilledCircle( cImageBGRRight, jointLeftElbow2D);
	MyFilledCircle( cImageBGRRight, jointLeftWrist2D);
	MyFilledCircle( cImageBGRRight, jointLeftHand2D);
	
	// Right Hand
	MyFilledCircle( cImageBGRRight, jointRightShoulder2D);
	MyFilledCircle( cImageBGRRight, jointRightElbow2D);
	MyFilledCircle( cImageBGRRight, jointRightWrist2D);
	MyFilledCircle( cImageBGRRight, jointRightHand2D);
	
	// Left Leg
	MyFilledCircle( cImageBGRRight, jointLeftHip2D);
	MyFilledCircle( cImageBGRRight, jointLeftKnee2D);
	MyFilledCircle( cImageBGRRight, jointLeftFoot2D);
	
	// Right Leg
	MyFilledCircle( cImageBGRRight, jointRightHip2D);
	MyFilledCircle( cImageBGRRight, jointRightKnee2D);
	MyFilledCircle( cImageBGRRight, jointRightFoot2D);
		
	// Draw bone
	MyLine( cImageBGRRight, jointNeck2D, jointRightShoulder2D );
	MyLine( cImageBGRRight, jointNeck2D, jointHead2D );
	MyLine( cImageBGRRight, jointNeck2D, jointLeftShoulder2D );
	
	MyLine( cImageBGRRight, jointRightShoulder2D, jointRightElbow2D );
	MyLine( cImageBGRRight, jointRightElbow2D, jointRightWrist2D );
	MyLine( cImageBGRRight, jointRightWrist2D, jointRightHand2D );
	
	MyLine( cImageBGRRight, jointLeftShoulder2D, jointLeftElbow2D );
	MyLine( cImageBGRRight, jointLeftElbow2D, jointLeftWrist2D );
	MyLine( cImageBGRRight, jointLeftWrist2D, jointLeftHand2D );
	
	MyLine( cImageBGRRight, jointNeck2D, jointShoulderSpine2D );
	MyLine( cImageBGRRight, jointShoulderSpine2D, jointMidSpine2D );
	MyLine( cImageBGRRight, jointMidSpine2D, jointBaseSpine2D );
	
	MyLine( cImageBGRRight, jointBaseSpine2D, jointLeftHip2D );
	MyLine( cImageBGRRight, jointLeftHip2D, jointLeftKnee2D );
	MyLine( cImageBGRRight, jointLeftKnee2D, jointLeftFoot2D );
	
	MyLine( cImageBGRRight, jointBaseSpine2D, jointRightHip2D );
	MyLine( cImageBGRRight, jointRightHip2D, jointRightKnee2D );
	MyLine( cImageBGRRight, jointRightKnee2D, jointRightFoot2D );
}


void processBodies(astra::Frame& frame)
{
	astra::BodyFrame bodyFrame = frame.get<astra::BodyFrame>();
	
	const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();
	
	cv::Mat mImageRGB(colorFrame.height(), colorFrame.width(), CV_8UC3, (void*)colorFrame.data());
	cv::cvtColor( mImageRGB, cImageBGRLeft, CV_RGB2BGR );
	
	cImageBGRLeft.copyTo(cImageBGRRight);
	
	const auto& bodyMask = bodyFrame.body_mask();
	const auto& bodies = bodyFrame.bodies();
	
	for (auto& body : bodies)
	{
		//printf("Processing frame #%d body %d left hand: %u\n",
		//	bodyFrame.frame_index(), body.id(), unsigned(body.hand_poses().left_hand()));
		
		// Pixels in the body mask with the same value as bodyId are
        // from the same body.
        const auto& bodyId = body.id();

        // bodyStatus is one of:
        // ASTRA_BODY_STATUS_NOT_TRACKING = 0,
        // ASTRA_BODY_STATUS_LOST = 1,
        // ASTRA_BODY_STATUS_TRACKING_STARTED = 2,
        // ASTRA_BODY_STATUS_TRACKING = 3,
        const auto& bodyStatus = body.status();
        
        //printf("Processing frame body left hand: %u\n", bodyStatus);
        
		if ((int)bodyStatus == 2) // ASTRA_BODY_STATUS_TRACKING_STARTED
        {
            //printf("Body Id: %d Status: Tracking started\n", bodyId);
        }
		if ((int)bodyStatus == 3) // ASTRA_BODY_STATUS_TRACKING
        {
            //printf("Body Id: %d Status: Tracking\n", bodyId);
        }
        
        if ((int)bodyStatus == 2 || (int)bodyStatus == 3)
        {
            const auto& joints = body.joints();
            
            if (joints.empty())
			{
				return;
			}	

            output_joints_body_pose(body, bodyId, joints, frame);
        }
        else if ((int)bodyStatus == 1) // ASTRA_BODY_STATUS_LOST
        {
            //printf("Body %u Status: Tracking lost.\n", bodyId);
        }
        else // bodyStatus == ASTRA_BODY_STATUS_NOT_TRACKING
        {
            //printf("Body Id: %d Status: Not Tracking\n", bodyId);
        }

	}
	
	
}

bool fexists(const std::string& filename)
{
	std::ifstream ifile(filename.c_str());
	return (bool)ifile;
}

int main(void)
{
	astra::initialize();
	
	set_key_handler();
	
	const char* licenseString = "<INSERT LICENSE KEY HERE>";
	orbbec_body_tracking_set_license(licenseString);
	
	astra::StreamSet streamSet;
	astra::StreamReader reader = streamSet.create_reader();
	
	auto colorStream = configure_color(reader);

	colorStream.start();

	reader.stream<astra::ColorStream>().enable_mirroring(1);

	reader.stream<astra::BodyStream>().start();
	
	cv::Mat frameBig = cv::Mat(700, 1350, CV_8UC3);
	// Init a OpenCV window and tell cvui to use it.
    cv::namedWindow(WINDOW_NAME);
    cvui::init(WINDOW_NAME);
    
// controls    
    bool start = false;
    bool exit = false;
    bool runApp = false;
    
// modes
    bool trainmode = false;
    bool testmode = false;
    
// train modes    
    bool mode01 = false;
    bool mode02 = false;
    bool mode03 = false;
    bool mode04 = false;
    int mode = 0;
    
    int trainframelength = 450;
    
    frameBig = cv::Scalar(135, 80, 20);
    
    cv::Mat frameLeft = cv::Mat(480, 640, CV_8UC3);
    cv::Mat frameRight = cv::Mat(480, 640, CV_8UC3);
    
    frameLeft = cv::Scalar(255, 0, 0);
    frameRight = cv::Scalar(0, 255, 0);
    
    bool videoMode01Exist;
    bool videoMode02Exist;
    bool videoMode03Exist;
    bool videoMode04Exist;
    
    string fileName01 = "outMode01.avi";
    string fileName02 = "outMode02.avi";
    string fileName03 = "outMode03.avi";
    string fileName04 = "outMode04.avi";
    
    videoMode01Exist = fexists(fileName01);
    videoMode02Exist = fexists(fileName02);
    videoMode03Exist = fexists(fileName03);
    videoMode04Exist = fexists(fileName04);
    
	// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
	//####### Train modes
	VideoWriter video01;
	VideoWriter video02;
	VideoWriter video03;
	VideoWriter video04;
	
	if (!videoMode01Exist)
	{
		video01.open("outMode01.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480));
	
	}
	if (!videoMode02Exist)
	{
		video02.open("outMode02.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480));
	
	}
	if (!videoMode03Exist)
	{
		video03.open("outMode03.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480));
	
	}
	if (!videoMode04Exist)
	{
		video04.open("outMode04.avi", CV_FOURCC('M','J','P','G'), 30, Size(640,480));
	
	}
 
 
    while (true) 
    {
        // Show everything on the screen
        cv::imshow(WINDOW_NAME, frameBig);
        
        if (cvui::button(frameBig, 400, 640, "START"))
		{
			start = true;
			exit = false;
			// The button was clicked, so let's increment our counter.
			cout<<"Program Started "<<endl;
			//break; // do something here
		}
		
		// Checkboxes also accept a pointer to a variable that controls
		// the state of the checkbox (checked or not). cvui::checkbox() will
		// automatically update the value of the boolean after all
		// interactions, but you can also change it by yourself. Just
		// do "checked = true" somewhere and the checkbox will change
		// its appearance.
		cvui::checkbox(frameBig, 150, 20, "Train Mode", &trainmode);
		if (cvui::button(frameBig, 100, 40, "Select Train Mode"))
		{
			trainmode = true;
			testmode = false;
		}
		
		cvui::checkbox(frameBig, 300, 20, "Train Mode", &testmode);
		if (cvui::button(frameBig, 300, 40, "Select Test Mode"))
		{
			trainmode = false;
			testmode = true;
		}
		
		cvui::checkbox(frameBig, 600, 20, "Mode 1", &mode01);
		if (cvui::button(frameBig, 540, 40, "Select Mode 01"))
		{
			mode02 = false;
			mode03 = false;
			mode04 = false;
			mode01 = true;
			mode = 1;
		}
		
		cvui::checkbox(frameBig, 750, 20, "Mode 2", &mode02);
		if (cvui::button(frameBig, 750, 40, "Select Mode 02"))
		{
			mode03 = false;
			mode04 = false;
			mode01 = false;
			mode02 = true;
			mode = 2;
		}
		
		cvui::checkbox(frameBig, 1000, 20, "Mode 3", &mode03);
		if (cvui::button(frameBig, 940, 40, "Select Mode 03"))
		{
			mode04 = false;
			mode01 = false;
			mode02 = false;
			mode03 = true;
			mode = 3;
		}
		
		cvui::checkbox(frameBig, 1150, 20, "Mode 4", &mode04);
		if (cvui::button(frameBig, 1150, 40, "Select Mode 04"))
		{
			mode01 = false;
			mode02 = false;
			mode03 = false;
			mode04 = true;
			mode = 4;
		}
		
		
		cvui::window(frameBig, 05, 110, 660, 510, "Left Window");
		cvui::window(frameBig, 685, 110, 660, 510, "Right Window");

		cvui::image(frameBig, 15, 130, frameLeft);
		cvui::image(frameBig, 695, 130, frameRight);
		
		// Display the lib version at the bottom of the screen
		cvui::printf(frameBig, frameBig.cols - 80, frameBig.rows - 20, 0.4, 0xCECECE, "cvui v.%s", cvui::VERSION);

/*		
		if (cvui::button(frameBig, 200, 640, "RESET TYPE"))
		{
			trainmode = false;
			testmode = false;
		}
*/		
		
		
		if (start)
		{
			VideoCapture capStart("CountDown.mp4"); // open the start video template
			VideoCapture capStop("CountUp.mp4"); // open the stop video template
			
			
		    int frameCountStart = int(capStart.get(CV_CAP_PROP_FRAME_COUNT));
		    int frameCountStop = int(capStop.get(CV_CAP_PROP_FRAME_COUNT));
		   		    
		    if(!capStart.isOpened())  // check if we succeeded
		    {
				
				cout<<"Start Video"<<endl;
				return -1;
			}
			
		    if(!capStart.isOpened())  // check if we succeeded
		    {
				
				cout<<"RUNNINg"<<endl;
				return -1;
			}
			
		    //if(!capApp.isOpened())  // check if we succeeded
		    //{
				
				//cout<<"RUNNINg"<<endl;
				//return -1;
			//}
			
			//capApp.set(CV_CAP_PROP_FRAME_WIDTH,640);
			//capApp.set(CV_CAP_PROP_FRAME_HEIGHT,480);
			
			capStart.set(CV_CAP_PROP_FRAME_WIDTH,640);
			capStart.set(CV_CAP_PROP_FRAME_HEIGHT,480);
			
			capStop.set(CV_CAP_PROP_FRAME_WIDTH,640);
			capStop.set(CV_CAP_PROP_FRAME_HEIGHT,480);
			
			// Default resolution of the frame is obtained.The default resolution is system dependent. 
			int frame_widthStart = capStart.get(CV_CAP_PROP_FRAME_WIDTH); 
			int frame_heightStart = capStart.get(CV_CAP_PROP_FRAME_HEIGHT);
			
			int frame_widthStop = capStop.get(CV_CAP_PROP_FRAME_WIDTH); 
			int frame_heightStop = capStop.get(CV_CAP_PROP_FRAME_HEIGHT);
			
			//int frame_widthApp = capApp.get(CV_CAP_PROP_FRAME_WIDTH); 
			//int frame_heightApp = capApp.get(CV_CAP_PROP_FRAME_HEIGHT);
			
			cout<<"frame_width : "<<frame_widthStart<<", frame_height"<<frame_heightStart<<endl; 
			cout<<"frame_count : "<<frameCountStart<<endl;
			 
			cout<<"frame_width : "<<frame_widthStop<<", frame_height"<<frame_heightStop<<endl; 
			cout<<"frame_count : "<<frameCountStop<<endl;


			
			struct timeval tp;
			gettimeofday(&tp, NULL);
			millisecondStart = tp.tv_sec * 1000 + tp.tv_usec / 1000;
			String timeStamp = std::to_string(millisecondStart);
			
			// ########## Test Recordings
			string strVidUser = "Mode0"+std::to_string(mode) + "_" + timeStamp+".avi"; 
			VideoWriter videoUser(strVidUser, CV_FOURCC('M','J','P','G'), 30, Size(640,480));
			
			VideoCapture capApp("outMode0"+std::to_string(mode)+".avi"); // open the train mode video
			 
			cout<<"frame_width : "<<frame_widthStop<<", frame_height"<<frame_heightStop<<endl; 
			cout<<"frame_count : "<<frameCountStop<<endl;
		   
		    int frameIndex = 0;
// START
		    while(frameIndex<frameCountStart)
		    {
		        Mat frameStart;
		        Mat frameStartCrop;
		        
		        capStart >> frameStart;
		        resize(frameStart, frameStartCrop, Size(), round(640/frame_widthStart), 1, INTER_AREA );
		        
		        // Show everything on the screen
				cv::imshow(WINDOW_NAME, frameBig);
		        
		        cvui::image(frameBig, 15, 130, frameStartCrop);
				cvui::image(frameBig, 695, 130, frameStartCrop);
				
		        if(waitKey(1) >= 0) break;
		        
		        frameIndex++;
		        cout<<"frameIndex: "<<frameIndex<<endl;
		        
		        waitKey(10);
		        cvui::update();
		    }
// RUN		    
		    while(frameIndex>=frameCountStart && frameIndex<frameCountStart+trainframelength)
		    {
				astra::Frame frame = reader.get_latest_frame();
				processBodies(frame);
				
				// Show everything on the screen
				cv::imshow(WINDOW_NAME, frameBig);
				
				if (trainmode)
				{
					cvui::image(frameBig, 15, 130, cImageBGRLeft);
					cvui::image(frameBig, 695, 130, cImageBGRRight);
					
					// Write the frame into the file 'outcpp.avi'
					if ( mode == 1 )
					{
						if (!videoMode01Exist)
						{
							video01.write(cImageBGRLeft);
						}
					}
					// Write the frame into the file 'outcpp.avi'
					else if ( mode == 2 )
					{
						if (!videoMode02Exist)
						{
							video02.write(cImageBGRLeft);
						}
					}
					else if ( mode == 3 )
					{
						if (!videoMode03Exist)
						{
							video03.write(cImageBGRLeft);
						}
					}
					else if ( mode == 4 )
					{
						if (!videoMode04Exist)
						{
							video04.write(cImageBGRLeft);
						}
					}
				}
				if (testmode)
				{
					Mat frameRun;
					capApp >> frameRun;
					
					cvui::image(frameBig, 15, 130, frameRun);
					cvui::image(frameBig, 695, 130, cImageBGRLeft);
					
					videoUser.write(cImageBGRLeft);
					
				}	
						        
		        frameIndex++;
		        cout<<"frameIndex: "<<frameIndex<<endl;
		        
		        check_fps();
		        
		        astra_update();
		        cvui::update();
		
				if( cv::waitKey(1) == 'q')
					break;
		    }
		    // When everything done, release the video capture and write object
			
			if (testmode)
			{
				capApp.release();
				videoUser.release();
			}
	
			if (trainmode)
			{
				if (!videoMode01Exist)
				{
					video01.release();
				}
				if (!videoMode02Exist)
				{
					video02.release();
				}
				if (!videoMode03Exist)
				{
					video03.release();
				}
				if (!videoMode01Exist)
				{
					video04.release();
				}
			}
// STOP		    
			while(frameIndex>=(frameCountStart+trainframelength) && frameIndex<(frameCountStart+trainframelength+frameCountStop))
		    {
		        Mat frameStop;
		        Mat frameStopCrop;
		        
		        capStop >> frameStop;
		        resize(frameStop, frameStopCrop, Size(), round(640/frame_widthStop), 1, INTER_AREA );
		        
		        // Show everything on the screen
				cv::imshow(WINDOW_NAME, frameBig);
		        
				cvui::image(frameBig, 15, 130, frameStopCrop);
				cvui::image(frameBig, 695, 130, frameStopCrop);
				
		        if(waitKey(1) >= 0) break;
		        
		        frameIndex++;
		        cout<<"frameIndex: "<<frameIndex<<endl;
		        
		        waitKey(10);
		        
		        cvui::update();
		    }
		    
			start = false;
			exit = false;
		}
		
		
		if (cvui::button(frameBig, 900, 640, "Exit!"))
		{
			// The button was clicked, so let's increment our counter.
			cout<<" Exit : "<<endl;
			
			start = false;
			exit = true;
			
		}
		
		if(exit)
		{
			astra::terminate();
			break;			
		}
		
		// Update cvui internal stuff
        cvui::update();
		
		// Check if ESC key was pressed
		if (cv::waitKey(1) == 27) 
		{
			break;
		}
    }
	
	// Closes all the frames
	destroyAllWindows();

	return 0;
}
