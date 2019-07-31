
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/videoio.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <ctime>
#include <time.h>
#include <cstdio>
#include <sys/stat.h>
#include <math.h>
#include <iomanip>
#include <chrono>

using namespace cv;
using namespace std;

#define VIDEO_FILE_NAME "fvsa.mp4"
#define CASCADE_FILE_NAME "cascade.xml"
#define thresholdpercentage 15
#define usedecision 1
#define numberofthresholdframes 30
#define splitdisplay 0
#define biggest_bbox 0

//=======================================================variable declaration=================================================
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double initialwidth, thresholdwidth = 0;
VideoCapture cap;
Mat mFrame, mGray, imageROI, carROI;
CascadeClassifier cars;
vector<Rect> cars_found;
Mat frame;
bool isGetInitialWidth = true;
bool isStart = false;
bool isUsingDetector = true;
int thresholdcount = 0;
Ptr<Tracker> tracker;
vector<Rect2d> v2;
Rect detected_vehicle;
int y_carROI;
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//==========================================================================n=================================================



//=======================================================functions declaration=================================================
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void getVideoData()
{
	cap.open(VIDEO_FILE_NAME);
}

void loadDetector()
{
	cars.load(CASCADE_FILE_NAME);
}

void framePreprocessing()
{
	resize(mFrame, mFrame, Size(704, 480));
	imageROI = mFrame(Rect(mFrame.cols*0.25, mFrame.rows * 0, mFrame.cols*0.5, mFrame.rows));
	cvtColor(imageROI, mGray, COLOR_BGR2GRAY);	
}

bool compare_rect(const cv::Rect & a, const cv::Rect &b)
{
	return a.width > b.width;
}

double getDistance(Point a, Point b)
{
	double x_diff = a.x - b.x;
	double y_diff = a.y - b.y;
	return sqrt(x_diff*x_diff + y_diff*y_diff);
}

int get_index_bbox_centre(Mat ROI, vector<Rect> bbox)
{
	int index;
	Point ROI_centre;
	Point bbox_temp_centre;
	double min_distance = 10000;

	ROI_centre.x = 0 + ROI.cols / 2;
	ROI_centre.y = 0 + ROI.rows / 2;

	for (int i = 0; i < bbox.size(); i++)
	{
		bbox_temp_centre.x = bbox[i].x + bbox[i].width / 2;
		bbox_temp_centre.y = bbox[i].y + bbox[i].height / 2;
		double dist = getDistance(ROI_centre, bbox_temp_centre);
		if (dist < min_distance)
		{
			min_distance = dist;
			index = i;
		}
	}
	return index;
}

void detectVehicle()
{
	cars.detectMultiScale(mGray, cars_found, 1.05, 3, 0 | 2, Size(60, 60));
	cout << cars_found.size() << endl;
	if (cars_found.size() > 0)
	{
		isUsingDetector = false;

		if (biggest_bbox == 1)
		{
			sort(cars_found.begin(), cars_found.end(), compare_rect);
			detected_vehicle = cars_found[0];
		}
		else
		{
			int index = get_index_bbox_centre(mGray, cars_found);
			detected_vehicle = cars_found[index];
		}


		//make new ROI for tracking car

		if (detected_vehicle.y - 50 >= 0)
			y_carROI = detected_vehicle.y - 50;
		else
			y_carROI = 0;

		mGray(Rect(0, y_carROI, mGray.cols, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);

		cout << "detector box: " << detected_vehicle << "   ";
		cv::rectangle(mGray, detected_vehicle, Scalar(0, 0, 225));

		v2.push_back(Rect(detected_vehicle.x, (detected_vehicle.y - y_carROI), detected_vehicle.width, detected_vehicle.height));
		tracker->init(carROI, v2[0]);
	}
}

void trackVehicle()
{
	mGray(Rect(0, y_carROI, mGray.cols, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);

	cv::rectangle(mGray, detected_vehicle, Scalar(0, 0, 225)); //draw the detector box


	bool ok = tracker->update(carROI, v2[0]);

	if (ok)
	{
		// Tracking success : Draw the tracked object
		cout << "tracker box: " << v2[0] << "   ";
		rectangle(carROI, v2[0], Scalar(255, 0, 0), 2, 1);
		rectangle(mGray, Rect(v2[0].x, v2[0].y + y_carROI, v2[0].width, v2[0].height), Scalar(255, 0, 0), 2, 1);


		#if usedecision //=================================================================================================================
		if (isGetInitialWidth == true)
		{
			initialwidth = v2[0].width;
			thresholdwidth = initialwidth*(100 - thresholdpercentage) / 100;
			isGetInitialWidth = false;
		}
		//frame by frame checking whether tracked vehicle has moved away from the camera; indicated by bbox size reduced by thresholdvalue
		else
		{
			if (v2[0].width <= thresholdwidth)
			{
				thresholdcount++;
				if (thresholdcount > numberofthresholdframes)
				{
					putText(mGray, "!!!Warning!!! ", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
					putText(carROI, "!!!Warning!!! ", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
				}
			}

		}
		#endif//===========================================================================================================================


	}
	else
	{
		// Tracking failure detected.
		putText(mGray, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		putText(carROI, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
	}
}

void StartFVSA()
{
	// reset all flags to start FVSA
	isStart = true;
	isUsingDetector = true;
	isGetInitialWidth = true;
	thresholdcount = 0;
	tracker = TrackerCSRT::create();
	
}

void EndFVSA()
{
	isStart = false;
	if (v2.empty() == false)
	v2.pop_back();
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//============================================================================================================================



int main()
{
	int framecount = 0;

	loadDetector();
	getVideoData();

	while (cap.read(mFrame))
	{	

		auto start_time = std::chrono::high_resolution_clock::now();
			
		framePreprocessing();

		if (isStart == true)
		{			
			if (isUsingDetector == true)
			{
				detectVehicle();
			}

			else if (isUsingDetector == false)
			{
				trackVehicle();
			}
		}

		framecount++;

		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;

		std::cout << "Time: " <<
			time / std::chrono::milliseconds(1) << "ms.\n";

		
		if (isUsingDetector == true)
			cv::imshow("detection", mGray);
		else
		{
			cv::imshow("detection", mGray);
			cv::imshow("tracking", carROI);
		}

		int c = waitKey(1);
		if ((char)c == 27) 
		{ break; }
		else if ((char)c == 'q') //starts FVSA
		{
			if (isStart==false) // check if it is already starting or not
			{
				StartFVSA();
			}
		}
		else if ((char)c == 'w') //ends FVSA
		{
			if (isStart==true) // check if it is already starting or not
			{
				EndFVSA();
			}
		}
	
	}

	if (v2.empty() == false)
		v2.pop_back();
	return 0;
}
