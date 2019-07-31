#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
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
//#include <windows.h>


using namespace cv;
using namespace std;

//#define VIDEO_FILE_NAME "fvsa.mp4"
//#define VIDEO_FILE_NAME "side_example.mp4"
//#define VIDEO_FILE_NAME "AWS/day/truck/sedan_1.mp4"
//#define VIDEO_FILE_NAME "AWS/day/truck/sedan_3.mp4"
//#define VIDEO_FILE_NAME "AWS/day/truck/suv_1.mp4"

//#define VIDEO_FILE_NAME "AWS/day/sedan/sedan_1.mp4"
//#define VIDEO_FILE_NAME "night3.mp4"
//#define VIDEO_FILE_NAME "snow.mp4"
//#define VIDEO_FILE_NAME "7.mp4" // wrong result
//#define VIDEO_FILE_NAME "suv_1.mp4"
//#define VIDEO_FILE_NAME "truck_1.mp4" //wrong result
//#define VIDEO_FILE_NAME "18.mp4" //not auto start
#define VIDEO_FILE_NAME "/home/rohit/adas/Record/test3.mp4" // not auto start

//#define CASCADE_FILE_NAME "cascade_325_90.xml"
#define CASCADE_FILE_NAME "/home/rohit/adas/car_lane_sign_detection/cascade_370_90.xml"
#define thresholdpercentage 2 //10 or 15 for csrt, 5 for kcf
#define usedecision 1
#define use_csrt 1
#define numberofthresholdframes 2 // if use saved video: depends on the saved video fps, not the algorithm fps!!!!!!!!!!!!!!!!
#define displayall 1
#define biggest_bbox_approach 0
#define trackingplot 0
#define timeperformanceplot 0
#define autostart 1
#define skipframe 10
#define use_size 0 // either use bbox size or y_position

#define CVUI_IMPLEMENTATION
#define debug 0 
//#include "cvui.h"

//=======================================================variable declaration=================================================
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double initialwidth, thresholdwidth = 0;
double initial_y, threshold_y = 0;
VideoCapture cap;
Mat mFrame, mGray, imageROI, carROI, scaled_carROI;
CascadeClassifier cars;
vector<Rect> cars_found;
Mat frame;
bool isGetInitialWidth = true;
bool isStart = false;
bool isUsingDetector = true;
int thresholdcount = 0;
Ptr<Tracker> tracker;
vector<Rect2d> v2;
Rect detected_vehicle, scaled_detected_vehicle;
int y_carROI, scaled_y_carROI;
double scale;
int carROI_fixedheight = 110;
int framecount = 0;

#if trackingplot
Mat plotdata(1000, 1, CV_64F);
//Mat plotthresholddata(1000, 1, CV_64F);
Mat plot_result;
#endif

#if timeperformanceplot
Mat plotdata1(1000, 1, CV_64F);
Mat plot_result1;
#endif
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//==========================================================================n=================================================



//=======================================================functions declaration=================================================
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



/*int fullDraw(cv::Mat frame, cv::Mat Image)
{
	int padding = 10;
	cv::Mat Grayframe;
	padding = 10;

	cvui::beginRow(frame, 45, 200, 100, 50, padding);

	if (!Image.empty())
	{
		//resize(Image, Image, Size(1000, 680));
		cvui::image(Image);
	}
	cvui::endRow();
	return 0;
}*/

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
	//resize(mFrame, mFrame, Size(704, 480));
	resize(mFrame, mFrame, Size(320, 240));
	//resize(mFrame, mFrame, Size(470, 320));
	//imageROI = mFrame(Rect(mFrame.cols*0.25, mFrame.rows*0.25, mFrame.cols*0.5, mFrame.rows*0.5));
	imageROI = mFrame(Rect(mFrame.cols*0.25, mFrame.rows * 0, mFrame.cols*0.5, mFrame.rows*1));
	cvtColor(imageROI, mGray, COLOR_BGR2GRAY);
	//equalizeHist(mGray, mGray);
	cout << mGray.size << endl;
	//cvtColor(imageROI, mGray, COLOR_YUV2GRAY_IYUV);
	//mGray = mGray*2.5; //experimental for brightening night videos
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
	//cars.detectMultiScale(mGray, cars_found, 1.05, 3, 0 | 2, Size(60, 60),Size(240,240));
	cars.detectMultiScale(mGray, cars_found, 1.05, 3, 0 | 2, Size(30, 30), Size(120, 120));
	//cars.detectMultiScale(mGray, cars_found, 1.05, 3, 0 | 2, Size(40, 40), Size(160, 160));
	cout << cars_found.size() << endl;
	if (cars_found.size() > 0)
	{
		isUsingDetector = false;

		if (biggest_bbox_approach == 1)
		{
			sort(cars_found.begin(), cars_found.end(), compare_rect);
			detected_vehicle = cars_found[0];
		}
		else
		{
			int index = get_index_bbox_centre(mGray, cars_found);
			detected_vehicle = cars_found[index];
		}

		//==================make new ROI for tracking car=========================================
		//if (detected_vehicle.y - 50 >= 0)
		//	y_carROI = detected_vehicle.y - 50;
		//int top_padding;
		//top_padding = 1.0*detected_vehicle.height;
		
		//if (detected_vehicle.y - top_padding >= 0)
		//	y_carROI = detected_vehicle.y - top_padding;
		//else
		//	y_carROI = 0;

		//mGray(Rect(0, y_carROI, mGray.cols, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);
		//mGray(Rect(detected_vehicle.x, y_carROI, detected_vehicle.width, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);
		mGray(Rect(detected_vehicle.x, detected_vehicle.y, detected_vehicle.width, detected_vehicle.height)).copyTo(carROI);
		//mGray.copyTo(carROI);
		
		cv::rectangle(mGray, detected_vehicle, Scalar(0, 0, 225));

		//========resize the car ROI to fixed size=======================================
		/*scale = double(carROI_fixedheight) / double(carROI.rows); //cout <<"scale:"<< scale << endl;
		resize(carROI, scaled_carROI, Size(carROI.cols*scale, carROI.rows*scale));
		scaled_detected_vehicle.x = detected_vehicle.x*scale;
		scaled_detected_vehicle.y = detected_vehicle.y*scale;
		scaled_detected_vehicle.width = detected_vehicle.width*scale;
		scaled_detected_vehicle.height = detected_vehicle.height*scale;
		scaled_y_carROI = y_carROI*scale;*/
		//===============================================================================

		
		
		#if debug
		std::ostringstream name;
		name << "firstframe.jpg";
		cv::imwrite(name.str(), mGray);
		#endif	

		Mat temp = carROI(Rect(carROI.cols*0.075, carROI.rows * 0.075, carROI.cols*0.85, carROI.rows*0.85));
		resize(temp,carROI,Size(52,52));

		v2.push_back(Rect(0,0, 52, 52));

		cout << "detector box: " << detected_vehicle << "   ";
		//v2.push_back(Rect(0,0, detected_vehicle.width, detected_vehicle.height));
		//v2.push_back(Rect(detected_vehicle.x, detected_vehicle.y, detected_vehicle.width, detected_vehicle.height));

		//v2.push_back(Rect(scaled_detected_vehicle.x, (scaled_detected_vehicle.y - scaled_y_carROI), scaled_detected_vehicle.width, scaled_detected_vehicle.height));
		//cout <<"carROI"<< carROI.size() << endl; cout << "v2[0]" << v2[0] << endl;
		tracker->init(carROI, v2[0]);
	}
}

void trackVehicle()
{
	//mGray(Rect(0, y_carROI, mGray.cols, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);
	//mGray(Rect(detected_vehicle.x, y_carROI, detected_vehicle.width, detected_vehicle.height + (detected_vehicle.y - y_carROI))).copyTo(carROI);
	mGray(Rect(detected_vehicle.x, detected_vehicle.y, detected_vehicle.width, detected_vehicle.height)).copyTo(carROI);
	Mat temp = carROI(Rect(carROI.cols*0.075, carROI.rows * 0.075, carROI.cols*0.85, carROI.rows*0.85));
	resize(temp,carROI,Size(52,52));

	//mGray.copyTo(carROI);
	cv::rectangle(mGray, detected_vehicle, Scalar(0, 0, 225)); //draw the detector box

	//scaling part==========
	//resize(carROI, scaled_carROI, Size(carROI.cols*scale, carROI.rows*scale));

	//=======================

																																					//v2.push_back(Rect(0,0,cars_found[0].width,cars_found[0].height));
																																					//tracker->init(carROI, v2[0]);
																																					//bool ok = tracker->update(mGray, v2[0]);

	bool ok = tracker->update(carROI, v2[0]);
	

	if (ok)
	{
		// Tracking success : Draw the tracked object
		cout << "tracker box: " << v2[0] << "   ";
		cout << "Y position tracker box: " << v2[0].y + v2[0].height << "   ";
		rectangle(carROI, v2[0], Scalar(255, 0, 0), 2, 1);
		//rectangle(mGray, Rect(v2[0].x, v2[0].y + y_carROI, v2[0].width, v2[0].height), Scalar(255, 0, 0), 2, 1);  //sementara
		
		#if debug 
		std::ostringstream name;
		name << "trackframe" << framecount << ".jpg";
		cv::imwrite(name.str(), carROI);
		#endif

		#if usedecision //=================================================================================================================
		if (isGetInitialWidth == true)
		{
			#if use_size 
			initialwidth = v2[0].width;
			thresholdwidth = initialwidth*(100 - thresholdpercentage) / 100;
			#else
			initial_y = v2[0].y + v2[0].height;
			threshold_y = initial_y*(100 - thresholdpercentage) / 100;
			#endif
			isGetInitialWidth = false;
		}
		//frame by frame checking whether tracked vehicle has moved away from the camera; indicated by bbox size reduced by thresholdvalue
		else
		{
			#if use_size
			if (v2[0].width <= thresholdwidth)
			{
				thresholdcount++;
				if (thresholdcount > numberofthresholdframes)
				{
					putText(mGray, "!!!Warning!!! ", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
					putText(carROI, "!!!Warning!!! ", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
					cout << "alarm starts" << endl;
				}
				else
				{
					cout << "front car starts/ passed threshold" << endl;
				}
			}
			#else
			//if ((v2[0].y + v2[0].height) <= threshold_y)
			if (((v2[0].y + v2[0].height) <= threshold_y) || (v2[0].width <= thresholdwidth))
			{
				thresholdcount++;
				threshold_y=threshold_y-1;
				thresholdwidth=thresholdwidth-1;
				if (thresholdcount > numberofthresholdframes)
				{
					putText(mGray, "!!!Warning!!! ", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
					putText(carROI, "!!!Warning!!! ", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
					cout << "alarm starts" << endl;
				}
				else
				{
					cout << "front car starts/ passed threshold" << endl;
				}
			}
			#endif

		}
		#endif//===========================================================================================================================


		//=======================================drawing plot==========================
		#if trackingplot
		#if use_csrt
		plotdata.push_back(v2[0].width);
		Ptr<plot::Plot2d> plot = plot::Plot2d::create(plotdata);
		plot->setPlotBackgroundColor(Scalar(50, 50, 50));
		plot->setPlotLineColor(Scalar(50, 50, 255));
		plot->setMinY(-1);
		plot->setMaxY(250);
		plot->setInvertOrientation(true);
		plot->render(plot_result); 
		imshow("Tracking plot (Vehicle Size)", plot_result);
		#else
		plotdata.push_back(v2[0].y+v2[0].height);
		Ptr<plot::Plot2d> plot = plot::Plot2d::create(plotdata);
		plot->setPlotBackgroundColor(Scalar(50, 50, 50));
		plot->setPlotLineColor(Scalar(50, 50, 255));
		plot->setMinY(50);
		plot->setMaxY(400);
		plot->setInvertOrientation(true);
		plot->render(plot_result);
		imshow("Tracking plot (Vehicle Y Position)", plot_result);
		#endif
		#endif
		//=============================================================================
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
	#if use_csrt
	TrackerCSRT::Params param;
	param.use_hog = true;
	param.use_channel_weights = false;
	param.use_segmentation = false;
	param.use_color_names = false;
	param.padding = 0.125f;
	param.hog_orientations = 3;
	param.num_hog_channels_used = 9;
	param.template_size = 100;
	param.admm_iterations = 1;
	param.histogram_bins = 1;
	param.scale_model_max_area = 1024.0f;
	tracker = TrackerCSRT::create(param);
	//tracker = TrackerCSRT::create();

	#else
	TrackerKCF::Params param;
	//param.desc_pca = TrackerKCF::GRAY | TrackerKCF::CN;
	//param.desc_npca = 0;
	param.compress_feature = true;
	param.compressed_size = 10;
	tracker = TrackerKCF::create(param); //no size reduction
	#endif
}

void EndFVSA()
{
	isStart = false;
	//isUsingDetector = false;
	//isGetInitialWidth = true;
	//thresholdcount = 0;
	if (v2.empty() == false)
		v2.pop_back();

	#if plotgraph
	plotdata.release();
	#endif
}

void saveimage()
{

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//============================================================================================================================

int main()
{
	cv::Mat bigframe = cv::Mat(1080, 1920, CV_8UC3);
	//std::cout << cv::getBuildInformation() << std::endl;
	//VideoWriter video("out.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(1920, 1080),1);
	
	//int framecount = 0;

	//freopen ("myfile.txt","a",stdout);
	
	loadDetector();
	getVideoData();

	#if autostart
	StartFVSA();
	#endif
	
	while (cap.read(mFrame))
	{	

		if (framecount % skipframe == 0)
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




			if (isUsingDetector == true)
			{
			#if displayall
				cv::imshow("detection", mGray);
			#endif			
			}
			else
			{
			#if displayall
				cv::imshow("detection", mGray);
				cv::imshow("tracking", carROI);
			#endif
			}


			auto end_time = std::chrono::high_resolution_clock::now();
			auto time = end_time - start_time;

			std::cout << "Time: " <<
				time / std::chrono::milliseconds(1) << "ms.       ";


			#if timeperformanceplot
			double a = time / std::chrono::milliseconds(1);
			plotdata1.push_back(a);
			Ptr<plot::Plot2d> plot1 = plot::Plot2d::create(plotdata1);
			plot1->setPlotBackgroundColor(Scalar(50, 50, 50));
			plot1->setPlotLineColor(Scalar(50, 50, 255));
			plot1->setMinY(-30);
			plot1->setMaxY(1200);
			plot1->setInvertOrientation(true);

			plot1->render(plot_result1);
			imshow("Time performance (milli seconds)", plot_result1);
			#endif

			//fullDraw(bigframe, mGray);
			//cvui::update();
			//video.write(bigframe);

			int c = waitKey(100);

			#if displayall
			if ((char)c == 27)
			{
				break;
			}
			else if ((char)c == 'q') //starts FVSA
			{
				if (isStart == false) // check if it is already starting or not
				{
					StartFVSA();
				}
			}
			else if ((char)c == 'w') //ends FVSA
			{
				if (isStart == true) // check if it is already starting or not
				{
					EndFVSA();
				}
			}
			#endif

			cout << "#frame: " << framecount << endl;

			

		}
		framecount++;
	}

	//fclose (stdout);

	if (v2.empty() == false)
		v2.pop_back();

	waitKey(0);
	//video.release();
	return 0;
}