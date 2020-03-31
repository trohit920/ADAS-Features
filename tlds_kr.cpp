
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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"  

#include <ctime>
#include <time.h>
#include <cstdio>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdlib>

extern "C" {
	#include <stdio.h>
}


using namespace std;
using namespace cv;


#define VIDEO_FILE_NAME "/home/rohit/adas/car_lane_sign_detection/New_sensor_data/new16.mp4"
#define TEMPLATE_PATH "/home/rohit/adas/TrafficLight-Detection/arrow/arrow_2.jpeg"
#define CASCADE_FILE_NAME "/home/rohit/adas/TrafficLight-Detection/cascade_87_24*12.xml"

#define SKIPFRAME 5
#define AUTOSTART 1
#define DISPLAYALL 1
#define HSV_VALUE 1
#define MEAN_VALUE 0
#define RED_COLOR 1
#define GREEN_COLOR 1
#define SHOWIMAGE 0
#define NUMOFTHRESHOLDFRAMES 1
#define DEBUG_FLAG 1
#define SAVE_DEBUG_TO_TEXT 0
#define DEBUG_GLOBAL_VAR 0
string template_path = "/home/rohit/adas/TrafficLight-Detection/arrow/arrow_2.jpeg";

int numberofthresholdframes;
int leftthresholdframes;

static unsigned long long get_localTimeStamp()
{
	time_t timestamp;
	unsigned long long secTime, msecTime;
	struct timeval tv;
	struct tm lt;

	gettimeofday(&tv, NULL);

	localtime_r(&tv.tv_sec, &lt);

	secTime = mktime(&lt) + lt.tm_gmtoff;

	msecTime = ((unsigned long long)secTime * (unsigned long long)1000) + ((unsigned long long)tv.tv_usec / (unsigned long long)1000);

	return msecTime;
}



//=======================================================variable declaration=================================================
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


CascadeClassifier traffic_light;
Mat leftSideROI, rightSideROI,  middleROI, middleROI1, middleROI2 ;
vector<Vec3f> red_circles, green_circles;
Rect detected_light;

Mat templ;
int framecount = 0;
int beepcount = 0;
int thresholdcount = 0;
int left_thresholdcount = 0;

bool isUsingDetector = true;
bool isStart = false;
bool isRedColor = false;
bool isGreenColor = false;
bool ifRedHappened = false;
bool isFourPattern = false;
bool isThreePattern = false;
bool isLightDetected = false;
bool arrowPresent = false;
int segmentationType = -1;
int red_fail_count = 0;
int detected_frames_count = 0;
int color_seg_fail_count = 0;

const int frame_width = 704;
const int frame_height = 480;
char* pYuvBuf;
char* pMemContent;
char timebuf[512];
char genbuf[512];
struct timeval tv;
struct tm stTime;
time_t lastTime = 0;
int len;

int value1 = 0, value2 = 0, value3 = 0, value4 = 0;

int load_dataset=0;

VideoCapture cap;

void getVideoData()
{
	cap.open(VIDEO_FILE_NAME);

}

void loadDetector()
{
	traffic_light.load(CASCADE_FILE_NAME);
}

Mat framePreprocessing(Mat& frame)
{
	Mat imageROI;

	//printf("7. frame preprocessing function  ++++#########\n");
	//printf(" frame size before framePreprocessing: %dX%d\n",frame.size().width, frame.size().height);
	resize(frame, frame, Size(704, 480));
	imageROI = frame(Rect(frame.cols*0.25, frame.rows*0.10, frame.cols*0.50, frame.rows*0.38));
	//printf("7. Size of the Image ROI: %dX%d\n\n",imageROI.size().width, imageROI.size().height);
	//imageROI.copyTo(frame);
	Mat frameGray;

	//namedWindow("image_ROI", WINDOW_NORMAL);
	//resizeWindow("image_ROI", Size(300, 200));
	//imshow("image_ROI", imageROI);
	printf("isStart______: %d\n", isStart);
	#if DEBUG_FLAG
		printf("1. Size of the Image ROI: %dX%d\n\n",imageROI.size().width, imageROI.size().height);
		// ostringstream name5;
		// name5 << "/app/sd/BlackVue/Record/imageROI_"<<framecount<<".jpg";
		// imwrite(name5.str(), imageROI);
	#endif

	#if HSV_VALUE
		GaussianBlur( imageROI, imageROI, Size(3, 3), 0 );
		return imageROI;

	#endif

	#if DEBUG_GLOBAL_VAR
		return NULL;
	#endif
}

int getAverage(Mat& frame)
{	
	double res;
	for (int i = 0; i < frame.rows; i++){
		for (int j = 0; j < frame.cols; j++){
			res += frame.at<uchar>(i, j);
			// printf("color, x, y: %d, %d, %d\n", color, j, i);
			}
		}
	int corow = frame.cols * frame.rows;  // image's full pixel number
	res /= corow;
	return res;

	
}

/*int segmentColor_night(Mat& imageROI)
{	
		printf("GOT HERE======================!!!!!!!!!!!!!!!!=========================\n");
	Mat red_image, green_image, lower_red_hue_range, upper_red_hue_range, saturated_red_image, saturated_green_image;
	Mat red_hue_image, green_hue_image, lower_green_hue_range, upper_green_hue_range;
	// For Red light: Threshold the HSV image, keep only the red pixels
	red_image = imageROI.clone();
	green_image = imageROI.clone();


	#if DEBUG_FLAG
		printf("6. Finding out which color is present.\n");
	#endif


	// Red color segmentation --------------------------------------------------------------------
	Mat leftRed, rightRed;

	cvtColor(red_image, red_image, CV_BGR2HSV);
	inRange(red_image, Scalar(0, 100, 100), Scalar(5, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(160, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	value1 = getAverage(red_hue_image);

	namedWindow("Red Combined threshold images", WINDOW_NORMAL);
	resizeWindow("Red Combined threshold images", 120, 90);
	imshow("Red Combined threshold images", red_hue_image);

	
	#if DEBUG_FLAG
		printf("Red image value: %d\n", value1);
	#endif
	// Red color segmentation --------------------------------------------------------------------

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	// Green color segmentation --------------------------------------------------------------------
	Mat leftGreen, rightGreen;

	cvtColor(green_image, green_image, CV_BGR2HSV);

	inRange(green_image, Scalar(50, 100, 90), Scalar(70, 255, 255), lower_green_hue_range);

	inRange(green_image, Scalar(70, 100, 90), Scalar(90, 255, 255), upper_green_hue_range);

	addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);
	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);

	
	value2 = getAverage(green_hue_image);

	namedWindow("Green Combined threshold images", WINDOW_NORMAL);
	resizeWindow("Green Combined threshold images", 120, 90);
	imshow("Green Combined threshold images", green_hue_image);
	#if DEBUG_FLAG
		printf("Green image value: %d \n", value2);
	#endif

	// Green color segmentation -------------------------------------------------------------------
	if ((value1 > 10) && (value2 > 10)){ // green and red circle
		return 1;
	}
	if ((value1 > 10) && (value2 < 10)){ // red cirle
		return 2;
	}
	if ((value2 > 10) && (value1 < 10)){ // green circle
		return 0;
	}
	if ((value1 == 0 && value2 == 0)){ // none
		return -1;
	}
	return -1;

}*/

vector<int> segmentColor_night(Mat& imageROI) // , bool contour)
{	

	Mat red_image, green_image, lower_red_hue_range, upper_red_hue_range, saturated_red_image, saturated_green_image;
	Mat red_hue_image, green_hue_image, lower_green_hue_range, upper_green_hue_range;
	// For Red light: Threshold the HSV image, keep only the red pixels
	red_image = imageROI.clone();
	green_image = imageROI.clone();
	vector<int> res;

	#if DEBUG_FLAG
		printf("6. Finding out which color is present.\n");
	#endif

	// Red color segmentation ----------------------------------------------------------------------
	Mat leftRed, rightRed;

	cvtColor(red_image, red_image, COLOR_BGR2HSV);
	inRange(red_image, Scalar(0, 100, 100), Scalar(8, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(150, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	value1 = getAverage(red_hue_image);
	
	imshow("Red Combined threshold images", red_hue_image);
	#if DEBUG_FLAG
		printf("Red image value: %d\n", value1);
	#endif


	// Red color segmentation ----------------------------------------------------------------------

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	// Green color segmentation --------------------------------------------------------------------
	Mat leftGreen, rightGreen;

	cvtColor(green_image, green_image, COLOR_BGR2HSV);

	inRange(green_image, Scalar(45, 75, 90), Scalar(90, 255, 255), green_hue_image);
	// inRange(green_image, Scalar(50, 100, 90), Scalar(70, 255, 255), lower_green_hue_range);

	// inRange(green_image, Scalar(70, 100, 90), Scalar(95, 255, 255), upper_green_hue_range);

	// addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);
	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);

	value2 = getAverage(green_hue_image);

	imshow("Green Combined threshold images", green_hue_image);
	#if DEBUG_FLAG
		printf("Green image value: %d \n", value2);
	#endif
	// Green color segmentation --------------------------------------------------------------------
	
	res.push_back(value1);
	res.push_back(value2);
	return res;
}

int segmentDecision_night(vector<int> &top, vector<int> &mid_top, vector<int> &mid, vector<int> &mid_bot, vector<int> &bot)
{
	int red_sum = 0, green_sum = 0;
	int red_arr[4] = {}, green_arr[4] = {};
	
	red_arr[0] = top[0];
	green_arr[0] = top[1];
	red_sum += top[0];
	green_sum += top[1];
	#if DEBUG_FLAG
		printf("Red vs green on TOP: %dx%d\n", red_arr[0], green_arr[0]);
	#endif

	red_arr[1] = mid_top[0];
	green_arr[1] = mid_top[1];
	red_sum += mid_top[0];
	green_sum += mid_top[1];
	#if DEBUG_FLAG
		printf("Red vs green on MID_TOP: %dx%d\n", red_arr[1], green_arr[1]);
	#endif
	
	red_arr[2] = mid[0];
	green_arr[2] = mid[1];
	red_sum += mid[0];
	green_sum += mid[1];
	#if DEBUG_FLAG
		printf("Red vs green on MID: %dx%d\n", red_arr[2], green_arr[2]);
	#endif
	
	red_arr[3] = mid_bot[0];
	green_arr[3] = mid_bot[1];
	red_sum += mid_bot[0];
	green_sum += mid_bot[1];
	#if DEBUG_FLAG
		printf("Red vs green on MID_BOT: %dx%d\n", red_arr[3], green_arr[3]);
	#endif
	
	red_arr[4] = bot[0];
	green_arr[4] = bot[1];
	red_sum += bot[0];
	green_sum += bot[1];
	#if DEBUG_FLAG
		printf("Red vs green on BOT: %dx%d\n\n", red_arr[4], green_arr[4]);
	#endif
	
	#if DEBUG_FLAG
		printf("Red vs green total: %d   x   %d\n\n", red_sum, green_sum);
	#endif

	if ((red_sum > 10) && (green_sum > 10)){ // green and red circle
		return 1;
	}
	if ((red_sum > 10) && (green_sum < 10)){ // red cirle
		return 2;
	}
	if ((green_sum > 10) && (red_sum < 10)){ // green circle
		return 0;
	}
	if ((red_sum == 0 && green_sum == 0)){ // none
		return -1;
	}
	return -1;
}

int colorDecision_night(int segmentationType)
{
	if (segmentationType == 1){
		#if DEBUG_FLAG
			printf("7. Segmented light area is on both sides: Turn left and RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
	}

	else if (segmentationType == 2)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is on leftside : RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
		return 0;
	}
	else if (segmentationType == 0)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is rightside : GREEN LIGHT. \n\n");
		#endif
		isGreenColor = true;
		isRedColor = false;
	}
	else
	{
		#if DEBUG_FLAG
			printf("7. Color segmentation failed !!!!!\n\n");
		#endif
		return -1;
	}

	if (isGreenColor == true && isRedColor == false && ifRedHappened == true) 
	{
		thresholdcount++;

		if (thresholdcount > numberofthresholdframes)
		{
			#if DEBUG_FLAG
				printf("8. Light change occured and the alarm started.\n\n");
			#endif
			return 1;
		}
	}
	return 0;
}

int segmentColor(Mat& lightROI)
{	
	Mat red_image, green_image, lower_red_hue_range, upper_red_hue_range, saturated_red_image, saturated_green_image;
	Mat red_hue_image, green_hue_image, lower_green_hue_range, upper_green_hue_range;
	// For Red light: Threshold the HSV image, keep only the red pixels
	red_image = lightROI.clone();
	green_image = lightROI.clone();

	vector<Mat> channel;

		
	#if DEBUG_FLAG
		printf("12. Finding out which color is present.\n");
	#endif
		
	// Red color segmentation --------------------------------------------------------------------
	Mat leftRed, rightRed, topRed, bottomRed;
	cvtColor(red_image, red_image, CV_BGR2HSV);
	inRange(red_image, Scalar(0, 100, 100), Scalar(15, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(150, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	Rect leftRedPart(0, 0, red_hue_image.cols / 2, red_hue_image.rows );
	red_hue_image(Rect(leftRedPart.x, leftRedPart.y, leftRedPart.width, leftRedPart.height)).copyTo(leftRed);
	imshow("leftRed", leftRed);

	// Rect topRedPart(0, 0, red_hue_image.cols , red_hue_image.rows / 2 );
	// red_hue_image(Rect(topRedPart.x, topRedPart.y, topRedPart.width, topRedPart.height)).copyTo(topRed);
	// imshow("topRed", topRed);

	inRange(red_image, Scalar(0, 100, 100), Scalar(15, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(150, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);		

	Rect rightRedPart(red_hue_image.cols / 2, 0, red_hue_image.cols - red_hue_image.cols / 2, red_hue_image.rows );
	red_hue_image(Rect(rightRedPart.x, rightRedPart.y, rightRedPart.width, rightRedPart.height)).copyTo(rightRed);

	// Rect bottomRedPart(0, red_hue_image.rows / 2, red_hue_image.cols, red_hue_image.rows / 2 );
	// red_hue_image(Rect(bottomRedPart.x, bottomRedPart.y, bottomRedPart.width, bottomRedPart.height)).copyTo(bottomRed);

	GaussianBlur(rightRed, rightRed, Size(3, 3), 0 );
	value1 = getAverage(leftRed);
	value2 = getAverage(rightRed);
	leftRed.release();
	rightRed.release();

	// GaussianBlur(bottomRed, bottomRed, Size(3, 3), 0 );
	// value1 = getAverage(topRed);
	// value2 = getAverage(bottomRed);
	// topRed.release();
	// bottomRed.release();


	if (value1 < 6 && red_fail_count < 3)
	{
		red_fail_count++;
	}
	else if ((value1 < 6 && red_fail_count == 3 && isGreenColor == false) || segmentationType == 1)
	{
		red_image = lightROI.clone();
		cvtColor(red_image, saturated_red_image, CV_BGR2HSV);
		split(saturated_red_image, channel);
		channel[1] *= 1.2;
		merge(channel, saturated_red_image);
		cvtColor(red_image, red_image, CV_BGR2HSV);
		#if DEBUG_FLAG
			printf("=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X===UPDATED COLOR RANGE\n");
		#endif
		inRange(red_image, Scalar(0, 100, 100), Scalar(15, 255, 255), lower_red_hue_range);

		inRange(red_image, Scalar(150, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

		addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);		

		Rect rightRedPart(red_hue_image.cols / 2, 0, red_hue_image.cols - red_hue_image.cols / 2, red_hue_image.rows );
		red_hue_image(Rect(rightRedPart.x, rightRedPart.y, rightRedPart.width, rightRedPart.height)).copyTo(rightRed);

		// Rect bottomRedPart(0, red_hue_image.rows / 2, red_hue_image.cols, red_hue_image.rows / 2 );
		// red_hue_image(Rect(bottomRedPart.x, bottomRedPart.y, bottomRedPart.width, bottomRedPart.height)).copyTo(bottomRed);

		// inRange(saturated_red_image, Scalar(0, 87, 100), Scalar(25, 255, 255), lower_red_hue_range);

		// inRange(saturated_red_image, Scalar(155, 87, 100), Scalar(180, 255, 255), upper_red_hue_range);

		// addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		// GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

		// Rect leftRedPart(0, 0, red_hue_image.cols / 2, red_hue_image.rows );
		// red_hue_image(Rect(leftRedPart.x, leftRedPart.y, leftRedPart.width, leftRedPart.height)).copyTo(leftRed);

		GaussianBlur(rightRed, rightRed, Size(3, 3), 0 );
		value1 = getAverage(leftRed);
		value2 = getAverage(rightRed);
		leftRed.release();
		rightRed.release();

		// value1 = getAverage(topRed);
		// value2 = getAverage(bottomRed);
		// topRed.release();
		// bottomRed.release();

	}
	else if (red_fail_count > 0)
	{
		red_fail_count--;
	}
	#if DEBUG_FLAG
		printf("	Red image left vs right: %d vs %d\n", value1, value2);
	#endif
	// Red color segmentation --------------------------------------------------------------------

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	// Green color segmentation --------------------------------------------------------------------
	
	cvtColor(green_image, saturated_green_image, CV_BGR2HSV);
	split(saturated_green_image, channel);
	channel[1] *= 1.2;
	merge(channel, saturated_green_image);

	Mat leftGreen, rightGreen, topGreen, bottomGreen;
	inRange(saturated_green_image, Scalar(45, 70, 100), Scalar(70, 255, 255), lower_green_hue_range);

	inRange(saturated_green_image, Scalar(70, 70, 100), Scalar(90, 255, 255), upper_green_hue_range);

	addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);
	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);
	
	Rect rightGreenPart(green_hue_image.cols / 2, 0, green_hue_image.cols - green_hue_image.cols / 2, red_hue_image.rows );
	green_hue_image(Rect(rightGreenPart.x, rightGreenPart.y, rightGreenPart.width, rightGreenPart.height)).copyTo(rightGreen);
	imshow("rightGreen", rightGreen);

	// Rect bottomGreenPart(0, green_hue_image.rows / 2, green_hue_image.cols , red_hue_image.rows / 2 );
	// green_hue_image(Rect(bottomGreenPart.x, bottomGreenPart.y, bottomGreenPart.width, bottomGreenPart.height)).copyTo(bottomGreen);
	// imshow("bottomGreen", bottomGreen);

	cvtColor(green_image, green_image, CV_BGR2HSV);
	inRange(green_image, Scalar(45, 100, 100), Scalar(70, 255, 255), lower_green_hue_range);

	inRange(green_image, Scalar(70, 100, 100), Scalar(90, 255, 255), upper_green_hue_range);

	addWeighted(lower_green_hue_range, 1.0, upper_green_hue_range, 1.0, 0.0, green_hue_image);
	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);

	Rect leftGreenPart(0, 0, green_hue_image.cols / 2, green_hue_image.rows );
	green_hue_image(Rect(leftGreenPart.x, leftGreenPart.y, leftGreenPart.width, leftGreenPart.height)).copyTo(leftGreen);

	// Rect topGreenPart(0, 0, green_hue_image.cols , green_hue_image.rows / 2 );
	// green_hue_image(Rect(topGreenPart.x, topGreenPart.y, topGreenPart.width, topGreenPart.height)).copyTo(topGreen);

	GaussianBlur(leftGreen, leftGreen, Size(3, 3), 0, 0);
	value3 = getAverage(leftGreen);
	value4 = getAverage(rightGreen);
	leftGreen.release();
	rightGreen.release();

	// GaussianBlur(topGreen, topGreen, Size(3, 3), 0, 0);
	// value3 = getAverage(topGreen);
	// value4 = getAverage(bottomGreen);
	// topGreen.release();
	// bottomGreen.release();

	#if DEBUG_FLAG
		printf("	Green image left vs right: %d vs %d\n", value3, value4);
	#endif
	// Green color segmentation --------------------------------------------------------------------

	if ((value1 > value2) && (value3 < value4) && (value1 >= 6) && (value4 >= 6) && (value2 <= 6) && (value3 <= 6)){ // green and red circle
		return 1;
	}
	if ((value1 > value2) && (value1 >= 6) && (value2 <= 6)){ // red cirle
		return 2;
	}
	if ((value3 < value4) && (value4 >= 6) && (value3 <= 6)){ // green circle
		return 0;
	}
	if (((value1 == 0 && value2 == 0) && (value3 == 0 && value4 == 0)) || ((value1 < value2) || (value3 > value4))){ // none
		return -1;
	}
	return -1;
}

bool isArrow(Mat& lightROI, string template_path, double minMatchQuality){
	Mat result;
	Mat img = lightROI;
	templ = imread(TEMPLATE_PATH);
	resize(templ, templ, Size(7, 7));
	// resize(lightROI, lightROI, Size(704, 480));

	/// Create windows
	//namedWindow("image_window", WINDOW_NORMAL);


	Mat img_display;
	img.copyTo(img_display);

	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching and Normalize
	matchTemplate(img, templ, result, 5);


	/// Create Trackbar
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	
    matchLoc = maxLoc;

	printf("maxVal: %f\n", maxVal);
    printf("minVal: %f\n", minVal);

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  	if (maxVal > minMatchQuality){

		Mat rect;
	    /// Show me what you got
	    rectangle(img, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0 );
	    
	    
	    img_display(Rect(matchLoc.x, matchLoc.y, templ.cols, templ.rows)).copyTo(rect);

	    //namedWindow("detected", WINDOW_NORMAL);
	    //resizeWindow("detected", Size(500, 500));
	    //imshow("detected", rect);

	    //imshow("image_window", img_display );

	    waitKey(1);
	    printf("detected the arrow\n");
	    return true;
  	}
  	else{
	    // rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
	    // rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
	  	// resizeWindow("image_window", 120, 90);
	  	//imshow("image_window", img_display );
	    //imshow( "result_window", result );
	    //MatchingMethod( 0, 0 );

	    waitKey(1);
	    printf("arrow detection failed!!!!!!\n");
	  	}
  return false;
}


bool compare_rect(const cv::Rect & a, const cv::Rect &b)
{
	return a.width > b.width;
}

int colorDecision(int segmentationType, bool fourPattern)
{
	if (segmentationType == 1){
		#if DEBUG_FLAG
			printf("12. Segmented light area is on both sides: Turn left and RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
	}
	else if (!fourPattern) 
	{	
		arrowPresent = false;
		if(segmentationType == 2)
		{	
			#if DEBUG_FLAG
				printf("12. Segmented light area is on leftside : RED LIGHT. \n\n");
			#endif
			isRedColor = true;
			ifRedHappened = true;
		}	
		else if (segmentationType == 0)
		{
			#if DEBUG_FLAG
				printf("12. Segmented light area is rightside : GREEN LIGHT\n\n");
			#endif
			isGreenColor = true;
			isRedColor = false;
		}
		else
		{
			#if DEBUG_FLAG
				printf("12. ThreePattern: Color segmentation failed !!!!!\n\n");
			#endif
				return -1;
		}

		if (isGreenColor == true && isRedColor == false && ifRedHappened == true) 
		{
			thresholdcount++;

			if (thresholdcount > numberofthresholdframes)
			{
				#if DEBUG_FLAG
					printf("13. Light change occured and the alarm started.\n\n");
				#endif
				return 1;
			}
		}
	}

	else if (fourPattern == true)
	{	
		if(segmentationType == 2)
		{	
			#if DEBUG_FLAG
				printf("12. Segmented light area is on leftside and arrow is present : RED LIGHT AND ARROW. \n\n");
			#endif
			isRedColor = true;
			ifRedHappened = true;
			left_thresholdcount++;
			if (left_thresholdcount > leftthresholdframes){
				#if DEBUG_FLAG
					printf("13. Left arrow number of frames passed a threshold.\n\n");
				#endif
				arrowPresent = true;
			}
		}	
		else if (segmentationType == 0)
		{
			#if DEBUG_FLAG
				printf("12. Segmented light area is rightside and arrow is present: GREEN LIGHT AND ARROW.\n\n");
			#endif
			isGreenColor = true;
			isRedColor = false;
			left_thresholdcount++;
			if (left_thresholdcount > leftthresholdframes){
				#if DEBUG_FLAG
					printf("13. Left arrow number of frames passed a threshold.\n\n");
				#endif
				arrowPresent = true;
			}
		}
		else
		{
			#if DEBUG_FLAG
				printf("12. FourPattern: Color segmentation failed !!!!!\n\n");
			#endif
				return -1;
		}
		

		if (isGreenColor == true && isRedColor == false && ifRedHappened == true) 
		{
			thresholdcount++;

			if (thresholdcount > numberofthresholdframes)
			{
				#if DEBUG_FLAG
					printf("13. Light change occured and the alarm started.\n\n");
				#endif
				return 1;
			}
		}
	}
	else
	{
		#if DEBUG_FLAG
			printf("20. Failed to go somewhere in color decision function!!!!!!!!!!!\n\n");
		#endif
	}
	return 0; 
}


Mat detect_TLDS_light(Mat& imageROI)
{

	// namedWindow("imageROI", WINDOW_NORMAL);
	// resizeWindow("imageROI", 120, 90);
	// imshow("imageROI", imageROI);

	Mat lightROI;
	vector<Rect> light;

	#if DEBUG_FLAG
		printf("2. Detecting the traffic light.\n\n");
	#endif
	traffic_light.detectMultiScale( imageROI, light, 1.05, 3, 0|2  );
	
	#if DEBUG_FLAG
		printf("3. Number of traffic light signs: %zu\n", light.size());
	#endif

	if(light.size() > 0)
	{

		sort(light.begin(), light.end(), compare_rect);
		detected_light = light[0];
		imageROI(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI);
		
		#if DEBUG_FLAG
			printf("	Size of the detected traffic light: %dX%d\n", detected_light.size().width, detected_light.size().height);
			//rectangle(lightROI, detected_light, Scalar(0, 0, 255));
			ostringstream name4;
			name4 << "/app/sd/BlackVue/Record/detected_light_"<<framecount<<".jpg";	
			imwrite(name4.str(), lightROI);

		#endif
		isLightDetected = true;
	}
	else
	{
		isLightDetected = false;
	}
	if (light.size() > 0){
		namedWindow("lightROI", WINDOW_NORMAL);
		resizeWindow("lightROI", 120, 90);
		imshow("lightROI", lightROI);
	}
	
	return lightROI;
}

void StartTLDS()
{
	isStart = true;
	isUsingDetector = true;
	thresholdcount = 0;
	left_thresholdcount = 0;
	red_fail_count = 0;
	detected_frames_count = 0;
	color_seg_fail_count = 0;
}


void EndTLDS()
{
	isStart = false;
	red_fail_count = 0;
	detected_frames_count = 0;
	color_seg_fail_count = 0;
}



int main()
{
	#if DEBUG_FLAG
		printf("2. Inside the tlds function.\n\n");
	#endif
	#if SAVE_DEBUG_TO_TEXT
		 freopen("/app/sd/Blackvue/Record/debug_tlds.txt","a",stdout);
	#endif
	int gps_speed = 0;
	int max_speed = 0;
	if (max_speed >= 0) //tlds only works when speed = 0
	{
		if ((gps_speed==0) && (gps_speed<=max_speed)) // tlds starts when car stops, speed = 0
		{
			#if DEBUG_FLAG
				printf("3. GPS SPEED and MAX SPEED satisfy the conditions.\n\n");
				printf("TLDS starts.\n\n");
			#endif

			if (isStart == false)
			{
				StartTLDS();
				numberofthresholdframes=2;
				leftthresholdframes=2;
			}
		}
		else
		{
			#if DEBUG_FLAG
				printf("TLDS stops.\n\n");
			#endif

			if(isStart == true)
			{
				EndTLDS();
				numberofthresholdframes=2;
			}
		}
	}
	Mat imageROI, lightROI;
	int flag_alarm = 0;

	loadDetector();
	#if DEBUG_FLAG
		printf("Finish loading dataset. \n");
	#endif

	getVideoData();

	#if AUTOSTART
	StartTLDS();
	#endif
	bool fourPattern;
	Mat frame;
	vector<int> whole_image, top, middle_top, middle, middle_bot, bot;
	while (cap.read(frame))
	{	
		//Mat frame_template = frame.clone();
		//resize(frame_template, frame_template, Size(704, 408));
		
		if (framecount % SKIPFRAME == 0)
		{
			auto start_time = std::chrono::high_resolution_clock::now();

			if (isStart == true)
			{	
				
				Mat frame_template = frame.clone();
				resize(frame_template, frame_template, Size(704, 480));

				imageROI = frame_template(Rect(frame_template.cols*0.25, frame_template.rows*0.10, frame_template.cols*0.50, frame_template.rows*0.38));
				imshow("image_ROI", imageROI);
				
				#if DISPLAYALL
				namedWindow("detection", WINDOW_NORMAL);
				resizeWindow("detection", 500, 500);
				imshow("detection", frame);
				#endif

				// int v = 0;
				// Mat hsv, valueImage ;
				// vector<Mat> channels;
				// cvtColor(imageROI, valueImage, CV_BGR2HSV);
				// split(valueImage, channels);
				// hsv = channels[2];
				// v = getAverage(hsv);

				// Mat black_hue;
				// inRange(frame_hsv, Scalar(0, 0, 0), Scalar(180, 255, 40), black_hue);

				// GaussianBlur(black_hue, black_hue, Size(5, 5), 0, 0);
				// int black = getAverage(black_hue);

				Mat frame_hsv;
				cvtColor(imageROI, frame_hsv, COLOR_BGR2HSV);

				vector<Mat> channel;
				split(frame_hsv, channel);
				int value = getAverage(channel[2]);
				channel.clear();

				Mat black_hue;
				inRange(frame_hsv, Scalar(0, 0, 0), Scalar(180, 255, 40), black_hue);

				GaussianBlur(black_hue, black_hue, Size(5, 5), 0, 0);
				int black = getAverage(black_hue);
				cout << " value v: " << value << "	black b: "<<  black << endl;
				//printf("VALUE, black: %d, %d\n", value, black);
				
				//if (isUsingDetector = true)
				{
					//if(value < 100)
					if ((value < 130 && black > 25) || (value < 90))
					{
						/*//segmentationType = segmentColor(lightROI);
						segmentationType = segmentColor_night(imageROI);
						printf("segmentationType: %d\n", segmentationType);
						//flag_alarm = colorDecision(lightROI, segmentationType, fourPattern);
						flag_alarm = colorDecision(segmentationType, fourPattern);*/
					
						#if DEBUG_FLAG
							printf("	after detection!\n");
						#endif

						imageROI = frame_template(Rect(frame_template.cols*0.30, frame_template.rows*0.0, frame_template.cols*0.35, frame_template.rows*0.40));
						whole_image = segmentColor_night(imageROI);

						imshow(" night ROI ", imageROI);
						// Getting color values from 5 stripes of the imageROI
						// =========================================================================================
						Mat imageROI_top = frame(Rect(frame.cols*0.35, frame.rows*0.00, frame.cols*0.30, frame.rows*0.09));
						top = segmentColor_night(imageROI_top);

						Mat imageROI_middle_top = frame(Rect(frame.cols*0.35, frame.rows*0.09, frame.cols*0.30, frame.rows*0.09));
						middle_top = segmentColor_night(imageROI_middle_top);

						Mat imageROI_middle = frame(Rect(frame.cols*0.35, frame.rows*0.18, frame.cols*0.30, frame.rows*0.08));
						middle = segmentColor_night(imageROI_middle);

						Mat imageROI_middle_bot = frame(Rect(frame.cols*0.35, frame.rows*0.26, frame.cols*0.30, frame.rows*0.08));
						middle_bot = segmentColor_night(imageROI_middle_bot);

						Mat imageROI_bot = frame(Rect(frame.cols*0.35, frame.rows*0.34, frame.cols*0.30, frame.rows*0.06));
						bot = segmentColor_night(imageROI_bot);
						segmentationType = segmentDecision_night(top, middle_top, middle, middle_bot, bot);
						
						// ==========================================================================================

						segmentationType = segmentDecision_night(top, middle_top, middle, middle_bot, bot);

						#if DEBUG_FLAG
							printf("segmentationType: %d\n", segmentationType);
						#endif
						flag_alarm = colorDecision_night(segmentationType);

						whole_image.clear();
						top.clear();
						middle_top.clear();
						middle.clear();
						middle_bot.clear();
						bot.clear();
					}
					else 
					{
						//if (isUsingDetector = true)
						{	
							if (detected_frames_count >= 3 || (ifRedHappened == true && isGreenColor == true && value4 != 0))
							{
								imageROI = frame_template(Rect(frame_template.cols*0.25, frame_template.rows*0.10, frame_template.cols*0.50, frame_template.rows*0.38));
								imageROI(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI);
								lightROI(Rect(lightROI.cols * 0.05, lightROI.rows * 0.14, lightROI.cols * 0.90, lightROI.rows * 0.72)).copyTo(lightROI);
								imshow("fixed traffic light", lightROI);
								fourPattern = isArrow(lightROI, template_path, 0.6);

								if (!fourPattern)
								{
									#if DEBUG_FLAG
										printf("It is a three pattern\n");
									#endif
								}
								else
								{
									#if DEBUG_FLAG
										printf("You can turn left and It is a four pattern\n");
									#endif
								}
								
								segmentationType = segmentColor(lightROI);

								if ((segmentationType == -1) && (detected_frames_count > 0) && (color_seg_fail_count < 10))
								{
									color_seg_fail_count++;
								}
								else if ((segmentationType == -1) && (color_seg_fail_count == 10) && (value1 < 4) && (detected_frames_count > 0))
								{
									detected_frames_count--;
									color_seg_fail_count = 0;
									framecount++;
									continue;
								}

								flag_alarm = colorDecision(segmentationType, fourPattern);
							}

							else
							{
								
								if (isUsingDetector = true)
								{	
									Mat lightROI_template;
									//Mat frame_template = frame.clone();

									imageROI = framePreprocessing(frame);

									lightROI = detect_TLDS_light(imageROI);
									
									if (isLightDetected == true)
									{
										// //lightROI_template = frame(Rect(lightROI.x - 20, ))
										// Mat imageROI_template = frame_template(Rect(frame.cols*0.30, frame.rows*0.01, frame.cols*0.29, frame.rows*0.50));
										// fourPattern = isArrow(imageROI_template, 0.68);
										Mat imageROI_template = frame_template(Rect(frame_template.cols*0.25, frame_template.rows*0.10, frame_template.cols*0.50, frame_template.rows*0.38));
										imageROI_template(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI_template);
																				
										lightROI_template(Rect(lightROI_template.cols * 0.05, lightROI_template.rows * 0.14, lightROI_template.cols * 0.90, lightROI_template.rows * 0.72)).copyTo(lightROI_template);
										fourPattern = isArrow(lightROI_template, template_path, 0.6);
									}
									else
									{
										#if DEBUG_FLAG
											printf("	NO TRAFFIC LIGHT DETECTED!!!\n\n\n\n");
										#endif
										framecount++;
										waitKey(1);
										continue;
									}

																											
									if (!fourPattern)
									{
										#if DEBUG_FLAG
											printf("It is a three pattern\n");
										#endif
									}
									else
									{
										#if DEBUG_FLAG
											printf("You can turn left and \n");
											printf("It is a four pattern\n");
										#endif
									}

																		
									segmentationType = segmentColor(lightROI_template);
									//segmentationType = segmentColor(imageROI);
									printf("segmentationType: %d\n", segmentationType);
									if ((segmentationType == 2 || segmentationType == 1) && (detected_frames_count < 3))
									{
										detected_frames_count++;
									}
									if ((segmentationType == -1) && (detected_frames_count > 0) && (value1 == 0) && (ifRedHappened == false))
									{
										detected_frames_count--;
									}
									flag_alarm = colorDecision(segmentationType, fourPattern);
								}
							}
						}
					}

				}
			}
			
			// #if DISPLAYALL
			// 	namedWindow("detection", WINDOW_NORMAL);
			// 	resizeWindow("detection", 500, 500);
			// 	imshow("detection", frame);
			// #endif
			auto end_time = std::chrono::high_resolution_clock::now();
			auto time = end_time - start_time;

			std::cout << "Time: " <<
				time / std::chrono::milliseconds(1) << "ms.       "<< endl;
			
			int c = waitKey(100);

			#if DISPLAYALL
				if ((char)c == 27) 
				{ 
					break; 
				}
				else if ((char)c == 'q') 
				{
					if (isStart==false) 
					{
						StartTLDS();
					}
				}
				else if ((char)c == 'w') 
				{
					if (isStart==true)
					{
						EndTLDS();
					}
				}
			#endif

			printf("#frame: %d\n", framecount);
			#if DEBUG_FLAG
				printf("flag_alarm: %d  ", flag_alarm);
				printf("========================================================================================\n\n\n\n");
			#endif
			flag_alarm=0;
		}
		framecount++;

		
	}
	waitKey(0);
	return 0;
}


	