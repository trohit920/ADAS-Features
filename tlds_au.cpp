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


#define VIDEO_FILE_NAME "/home/rohit/adas/car_lane_sign_detection/au_tlds/au_tlds32.mp4"
// #define VIDEO_FILE_NAME "/media/richard/BLACKVUE/20191121_153145_EF.mp4"
// #define VIDEO_FILE_NAME "/home/richard/Documents/TLDS_videos/final_test/tlds52.mp4"
// #define VIDEO_FILE_NAME "/home/richard/Documents/TLDS_videos/confusing_videos/conf9.mp4"
#define TEMPLATE_PATH "/home/rohit/adas/TrafficLight-Detection/arrow/arrow_2.jpeg"
// #define CASCADE_FILE_NAME "/home/rohit/adas/TrafficLight-Detection/cascade_88_12_24.xml"
#define CASCADE_FILE_NAME "/home/rohit/adas/TrafficLight-Detection/cascade_90_12_24_extended.xml"
#define SKIPFRAME 5
// #define DISPLAYALL 0
#define SHOWIMAGE 0
#define NUMOFTHRESHOLDFRAMES 5
#define DEBUG_FLAG 1
#define SAVE_DEBUG_TO_TEXT 0
#define DEBUG_GLOBAL_VAR 0
#define TLDS 1
#define SAVE_VIDEO 0

#define GREEN 0
#define RED 2
#define GREEN_LEFT_RED_ARROW 0
#define GREEN_RIGHT_RED_ARROW 0
#define RED_RIGHT_GREEN_ARROW 1
#define RED_LEFT_GREEN_ARROW 1
#define RED_AND_GREEN 1
#define SEG_FAIL -1

//=======================================================variable declaration=================================================
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int value1, value2, value3, value4;
int value5, value6, value7, value8;
int yel_val1 = 0, yel_val2 = 0, val3_sat = -20, val4_sat = -20;
int numberofthresholdframes;
int leftthresholdframes;

CascadeClassifier traffic_light;
Mat leftSideROI, rightSideROI,  middleROI, middleROI1, middleROI2 ;
vector<Vec3f> red_circles, green_circles;
Rect detected_light;

Mat templ;
int framecount = 0;
int beepcount = 0;
int thresholdcount = 0;
int left_thresholdcount = 0;
int detected_frames_count = 0;
int color_seg_fail_count = 0;
int segmentationType = -1;
int red_fail_count = 0;
int flag_alarm = 0;

bool isUsingDetector = true;
bool isStart = false;
bool isRedColor = false;
bool isGreenColor = false;
bool ifRedHappened = false;
bool ifYelHappened = false;
bool isFourPattern = false;
bool isThreePattern = false;
bool isLightDetected = false;
bool arrowPresent = false;
bool startRecording = false;
bool isLeft = false;
bool isRight = false;
bool onSides = false;

// =======middle width is 0.5
// float top_edge = 0.06;
// float ROI_height = 0.33;
// float mid_width = 0.5;
// float left_mid = 0.25;
// float side_width = 0.2;
// float left_edge = 0.05;
// float right_edge = 0.75;

// =======middle width is 0.4
// float top_edge = 0.05;
// float ROI_height = 0.33;
// float mid_width = 0.4;
// float left_mid = 0.3;
// float side_width = 0.25;
// float left_edge = 0.05;
// float right_edge = 0.7;

// =======middle width is 0.45 and middle and sides are occluded
float top_edge = 0.15;
float ROI_height = 0.33;
float mid_width = 0.45;
float left_mid = 0.275;
float side_width = 0.25;
float left_edge = 0.075;
float right_edge = 0.675;

// =======night ROI
float night_top = 0.05;
float night_top_height = 0.13;
float night_mid_top_height = 0.12;
float night_mid_height = 0.10;
float night_mid_bot_height = 0.06;
float night_bot_height = 0.04;
float night_width = 0.60;
float night_left = 0.20;
// float night_ROI_height = night_top_height + night_mid_top_height + night_mid_height + night_mid_bot_height + night_bot_height;


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
int u0_dist = 540, v0_dist = 960;


int load_dataset=0;

Mat cam_mat, distCoeffs;

VideoCapture cap;
#if SAVE_VIDEO
	VideoWriter video("outcpp.mp4", VideoWriter::fourcc('F','M','P','4'), 60, Size(1920, 1080));
#endif
Mat res_image(1080, 1920, CV_8UC3, (Scalar(0, 0, 0)));
Mat alarm_stop(181, 300, CV_8UC3, (Scalar(0, 0, 0)));

void getVideoData()
{
	cap.open(VIDEO_FILE_NAME);
}

void loadDetector()
{
	traffic_light.load(CASCADE_FILE_NAME);
}

int getAverage(Mat& frame)
{	
	double res;
	for (int i = 0; i < frame.rows; i++)
	{
		for (int j = 0; j < frame.cols; j++)
		{
			res += frame.at<uchar>(i, j);
		}
	}
	int corow = frame.cols * frame.rows;  // image's full pixel number
	res /= corow;
	return res;
}

bool compare_rect(const cv::Rect & a, const cv::Rect &b)
{
	return a.width > b.width;
}

vector<Mat> framePreprocessing(Mat& input_frame)
{
	vector<Mat> res;
	Mat frame;
	resize(input_frame, frame, Size(704, 480));
	
	Mat imageROI_sides(frame.rows * ROI_height, frame.cols * side_width * 2, CV_8UC3, (Scalar(0, 0, 0))), imageROI_mid;
	imageROI_mid = frame(Rect(frame.cols * left_mid, frame.rows * top_edge, frame.cols * mid_width, frame.rows * ROI_height));

	Mat imageROI_left = frame(Rect(frame.cols * left_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));
	Mat imageROI_right = frame(Rect(frame.cols * right_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));

	imageROI_left.copyTo(imageROI_sides(Rect(0, 0, imageROI_left.cols, imageROI_left.rows)));
	imageROI_right.copyTo(imageROI_sides(Rect(imageROI_left.cols, 0, imageROI_right.cols, imageROI_right.rows)));

	#if DEBUG_FLAG
		printf("isStart______: %d\n", isStart);
		printf("1. Size of the Image ROI on sides: %dX%d\n\n",imageROI_sides.size().width, imageROI_sides.size().height);
		printf("2. Size of the Image ROI in mid: %dX%d\n\n",imageROI_mid.size().width, imageROI_sides.size().height);
	#endif

	GaussianBlur(imageROI_sides, imageROI_sides, Size(3, 3), 0 );
	GaussianBlur(imageROI_mid, imageROI_mid, Size(3, 3), 0 );
	res.push_back(imageROI_sides);
	res.push_back(imageROI_mid);

	return res;
}

Mat detect_TLDS_light(Mat& imageROI)
{
	Mat lightROI;
	vector<Rect> light;

	#if DEBUG_FLAG
		printf("2. Detecting the traffic light.\n\n");
	#endif
	traffic_light.detectMultiScale(imageROI, light, 1.02, 3, 0|2);
	
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
		#endif
		isLightDetected = true;
	}
	else
	{
		isLightDetected = false;
	}
	if (light.size() > 0){
		#if DISPLAYALL
			namedWindow("lightROI", WINDOW_NORMAL);
			resizeWindow("lightROI", 120, 90);
			imshow("lightROI", lightROI);
		#endif
	}
	return lightROI;
}

bool isArrow(Mat& lightROI, double minMatchQuality)
{
	Mat result;
	Mat img = lightROI;
	templ = imread(TEMPLATE_PATH);
	resize(templ, templ, Size(7, 7));
	rotate(templ, templ, ROTATE_90_CLOCKWISE);
	rotate(templ, templ, ROTATE_90_CLOCKWISE);

	/// Create windows
	Mat img_display;
	img.copyTo(img_display);

	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching
	matchTemplate(img, templ, result, 5);

	/// Create Trackbar
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	
	matchLoc = maxLoc;
	#if DEBUG_FLAG
		printf("maxVal: %f\n", maxVal);
		printf("minVal: %f\n", minVal);
	#endif
	if ((templ.cols > lightROI.cols) || (templ.rows > lightROI.rows))
		return false;
	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (maxVal > minMatchQuality){

		Mat rect;
		img_display(Rect(matchLoc.x, matchLoc.y, templ.cols, templ.rows)).copyTo(rect);
		#if DISPLAYALL
			namedWindow("detected", WINDOW_NORMAL);
			resizeWindow("detected", Size(500, 500));
			imshow("detected", rect);
			namedWindow("image_window", WINDOW_AUTOSIZE);
			imshow("image_window", img_display );
		#endif

		#if DEBUG_FLAG
			printf("detected the arrow\n");
			#endif
		return true;
	}
	else
	{
		#if DISPLAYALL
			namedWindow("image_window", WINDOW_AUTOSIZE);
			imshow("image_window", img_display );
		#endif

		#if DEBUG_FLAG
			printf("arrow detection failed!!!!!!\n");
		#endif
	}
	return false;
}

int segmentColor(Mat& lightROI)
{	
	Mat red_image, green_image, lower_red_hue_range, upper_red_hue_range, saturated_red_image, saturated_green_image, yellow_hue_image;
	Mat red_hue_image, green_hue_image, lower_green_hue_range, upper_green_hue_range;
	// For Red light: Threshold the HSV image, keep only the red pixels
	red_image = lightROI.clone();
	green_image = lightROI.clone();

	vector<Mat> channel;

	cvtColor(green_image, saturated_green_image, COLOR_BGR2HSV);
	split(saturated_green_image, channel);
	channel[1] *= 1.2;
	merge(channel, saturated_green_image);

	#if DEBUG_FLAG
		printf("6. Finding out which color is present.\n");
	#endif

	// Yellow color segmentation --------------------------------------------------------------------
	Mat leftYel, rightYel;
	cvtColor(red_image, red_image, COLOR_BGR2HSV);

	inRange(red_image, Scalar(20, 150, 100), Scalar(40, 255, 255), yellow_hue_image);
	GaussianBlur(yellow_hue_image, yellow_hue_image, Size(5, 5), 0, 0);

	Rect leftYelRect(0, 0, yellow_hue_image.cols / 2, yellow_hue_image.rows / 2);
	yellow_hue_image(Rect(leftYelRect.x, leftYelRect.y, leftYelRect.width, leftYelRect.height)).copyTo(leftYel);

	Rect rightYelRect(yellow_hue_image.cols / 2, 0, yellow_hue_image.cols / 2, yellow_hue_image.rows / 2);
	yellow_hue_image(Rect(rightYelRect.x, rightYelRect.y, rightYelRect.width, rightYelRect.height)).copyTo(rightYel);

	yel_val1 = getAverage(leftYel);
	yel_val2 = getAverage(rightYel);

	if (yel_val1 != 0 || yel_val2 != 0)
	{
		ifYelHappened = true;
	}

	#if DEBUG_FLAG
		printf("Left yellow vs right yellow: %d vs %d\n", yel_val1, yel_val2);
	#endif

	// Red color segmentation --------------------------------------------------------------------
	Mat topLeftRed, topRightRed, botLeftRed, botRightRed;

	inRange(red_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(170, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	Rect topLeftRedRect(0, 0, red_hue_image.cols / 2, red_hue_image.rows / 2);
	red_hue_image(Rect(topLeftRedRect.x, topLeftRedRect.y, topLeftRedRect.width, topLeftRedRect.height)).copyTo(topLeftRed);


	inRange(red_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(170, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	Rect topRightRedRect(red_hue_image.cols / 2, 0, red_hue_image.cols / 2, red_hue_image.rows / 2);
	red_hue_image(Rect(topRightRedRect.x, topRightRedRect.y, topRightRedRect.width, topRightRedRect.height)).copyTo(topRightRed);


	inRange(red_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(170, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	Rect botLeftRedRect(0, red_hue_image.rows / 2, red_hue_image.cols / 2, red_hue_image.rows / 2);
	red_hue_image(Rect(botLeftRedRect.x, botLeftRedRect.y, botLeftRedRect.width, botLeftRedRect.height)).copyTo(botLeftRed);
	GaussianBlur(botLeftRed, botLeftRed, Size(3, 3), 0 );


	inRange(red_image, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(170, 100, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	Rect botRightRedRect(red_hue_image.cols / 2, red_hue_image.rows / 2, red_hue_image.cols / 2, red_hue_image.rows / 2);
	red_hue_image(Rect(botRightRedRect.x, botRightRedRect.y, botRightRedRect.width, botRightRedRect.height)).copyTo(botRightRed);
	GaussianBlur(botRightRed, botRightRed, Size(3, 3), 0 );


	value1 = getAverage(topLeftRed);
	value2 = getAverage(topRightRed);
	value3 = getAverage(botLeftRed);
	value4 = getAverage(botRightRed);

	#if DISPLAYALL
		namedWindow("Red Combined threshold images", WINDOW_NORMAL);
		resizeWindow("Red Combined threshold images", 120, 90);
		imshow("Red Combined threshold images", red_hue_image);
	#endif
	if (value1 + value2 < 6 && red_fail_count < 3)
	{
		red_fail_count++;
	}
	/*else if ((value1 + value2 < 6 && red_fail_count == 3 && isGreenColor == false) || segmentationType == 1)
	{	
		saturated_red_image = red_image.clone();
		split(saturated_red_image, channel);
		channel[1] *= 1.2;
		merge(channel, saturated_red_image);
		#if DEBUG_FLAG
			printf("=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X=X===UPDATED COLOR RANGE\n");
		#endif

		inRange(saturated_red_image, Scalar(0, 87, 100), Scalar(25, 255, 255), lower_red_hue_range);

		inRange(saturated_red_image, Scalar(155, 87, 100), Scalar(180, 255, 255), upper_red_hue_range);

		addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

		Rect topLeftRedRect(0, 0, red_hue_image.cols, red_hue_image.rows / 2);
		red_hue_image(Rect(topLeftRedRect.x, topLeftRedRect.y, topLeftRedRect.width, topLeftRedRect.height)).copyTo(topLeftRed);

		inRange(saturated_red_image, Scalar(0, 87, 100), Scalar(25, 255, 255), lower_red_hue_range);

		inRange(saturated_red_image, Scalar(155, 87, 100), Scalar(180, 255, 255), upper_red_hue_range);

		addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

		Rect topRightRedRect(0, 0, red_hue_image.cols, red_hue_image.rows / 2);
		red_hue_image(Rect(topRightRedRect.x, topRightRedRect.y, topRightRedRect.width, topRightRedRect.height)).copyTo(topRightRed);

		inRange(saturated_red_image, Scalar(0, 87, 100), Scalar(25, 255, 255), lower_red_hue_range);

		inRange(saturated_red_image, Scalar(155, 87, 100), Scalar(180, 255, 255), upper_red_hue_range);

		addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

		Rect botLeftRedRect(0, red_hue_image.rows / 2, red_hue_image.cols / 2, red_hue_image.rows / 2);
		red_hue_image(Rect(botLeftRedRect.x, botLeftRedRect.y, botLeftRedRect.width, botLeftRedRect.height)).copyTo(botLeftRed);

		inRange(saturated_red_image, Scalar(0, 87, 100), Scalar(25, 255, 255), lower_red_hue_range);

		inRange(saturated_red_image, Scalar(155, 87, 100), Scalar(180, 255, 255), upper_red_hue_range);

		addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
		GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

		Rect botRightRedRect(red_hue_image.cols / 2, red_hue_image.rows / 2, red_hue_image.cols / 2, red_hue_image.rows / 2);
		red_hue_image(Rect(botRightRedRect.x, botRightRedRect.y, botRightRedRect.width, botRightRedRect.height)).copyTo(botRightRed);
		
		value1 = getAverage(topLeftRed);
		value2 = getAverage(topRightRed);
		val3_sat = getAverage(botLeftRed);
		val4_sat = getAverage(botRightRed);
	}*/
	else if (red_fail_count > 0)
	{
		red_fail_count--;
	}
	#if DEBUG_FLAG
		printf("Saturated Red image botleft vs botright: %d vs %d\n", val3_sat, val4_sat);
	#endif
	#if DEBUG_FLAG
		printf("Red image top left vs top right vs botleft vs botright: %d vs %d vs %d vs %d\n", value1, value2, value3, value4);
	#endif
	// Red color segmentation --------------------------------------------------------------------
	vector<Mat> arr_to_merge;
	arr_to_merge.push_back(red_hue_image);
	arr_to_merge.push_back(red_hue_image);
	arr_to_merge.push_back(red_hue_image);
	Mat color_red_hue_image, red_rect(350, 220, CV_8UC3, (Scalar(0, 0, 255)));
	merge(arr_to_merge, color_red_hue_image);
	resize(color_red_hue_image, color_red_hue_image, Size(170, 300));

	red_rect.copyTo(res_image(Rect(995, 475, red_rect.cols, red_rect.rows)));
	color_red_hue_image.copyTo(res_image(Rect(1020, 500, color_red_hue_image.cols, color_red_hue_image.rows)));
	line(res_image, cvPoint(995, 475 + int(red_rect.rows / 2)), cvPoint(995 + red_rect.cols, 475 + int(red_rect.rows / 2)), Scalar(255, 255, 255), 2);
	arr_to_merge.clear();

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	// Green color segmentation --------------------------------------------------------------------
	
	Mat topRightGreen, topLeftGreen, botLeftGreen, botRightGreen;
	cvtColor(green_image, green_image, COLOR_BGR2HSV);

	inRange(green_image, Scalar(50, 120, 100), Scalar(80, 255, 255), green_hue_image);
	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);

	Rect topLeftGreenRect(0, 0, green_hue_image.cols / 2, red_hue_image.rows / 2);
	green_hue_image(Rect(topLeftGreenRect.x, topLeftGreenRect.y, topLeftGreenRect.width, topLeftGreenRect.height)).copyTo(topLeftGreen);
	GaussianBlur(topLeftGreen, topLeftGreen, Size(3, 3), 0, 0);

	Rect topRightGreenRect(green_hue_image.cols / 2, 0, green_hue_image.cols / 2, red_hue_image.rows / 2);
	green_hue_image(Rect(topRightGreenRect.x, topRightGreenRect.y, topRightGreenRect.width, topRightGreenRect.height)).copyTo(topRightGreen);
	GaussianBlur(topRightGreen, topRightGreen, Size(3, 3), 0, 0);


	// inRange(saturated_green_image, Scalar(45, 80, 90), Scalar(90, 255, 255), green_hue_image);
	// GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);
	
	Rect botLeftGreenRect(0, green_hue_image.rows / 2, green_hue_image.cols / 2, red_hue_image.rows / 2);
	green_hue_image(Rect(botLeftGreenRect.x, botLeftGreenRect.y, botLeftGreenRect.width, botLeftGreenRect.height)).copyTo(botLeftGreen);

	Rect botRightGreenRect(green_hue_image.cols / 2, green_hue_image.rows / 2, green_hue_image.cols / 2, red_hue_image.rows / 2);
	green_hue_image(Rect(botRightGreenRect.x, botRightGreenRect.y, botRightGreenRect.width, botRightGreenRect.height)).copyTo(botRightGreen);


	value5 = getAverage(topLeftGreen);
	value6 = getAverage(topRightGreen);
	value7 = getAverage(botLeftGreen);
	value8 = getAverage(botRightGreen);
	#if DEBUG_FLAG
		printf("Green image top left vs top right vs botleft vs botright: %d vs %d vs %d vs %d\n", value5, value6, value7, value8);
	#endif

	#if DISPLAYALL
		namedWindow("Green Combined threshold images", WINDOW_NORMAL);
		resizeWindow("Green Combined threshold images", 120, 90);
		imshow("Green Combined threshold images", green_hue_image);
	#endif

	// Green color segmentation --------------------------------------------------------------------
	arr_to_merge.push_back(green_hue_image);
	arr_to_merge.push_back(green_hue_image);
	arr_to_merge.push_back(green_hue_image);
	Mat color_green_hue_image, green_rect(350, 220, CV_8UC3, (Scalar(0, 255, 0)));
	merge(arr_to_merge, color_green_hue_image);
	resize(color_green_hue_image, color_green_hue_image, Size(170, 300));

	green_rect.copyTo(res_image(Rect(1295, 475, green_rect.cols, green_rect.rows)));
	color_green_hue_image.copyTo(res_image(Rect(1320, 500, color_green_hue_image.cols, color_green_hue_image.rows)));
	line(res_image, cvPoint(1295, 475 + int(green_rect.rows / 2)), cvPoint(1295 + green_rect.cols, 475 + int(green_rect.rows / 2)), Scalar(255, 255, 255), 2);
	arr_to_merge.clear();

	if ((value7 + value8 < 12) && ((value1 + value2 < val3_sat + val4_sat + 20 && value1 + value2 > val3_sat + val4_sat) || (val3_sat + val4_sat < value1 + value2 + 20 && val3_sat + val4_sat > value1 + value2)))
	{
		#if DEBUG_FLAG
			printf("first SEG_FAIL\n");
		#endif
		return SEG_FAIL;
	}
	if (value7 + value8 < 12 && value1 + value2 > 12 && value3 + value4 < 12) // && value1 > 0 && value2 > 0) // red color
	{
		if ((isLeft == true || isRight == true) && (value1 == 0 || value2 == 0))
		{
			#if DEBUG_FLAG
				printf("left vs right: %d vs %d\n", isLeft, isRight);
				printf("isLeft || isRight RED\n");
			#endif
			return RED;
		}
		if (yel_val1 > yel_val2 || (value1 > value2 && yel_val1 == 0 && yel_val2 == 0 && ifYelHappened == false))
		{
			isLeft = true;
			isRight = false;
		}
		else if (yel_val2 > yel_val1 || (value2 > value1 && yel_val1 == 0 && yel_val2 == 0 && ifYelHappened == false))
		{
			isRight = true;
			isLeft = false;
		}
		#if DEBUG_FLAG
			printf("yel_val1 || yel_val2 RED\n");
			printf("left vs right: %d vs %d\n", isLeft, isRight);
		#endif
		return RED;
	}
	if (value1 + value2 > 12 && value7 + value8 > 12 && isLeft && value7 > value8) // green color and right red arrow
	{
		#if DEBUG_FLAG
			printf("GREEN_RIGHT_RED_ARROW\n");
		#endif
		return GREEN_RIGHT_RED_ARROW;
	}

	else if (value7 + value8 > 12 && value1 + value2 > 12 && isRight && value7 < value8) // green color and left red arrow
	{
		#if DEBUG_FLAG
			printf("GREEN_LEFT_RED_ARROW\n");
		#endif
		return GREEN_LEFT_RED_ARROW;
	}

	else if (value1 + value2 > 12 && value7 + value8 > 12 && isLeft && value7 <= value8) // red color and right green arrow
	{
		#if DEBUG_FLAG
			printf("RED_RIGHT_GREEN_ARROW\n");
		#endif
		return RED_RIGHT_GREEN_ARROW;
	}

	else if (value1 + value2 > 12 && value7 + value8 > 12 && isRight && value7 >= value8) // red color and left green arrow
	{
		#if DEBUG_FLAG
			printf("RED_LEFT_GREEN_ARROW\n");
		#endif
		return RED_LEFT_GREEN_ARROW;
	}
	else if (value1 + value2 > 12 && value7 + value8 > 12) // red color and left green arrow
	{
		#if DEBUG_FLAG
			printf("RED_LEFT_GREEN_ARROW\n");
		#endif
		return RED_LEFT_GREEN_ARROW;
	}

	if (value1 + value2 < 12 && value7 + value8 > 12 && value5 + value6 < 12) // green color
	{
		#if DEBUG_FLAG
			printf("GREEN\n");
		#endif
		return GREEN;
	}
	#if DEBUG_FLAG
		printf("SEG_FAIL\n");
	#endif
	return SEG_FAIL;
}

int colorDecision(int segmentationType, bool fourPattern)
{
	if (segmentationType == RED_RIGHT_GREEN_ARROW || segmentationType == RED_LEFT_GREEN_ARROW){
		#if DEBUG_FLAG
			printf("7. Segmented light area is on both sides: Turn left and RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
	}
	if (segmentationType == RED)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is on top : RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
		return 0;
	}
	else if (segmentationType == GREEN || segmentationType == GREEN_LEFT_RED_ARROW || segmentationType == GREEN_RIGHT_RED_ARROW)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is bottom : GREEN LIGHT. \n\n");
		#endif
		isGreenColor = true;
		isRedColor = false;
	}
	else
	{
		#if DEBUG_FLAG
			printf("7. ThreePattern: Color segmentation failed !!!!!\n\n");
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

vector<int> segmentColor_night(Mat& imageROI, bool display) // , bool contour)
{	
	Mat red_image, green_image, lower_red_hue_range, upper_red_hue_range, saturated_red_image, saturated_green_image;
	Mat red_hue_image, green_hue_image, lower_green_hue_range, upper_green_hue_range;
	vector<Mat> arr_to_merge;
	Rect bounding_rect;
	// For Red light: Threshold the HSV image, keep only the red pixels
	red_image = imageROI.clone();
	green_image = imageROI.clone();

	if (display)
	{
		#if DEBUG_FLAG
			printf("6. Finding out which color is present.\n");
		#endif
	}
	// vector<Mat> channel;

	// saturated_red_image = red_image.clone();
	// split(saturated_red_image, channel);
	// channel[1] *= 1.2;
	// merge(channel, saturated_red_image);
	// Red color segmentation --------------------------------------------------------------------

	cvtColor(red_image, red_image, COLOR_BGR2HSV);
	inRange(red_image, Scalar(0, 90, 100), Scalar(10, 255, 255), lower_red_hue_range);

	inRange(red_image, Scalar(150, 90, 100), Scalar(180, 255, 255), upper_red_hue_range);

	addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
	GaussianBlur(red_hue_image, red_hue_image, Size(5, 5), 0, 0);	

	value1 = getAverage(red_hue_image);
	
	if (display)
	{
		#if DEBUG_FLAG
			printf("Red image value: %d\n", value1);
		#endif
	}
	if (display)
	{
		arr_to_merge.push_back(red_hue_image);
		arr_to_merge.push_back(red_hue_image);
		arr_to_merge.push_back(red_hue_image);
		Mat color_red_hue_image, red_rect(376, 450, CV_8UC3, (Scalar(0, 0, 255)));
		merge(arr_to_merge, color_red_hue_image);
		resize(color_red_hue_image, color_red_hue_image, Size(400, 326));

		red_rect.copyTo(res_image(Rect(875, 275, red_rect.cols, red_rect.rows)));
		color_red_hue_image.copyTo(res_image(Rect(900, 300, color_red_hue_image.cols, color_red_hue_image.rows)));
		arr_to_merge.clear();
		line(res_image, cvPoint(900 + int(color_red_hue_image.cols * 0), 300 + color_red_hue_image.rows * 0.225), cvPoint(900 + int(color_red_hue_image.cols), 300 + color_red_hue_image.rows * 0.225), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(900 + int(color_red_hue_image.cols * 0), 300 + color_red_hue_image.rows * 0.45), cvPoint(900 + int(color_red_hue_image.cols), 300 + color_red_hue_image.rows * 0.45), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(900 + int(color_red_hue_image.cols * 0), 300 + color_red_hue_image.rows * 0.65), cvPoint(900 + int(color_red_hue_image.cols), 300 + color_red_hue_image.rows * 0.65), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(900 + int(color_red_hue_image.cols * 0), 300 + color_red_hue_image.rows * 0.85), cvPoint(900 + int(color_red_hue_image.cols), 300 + color_red_hue_image.rows * 0.85), Scalar(255, 255, 255), 2);
	}
	// Red color segmentation --------------------------------------------------------------------

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	// Green color segmentation --------------------------------------------------------------------

	cvtColor(green_image, green_image, COLOR_BGR2HSV);

	inRange(green_image, Scalar(40, 70, 90), Scalar(90, 255, 255), green_hue_image);

	GaussianBlur(green_hue_image, green_hue_image, Size(5, 5), 0, 0);

	
	value2 = getAverage(green_hue_image);
	if (display)
	{
		#if DEBUG_FLAG
			printf("Green image value: %d \n", value2);
		#endif
	}

	if (display)
	{
		arr_to_merge.push_back(green_hue_image);
		arr_to_merge.push_back(green_hue_image);
		arr_to_merge.push_back(green_hue_image);
		Mat color_green_hue_image, green_rect(376, 450, CV_8UC3, (Scalar(0, 255, 0)));
		merge(arr_to_merge, color_green_hue_image);
		resize(color_green_hue_image, color_green_hue_image, Size(400, 326));

		green_rect.copyTo(res_image(Rect(1325, 275, green_rect.cols, green_rect.rows)));
		color_green_hue_image.copyTo(res_image(Rect(1350, 300, color_green_hue_image.cols, color_green_hue_image.rows)));
		arr_to_merge.clear();
		line(res_image, cvPoint(1350 + int(color_green_hue_image.cols * 0), 300 + color_green_hue_image.rows * 0.225), cvPoint(1350 + int(color_green_hue_image.cols), 300 + color_green_hue_image.rows * 0.225), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(1350 + int(color_green_hue_image.cols * 0), 300 + color_green_hue_image.rows * 0.45), cvPoint(1350 + int(color_green_hue_image.cols), 300 + color_green_hue_image.rows * 0.45), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(1350 + int(color_green_hue_image.cols * 0), 300 + color_green_hue_image.rows * 0.65), cvPoint(1350 + int(color_green_hue_image.cols), 300 + color_green_hue_image.rows * 0.65), Scalar(255, 255, 255), 2);
		line(res_image, cvPoint(1350 + int(color_green_hue_image.cols * 0), 300 + color_green_hue_image.rows * 0.85), cvPoint(1350 + int(color_green_hue_image.cols), 300 + color_green_hue_image.rows * 0.85), Scalar(255, 255, 255), 2);
	}

	vector<int> res;
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

	if ((red_sum >= 10) && (green_sum >= 10)){ // green and red circle
		return RED_AND_GREEN;
	}
	if ((red_sum >= 10) && (green_sum <= 10)){ // red cirle
		return RED;
	}
	if ((green_sum >= 10) && (red_sum <= 10)){ // green circle
		return GREEN;
	}
	if ((red_sum == 0 && green_sum == 0)){ // none
		return SEG_FAIL;
	}
	return SEG_FAIL;
}

int colorDecision_night(int segmentationType)
{
	if (segmentationType == RED_AND_GREEN)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is on both sides: Turn left and RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
	}

	else if (segmentationType == RED)
	{
		#if DEBUG_FLAG
			printf("7. Segmented light area is on leftside : RED LIGHT. \n\n");
		#endif
		isRedColor = true;
		ifRedHappened = true;
		return 0;
	}
	else if (segmentationType == GREEN)
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

void StartTLDS()
{
	isStart = true;
	isUsingDetector = true;
	thresholdcount = 0;
	left_thresholdcount = 0;
	detected_frames_count = 0;
	color_seg_fail_count = 0;
	onSides = false;
}


void EndTLDS()
{
	isStart = false;
	isRedColor = false;
	isGreenColor = false;
	ifRedHappened = false;
	ifYelHappened = false;
	detected_frames_count = 0;
	color_seg_fail_count = 0;
	segmentationType = -1;
	flag_alarm = 0;
	onSides = false;
	isLeft = false;
	isRight = false;
}

int main()
{
	namedWindow("demo", WINDOW_AUTOSIZE);
	Mat tr_light, tr_stand, alarm_image;
	tr_stand = imread("/home/rohit/adas/car_lane_sign_detection/au_tlds/tr_stand.png");
	tr_stand.copyTo(res_image(Rect(1662, 500, tr_stand.cols, tr_stand.rows)));
	tr_light = imread("/home/rohit/adas/car_lane_sign_detection/au_tlds/zdemo_traffic_light.png");
	rotate(tr_light, tr_light, ROTATE_90_CLOCKWISE);
	alarm_image = imread("/home/rohit/adas/car_lane_sign_detection/au_tlds/zalarm_image.png");
	resize(tr_light, tr_light, Size(300, 300));
	int tr_light_x = 1550, tr_light_y = 300;
	tr_light.copyTo(res_image(Rect(tr_light_x, tr_light_y, tr_light.cols, tr_light.rows)));
	circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(30, 30, 30), FILLED);
	circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 150), 39, Scalar(30, 30, 30), FILLED);
	circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(30, 30, 30), FILLED);
	
	#if DEBUG_FLAG
		printf("2. Inside the tlds function.\n\n");
	#endif
	#if SAVE_DEBUG_TO_TEXT
		 freopen("/app/sd/Blackvue/Record/debug_tlds.txt","a",stdout);
	#endif

	#if DEBUG_FLAG
		printf("3. GPS SPEED and MAX SPEED satisfy the conditions.\n\n");
		printf("TLDS starts.\n\n");
	#endif

	Mat imageROI_sides, imageROI_mid, lightROI_sides, lightROI_mid;
	vector<Mat> preproc_ROI;
	vector<int> whole_image, top, middle_top, middle, middle_bot, bot;
	
	bool fourPattern;

	loadDetector();
	#if DEBUG_FLAG
		printf("Finish loading dataset. \n");
	#endif
	int text_count = 1;

	getVideoData();

	StartTLDS();

	Mat frame;
	while (cap.read(frame))
	{	
		resize(frame, frame, Size(704, 480));
		Mat imROI = frame(Rect(frame.cols*0.30, frame.rows*0.05, frame.cols*0.40, frame.rows*0.40));

		Mat frame_hsv;
		cvtColor(imROI, frame_hsv, COLOR_BGR2HSV);

		vector<Mat> channel;
		split(frame_hsv, channel);
		int value = getAverage(channel[2]);

		Mat black_hue;
		inRange(frame_hsv, Scalar(0, 0, 0), Scalar(180, 255, 40), black_hue);

		GaussianBlur(black_hue, black_hue, Size(5, 5), 0, 0);
		int black = getAverage(black_hue);
		Mat imageROI;
		if (onSides)
		{
			imageROI = Mat(frame.rows * ROI_height, frame.cols * side_width * 2, CV_8UC3, (Scalar(0, 0, 0)));

			Mat imageROI_left = frame(Rect(frame.cols * left_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));
			Mat imageROI_right = frame(Rect(frame.cols * right_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));

			imageROI_left.copyTo(imageROI(Rect(0, 0, imageROI_left.cols, imageROI_left.rows)));
			imageROI_right.copyTo(imageROI(Rect(imageROI_left.cols, 0, imageROI_right.cols, imageROI_right.rows)));	
		}
		else
		{
			imageROI = frame(Rect(frame.cols * left_mid, frame.rows * top_edge, frame.cols * mid_width, frame.rows * ROI_height));
		}

		if (framecount % SKIPFRAME == 0)
		{	
			auto start_time = std::chrono::high_resolution_clock::now();
			
			if (isStart == true)
			{	
				printf("VALUE, black: %d, %d\n", value, black);


				// NIGHT PART
// -NIGHT------NIGHT-----NIGHT-----NIGHT------NIGHT------NIGHT----NIGHT---NIGHT----NIGHT----NIGHT-----NIGHT---NIGHT------NIGHT---NIGHT----NIGHT----NIGHT---
				if ((value < 130 && black > 25) || (value < 90))
				{
					if (text_count == 1)
					{
						putText(res_image, "INPUT FRAME", cvPoint(10 + int(704 / 2) - 40, 30 + 485), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "IMAGE ROI", cvPoint(240 + int(211 / 2), 600 + 330), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "RED COLOR SEGM", cvPoint(950 + 70, 670 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "GREEN COLOR SEGM", cvPoint(1390 + 70, 670 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						text_count++;
					}
					
					Mat lightROI_template;

					#if DEBUG_FLAG
						printf("	night time!\n");
					#endif

					imageROI = frame(Rect(frame.cols*night_left, frame.rows*night_top, frame.cols*night_width, frame.rows*0.45));
					whole_image = segmentColor_night(imageROI, true);

					// =========================================================================================
					Mat imageROI_top = frame(Rect(frame.cols*night_left, frame.rows*(night_top), frame.cols*night_width, frame.rows*night_top_height));
					top = segmentColor_night(imageROI_top, false);

					Mat imageROI_middle_top = frame(Rect(frame.cols*night_left, frame.rows*(night_top + night_top_height), frame.cols*night_width, frame.rows*night_mid_top_height));
					middle_top = segmentColor_night(imageROI_middle_top, false);
 
					Mat imageROI_middle = frame(Rect(frame.cols*night_left, frame.rows*(night_top + night_mid_top_height + night_mid_top_height), frame.cols*night_width, frame.rows*night_mid_height));
					middle = segmentColor_night(imageROI_middle, false);

					Mat imageROI_middle_bot = frame(Rect(frame.cols*night_left, frame.rows*(night_top + night_mid_top_height + night_mid_top_height + night_mid_height), frame.cols*night_width, frame.rows*night_mid_bot_height));
					middle_bot = segmentColor_night(imageROI_middle_bot, false);

					Mat imageROI_bot = frame(Rect(frame.cols*night_left, frame.rows*(night_top + night_mid_top_height + night_mid_top_height + night_mid_height + night_mid_bot_height), frame.cols*night_width, frame.rows*night_bot_height));
					bot = segmentColor_night(imageROI_bot, false);
					// ==========================================================================================

					segmentationType = segmentDecision_night(top, middle_top, middle, middle_bot, bot);

					#if DEBUG_FLAG
						printf("segmentationType: %d\n", segmentationType);
					#endif
					flag_alarm = colorDecision_night(segmentationType);
					resize(imageROI, imageROI, Size(633, 292));
					imageROI.copyTo(res_image(Rect(85, 580, imageROI.cols, imageROI.rows)));
					// resize(imageROI, imageROI, Size(400, 326));
					// imageROI.copyTo(res_image(Rect(200, 580, imageROI.cols, imageROI.rows)));
				}

				else
// -NIGHT------NIGHT-----NIGHT-----NIGHT------NIGHT------NIGHT----NIGHT---NIGHT----NIGHT----NIGHT-----NIGHT---NIGHT------NIGHT---NIGHT----NIGHT----NIGHT---

				/*FIXED PART below
				If the detection succeeded for 3 times 
				and it was a red light
				we fix the ROI*/
				{
					if (text_count == 1)
					{
						putText(res_image, "INPUT FRAME", cvPoint(10 + int(704 / 2) - 40, 30 + 485), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "IMAGE ROI", cvPoint(240 + int(211 / 2), 600 + 330), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "DETECTED LIGHT ROI", cvPoint(950 + 50, 420 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "FIXED LIGHT ROI", cvPoint(1250 + 70, 420 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "RED SEGM", cvPoint(985 + 70, 850 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						putText(res_image, "GREEN SEGM", cvPoint(1275 + 70, 850 + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250, 250, 250), 1, LINE_AA);
						text_count++;
					}
					Mat lightROI;
					if (detected_frames_count >= 3 || (ifRedHappened == true && isGreenColor == true && value7 + value8 != 0))
					{
						if (onSides)
						{
							Mat imageROI_template_sides(frame.rows * ROI_height, frame.cols * side_width * 2, CV_8UC3, (Scalar(0, 0, 0)));

							Mat imageROI_left = frame(Rect(frame.cols * left_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));
							Mat imageROI_right = frame(Rect(frame.cols * right_edge, frame.rows * top_edge, frame.cols * side_width, frame.rows * ROI_height));

							imageROI_left.copyTo(imageROI_template_sides(Rect(0, 0, imageROI_left.cols, imageROI_left.rows)));
							imageROI_right.copyTo(imageROI_template_sides(Rect(imageROI_left.cols, 0, imageROI_right.cols, imageROI_right.rows)));

							imageROI_template_sides(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI);	
						}
						else
						{
							Mat imageROI_template_mid = frame(Rect(frame.cols * left_mid, frame.rows * top_edge, frame.cols * mid_width, frame.rows * ROI_height));

							imageROI_template_mid(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI);
						}
						lightROI(Rect(lightROI.cols * 0.1, lightROI.rows * 0.14, lightROI.cols * 0.8, lightROI.rows * 0.72)).copyTo(lightROI);
						#if DEBUG_FLAG
							printf("lightROI RESIZED: %dX%d\n", lightROI.cols, lightROI.rows);
						#endif

						fourPattern = false;

						segmentationType = segmentColor(lightROI);
						if ((segmentationType == SEG_FAIL) && (detected_frames_count > 0) && (color_seg_fail_count < 10)){ // && ifRedHappened == false){
							color_seg_fail_count++;
						}

						else if ((segmentationType == SEG_FAIL) && (color_seg_fail_count == 10) && (value1 + value2 < 12) && (detected_frames_count > 0)){
							detected_frames_count--;
							color_seg_fail_count = 0;
							imshow("demo", res_image);
							waitKey(1);
							continue;
						}
						else if ((segmentationType == RED && value2 + value1 > 200) || (value3 + value4 > value1 + value2 && segmentationType == SEG_FAIL))
						{
							detected_frames_count--;
							color_seg_fail_count = 0;
							imshow("demo", res_image);
							waitKey(1);
							continue;
						}

						#if DEBUG_FLAG
							cout << "segmentationType: " << segmentationType << endl;
							cout << "color_seg_fail_count: " << color_seg_fail_count << endl;
							printf("detected_frames_count_fixed: %d\n", detected_frames_count);
						#endif

						flag_alarm = colorDecision(segmentationType, fourPattern);
						resize(imageROI, imageROI, Size(633, 292));
						imageROI.copyTo(res_image(Rect(85, 580, imageROI.cols, imageROI.rows)));
						
						resize(lightROI, lightROI, Size(170, 300));
						lightROI.copyTo(res_image(Rect(1320, 100, lightROI.cols, lightROI.rows)));
					}


					// /*DETECTION PART below
					// Here we find the red traffic light 
					// and if it passes certain threshold number of frames
					// we fix it and go to the above section*/

					else{
						Mat lightROI_template;
						Mat frame_template = frame.clone();
						preproc_ROI = framePreprocessing(frame);

						imageROI_sides = preproc_ROI.at(0);
						imageROI_mid = preproc_ROI.at(1);

						if (isUsingDetector = true)
						{	

							lightROI_mid = detect_TLDS_light(imageROI_mid);

							#if DEBUG_FLAG
								printf("	after mid detection!\n");
							#endif

							if (isLightDetected == true)
							{	
								onSides = false;
								resize(frame_template, frame_template, Size(704, 480));
								Mat imageROI_template_mid = frame_template(Rect(frame_template.cols * left_mid, frame_template.rows * top_edge, frame_template.cols * mid_width, frame_template.rows * ROI_height));

								imageROI_template_mid(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI_template);
								lightROI_template(Rect(lightROI_template.cols * 0.1, lightROI_template.rows * 0.14, lightROI_template.cols * 0.8, lightROI_template.rows * 0.72)).copyTo(lightROI_template);

								#if DEBUG_FLAG
									printf("lightROI_template size: %dx%d\n", lightROI_template.cols, lightROI_template.rows);
								#endif
								fourPattern = false;	
								imageROI = imageROI_template_mid;
							}
							else
							{
								lightROI_sides = detect_TLDS_light(imageROI_sides);
						
								#if DEBUG_FLAG
									printf("	after sides detection!\n");
								#endif
								if (isLightDetected == true)
								{	
									onSides = true;
									lightROI_sides(Rect(lightROI_sides.cols * 0.1, lightROI_sides.rows * 0.14, lightROI_sides.cols * 0.8, lightROI_sides.rows * 0.72)).copyTo(lightROI_sides);
									resize(frame_template, frame_template, Size(704, 480));
									Mat imageROI_template_sides(frame_template.rows * ROI_height, frame_template.cols * side_width * 2, CV_8UC3, (Scalar(0, 0, 0)));

									Mat imageROI_left = frame_template(Rect(frame_template.cols * left_edge, frame_template.rows * top_edge, frame_template.cols * side_width, frame_template.rows * ROI_height));
									Mat imageROI_right = frame_template(Rect(frame_template.cols * right_edge, frame_template.rows * top_edge, frame_template.cols * side_width, frame_template.rows * ROI_height));

									imageROI_left.copyTo(imageROI_template_sides(Rect(0, 0, imageROI_left.cols, imageROI_left.rows)));
									imageROI_right.copyTo(imageROI_template_sides(Rect(imageROI_left.cols, 0, imageROI_right.cols, imageROI_right.rows)));

									imageROI_template_sides(Rect(detected_light.x, detected_light.y, detected_light.width, detected_light.height)).copyTo(lightROI_template);
									lightROI_template(Rect(lightROI_template.cols * 0.1, lightROI_template.rows * 0.14, lightROI_template.cols * 0.8, lightROI_template.rows * 0.72)).copyTo(lightROI_template);
									
									#if DEBUG_FLAG
										printf("lightROI_template size: %dx%d\n", lightROI_template.cols, lightROI_template.rows);
									#endif
									fourPattern = false;	
									imageROI = imageROI_template_sides;
								}
								else
								{	
									#if DEBUG_FLAG
										printf("	NO TRAFFIC LIGHT DETECTED!!!\n\n\n\n");
									#endif
									framecount++;
									imshow("demo", res_image);
									waitKey(1);
									continue;
								}
							}

							segmentationType = segmentColor(lightROI_template);
							#if DEBUG_FLAG
								printf("segmentationType: %d\n", segmentationType);
								printf("detected_frames_count: %d\n", detected_frames_count);
							#endif
							if ((segmentationType == RED || segmentationType == RED_LEFT_GREEN_ARROW || segmentationType == RED_RIGHT_GREEN_ARROW) && (detected_frames_count < 3))
							{
								detected_frames_count++;
							}
							if ((segmentationType == SEG_FAIL) && (detected_frames_count > 0) && (value1 + value2 < 12) && ifRedHappened == false)
							{
								detected_frames_count--;
							}
							flag_alarm = colorDecision(segmentationType, fourPattern);
							resize(imageROI, imageROI, Size(633, 292));
							imageROI.copyTo(res_image(Rect(85, 580, imageROI.cols, imageROI.rows)));
							resize(lightROI_template, lightROI_template, Size(170, 300));
							lightROI_template.copyTo(res_image(Rect(1020, 100, lightROI_template.cols, lightROI_template.rows)));
						}
					}
				}
			}
			
			auto end_time = std::chrono::high_resolution_clock::now();
			auto time = end_time - start_time;

			#if DEBUG_FLAG
				printf("Time: %lldms.\n       ", static_cast<long long int>(time / std::chrono::milliseconds(1)));
			#endif
						
			int c = waitKey(100);

			#if TLDS
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
					// break;
					alarm_stop.copyTo(res_image(Rect(685, 730, alarm_stop.cols, alarm_stop.rows)));
					if (isStart==true)
					{
						EndTLDS();
					}
				}
			#endif

			#if DEBUG_FLAG
				printf("#frame: %d\n", framecount);
				printf("flag_alarm: %d  ", flag_alarm);
				printf("========================================================================================\n\n\n\n");
			#endif

		}
		if (ifRedHappened == true && isGreenColor == false && isRedColor == true){
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(0, 0, 255), FILLED);
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(30, 30, 30), FILLED);
		}
		else if (ifRedHappened == false && isGreenColor == true){
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(30, 30, 30), FILLED);
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(0, 255, 0), FILLED);
		}
		else if (ifRedHappened == true && isGreenColor == true && isRedColor == true){
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(0, 0, 255), FILLED);
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(0, 255, 0), FILLED);
		}
		else if (ifRedHappened == true && isGreenColor == true && isRedColor == false){
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(30, 30, 30), FILLED);
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(0, 255, 0), FILLED);
		}
		else{
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 60), 39, Scalar(30, 30, 30), FILLED);
			circle(res_image, cvPoint(tr_light_x + 150, tr_light_y + 240), 39, Scalar(30, 30, 30), FILLED);
		}
		if (flag_alarm == 1){
			alarm_image.copyTo(res_image(Rect(750, 730, alarm_image.cols, alarm_image.rows)));
		}
		resize(frame, frame, Size(704, 480));
		frame.copyTo(res_image(Rect(50, 10, frame.cols, frame.rows)));
		resize(imageROI, imageROI, Size(633, 292));
		imageROI.copyTo(res_image(Rect(85, 580, imageROI.cols, imageROI.rows)));
		
		imshow("demo", res_image);
		framecount++;
		#if SAVE_VIDEO
			video.write(res_image);
		#endif
	}
	#if DISPLAYALL
		waitKey(1);
	#endif
	return 0;
}



