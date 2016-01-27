#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int val[3];

enum Action { none, binarize, canny, lines, diff, harris, pretzel, otsu, hsvHue, hsvSat, hsvVal};
Action action = none;

RNG rng(12345);

#define LEFT_ARROW 2424832
#define UP_ARROW 2490368
#define RIGHT_ARROW 2555904
#define DOWN_ARROW 2621440
#define MIN_IMG_NUMBER 1
#define MAX_IMG_NUMBER 15
#define STARTING_IMG_NUMBER 1

Mat img, hsv;
int imageNumber;
Mat result, frame2;

Point2i ballCenter(260, 175);
int roiAbove = 20;
int roiSide = 70;
int roiBelow = 170;

void loadImage(int imageNumber)
{
	char name[20];
	sprintf(name, "img (%d).bmp", imageNumber);
	string str_name(name);
	img = imread(str_name);
}

void pressKey(int key)
{
	if (key == LEFT_ARROW)
	{
		imageNumber -= 1;
		if (imageNumber < MIN_IMG_NUMBER)
			imageNumber = MAX_IMG_NUMBER;
		loadImage(imageNumber);
	}
	else if (key == RIGHT_ARROW)
	{
		imageNumber += 1;
		if (imageNumber > MAX_IMG_NUMBER)
			imageNumber = MIN_IMG_NUMBER;
		loadImage(imageNumber);
	}
	else if (key == 'o' || key == 'n')
		action = none;
	else if (key == 't')
		action = pretzel;
	else if (key == 'b')
		action = binarize;
	else if (key == 'u')
		action = otsu;
	else if (key == 'c')
		action = canny;
	else if (key == 'l')
		action = lines;
	else if (key == 'd')
		action = diff;
	else if (key == 'h')
		action = hsvHue;
	else if (key == 's')
		action = hsvSat;
	else if (key == 'v')
		action = hsvVal;
	//else if (key == 's')
	//	imwrite("test.png", result);
	else if (key != -1)
		printf("%d\n", key);
}

Point2f calcLineIntersection(Size imgSize, float rho, float theta, bool vertical, bool dir)
{
	int dirInt = dir ? 1 : -1;
	float dy1 = -cos(theta)*dirInt;
	float dx1 = sin(theta)*dirInt;
	Point2f point(rho*cos(theta), rho*sin(theta));
	if (vertical)
	{
		float dx2 = (dx1 < 0 ? 0 : imgSize.width) - point.x;
		int dy2 = dy1 * dx2 / dx1;
		return Point(point.x + dx2, point.y + dy2);
	}
	else
	{
		float dy2 = (dy1 < 0 ? 0 : imgSize.height) - point.y;
		int dx2 = dx1 * dy2 / dy1;
		return Point(point.x + dx2, point.y + dy2);
	}
}

Point calcRectIntersection(Size imgSize, float rho, float theta, bool dir)
{
	Point p1 = calcLineIntersection(imgSize, rho, theta, true, dir);
	if (p1.x >= 0 && p1.y >= 0 && p1.x <= imgSize.width && p1.y <= imgSize.height)
		return p1;
	else
		return calcLineIntersection(imgSize, rho, theta, false, dir);
}

void calcCrop(Size& imgSize, vector<Vec2f>& lines, vector<Point2f>& pts, Rect& roi)
{
	Mat edgesC;
	int roiX1 = 0;
	int roiX2 = imgSize.width;
	for (int i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point2f p1 = calcRectIntersection(img.size(), rho, theta, true);
		Point2f p2 = calcRectIntersection(img.size(), rho, theta, false);
		if (p1.x < imgSize.width / 2)
		{
			if (p1.x > roiX1)
				roiX1 = p1.x;
		}
		else
			if (p1.x < roiX2)
				roiX2 = p1.x;

		if (p2.x < imgSize.width / 2)
		{
			if (p2.x > roiX1)
				roiX1 = p2.x;
		}
		else
			if (p2.x < roiX2)
				roiX2 = p2.x;
		pts.push_back(p1);
		pts.push_back(p2);
	}
	Rect roi2(roiX1, 0, roiX2 - roiX1, img.size().height);
	roi = roi2;
}

void drawCrop(Mat& img, vector<Point2f> pts, Rect& roi)
{
	for (int i = 0; i < pts.size(); i += 2)
	{
		line(img, pts[i], pts[i + 1], pts[i].x > img.size().width / 2 ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 1, CV_AA);
	}
	rectangle(img, roi, Scalar(0, 255, 0), 1, CV_AA);
}

void getChannel(Mat& img, Mat& imgChannel, int channel)
{
	Mat channels[3];
	split(img, channels);
	imgChannel = channels[channel];
}

void setHsv(Mat& before, Mat& hsv, Mat& after, int channel)
{
	vector<Mat> channels;
	cvtColor(before, hsv, CV_BGR2HSV);
	split(hsv, channels);
	if (channel != 0)
		channels[0].setTo(179);
	if (channel != 1)
		channels[1].setTo(255);
	if (channel != 2)
		channels[2].setTo(255);
	merge(channels, hsv);
	cvtColor(hsv, after, CV_HSV2BGR);
}

void hueDist(Mat& hsv, Mat& dist, int hueVal)
{
	Mat hue;
	getChannel(hsv, hue, 0);
	absdiff(hue, hueVal, dist);
}

void processImage()
{
	//Process image using chosen settings
	if (action == none)
		result = img;
	else if (action == binarize)
	{
		Mat result_gray, hsv;
		cvtColor(img, hsv, CV_BGR2HSV);
		inRange(hsv, Scalar(val[0], val[1], val[2]), Scalar(255, 255, 255), result_gray);
		cvtColor(result_gray, result, CV_GRAY2BGR);
	}
	else if (action == hsvHue)
		setHsv(img, hsv, result, 0);
	else if (action == hsvSat)
		setHsv(img, hsv, result, 1);
	else if (action == hsvVal)
		setHsv(img, hsv, result, 2);
	else if (action == otsu)
	{
		// Thresholding image
		Mat gray, bw;
		cvtColor(img, gray, CV_BGR2GRAY);
		cv::threshold(gray, bw, 60, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		result = bw;
	}
	else if (action == pretzel)
	{
		cvtColor(img, hsv, CV_BGR2HSV);
		hueDist(hsv, result, 16);

		


		////find contours of filtered image using openCV findContours function
		//findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		//int largestMomentIndex = 0;
		//int largestMoment = 0;
		//for (int i = 0; i < hierarchy.size(); i++)
		//{
		//	Moments mm = moments((Mat)contours[i]);
		//	if (mm.m00 > largestMoment)
		//	{
		//		largestMomentIndex = i;
		//		largestMoment = mm.m00;
		//	}
		//}

		//Moments mm = moments((Mat)contours[largestMomentIndex]);
		//double m00 = mm.m00;
		//double m10 = mm.m10;
		//double m01 = mm.m01;
		//double centerX = (m10 / m00);
		//double centerY = (m01 / m00);

		//circle(croppedImage, Point(centerX, centerY), 4, Scalar(255, 0, 0));

		//ballCenter.x += centerX - roiSide;
		//ballCenter.y += centerY - roiAbove;
	}
	else if (action == canny)
	{
		// Convert the image to grayscale
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		Mat detectedEdges;
		// Reduce noise with a kernel 3x3
		blur(gray, detectedEdges, Size(3, 3));

		// Canny detector
		Canny(detectedEdges, detectedEdges, val[0], val[0] * 3, 3);
		result = detectedEdges;
	}
	else if (action == lines)
	{
		/// Convert the image to grayscale
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		Mat detectedEdges;
		/// Reduce noise with a kernel 3x3
		blur(gray, detectedEdges, Size(3, 3));

		/// Canny detector
		int val2 = 30;
		Canny(detectedEdges, detectedEdges, val2, val2 * 3, 3);

		vector<Vec2f> lines;
		HoughLines(detectedEdges, lines, 1, CV_PI / 180, 150);
		Mat edgesC;
		cvtColor(detectedEdges, edgesC, CV_GRAY2BGR);
		result = edgesC;

		vector<Point2f> pts;
		Rect roi;
		calcCrop(img.size(), lines, pts, roi);

		drawCrop(result, pts, roi);

		//Mat croppedImg(img, roi);
		//result = croppedImg;		
	}
	else if (action == diff)
	{
		if (frame2.data)
		{
			result = img.clone(); // Mat::zeros(frame.size(), frame.type());
			absdiff(img, frame2, result);
		}
		frame2 = img.clone();
	}
	else if (action == harris)
	{
		/// Convert the image to grayscale
		Mat gray;
		Mat dst, dst_norm, dst_norm_scaled;
		cvtColor(img, gray, CV_BGR2GRAY);
		if (val[0] < 1)
			val[0] = 1;

		/// Parameters for Shi-Tomasi algorithm
		vector<Point2f> corners;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3;
		bool useHarrisDetector = false;
		double k = 0.04;

		/// Copy the source image
		Mat copy;
		copy = img.clone();

		/// Apply corner detection
		goodFeaturesToTrack(gray, corners, val[0], qualityLevel,
			minDistance, Mat(), blockSize, useHarrisDetector, k);

		/// Draw corners detected
		int r = 4;
		for (int i = 0; i < corners.size(); i++)
		{
			circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255)), -1, 8, 0);
		}
		result = copy;

		/// Set the neeed parameters to find the refined corners
		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);

		/// Calculate the refined corner locations
		cornerSubPix(gray, corners, winSize, zeroZone, criteria);

		/// Write them down
		//for (int i = 0; i < corners.size(); i++)
		//{
		//	cout << " -- Refined Corner [" << i << "]  (" << corners[i].x << "," << corners[i].y << ")" << endl;
		//}
	}
	imshow("White", result);
}

void trackbarCallback(int, void*)
{
	processImage();

}

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (hsv.empty())
		{
			printf("empty\n");
			return;
		}
		Vec3b v = hsv.at<Vec3b>(y, x);
		printf("%d %d %d\n", v[0], v[1], v[2]);
		Mat dist;
		hueDist(hsv, dist, hsv.at<Vec3b>(y, x)[0]);
		imshow("White", dist);
	}
}

int main(int argc, char** argv)
{
	VideoWriter VOut;
	imageNumber = STARTING_IMG_NUMBER;

	namedWindow("White", CV_WINDOW_AUTOSIZE);
	setMouseCallback("White", mouseCallback, NULL);
	createTrackbar("H Threshold", "White", &val[0], 255, trackbarCallback);
	createTrackbar("S Threshold", "White", &val[1], 255, trackbarCallback);
	createTrackbar("V Threshold", "White", &val[2], 255, trackbarCallback);
	loadImage(imageNumber);
	while (true)
	{
		//Check for keyboard input
		int key = waitKey(0);
		if (key == 27) //Escape
			return 0;
		else
			pressKey(key);
		processImage();
	}


	//if (!VOut.isOpened())
	//{
	//	VOut.open("VideoOut.avi", CV_FOURCC('M', 'P', 'E', 'G'), 10, img.size(), 1);
	//	printf("Opening video stream with size %d %d\n", img.size().width, img.size().height);
	//}
	//VOut << img;
}