#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int val;

enum Action { none, binarize, canny, lines, diff, harris, tenis };
Action action = none;

RNG rng(12345);

#define LEFT_ARROW 2424832
#define UP_ARROW 2490368
#define RIGHT_ARROW 2555904
#define DOWN_ARROW 2621440
#define MIN_IMG_NUMBER 1
#define MAX_IMG_NUMBER 15
#define STARTING_IMG_NUMBER 1

Mat img;
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
		action = tenis;
	else if (key == 'b')
		action = binarize;
	else if (key == 'c')
		action = canny;
	else if (key == 'h')
		action = harris;
	else if (key == 'l')
		action = lines;
	else if (key == 'd')
		action = diff;
	else if (key == 's')
		imwrite("test.png", result);
	else if (key != -1)
		printf("%d\n", key);
}

void processImage()
{
	//Process image using chosen settings
	if (action == none)
		result = img;
	else if (action == binarize)
	{
		Mat result_gray;
		inRange(img, Scalar(val, val, val), Scalar(255, 255, 255), result_gray);
		cvtColor(result_gray, result, CV_GRAY2BGR);
	}
	else if (action == tenis)
	{
		// Crop image
		cv::Rect myROI(ballCenter.x - roiSide, ballCenter.y - roiAbove, roiSide * 2, roiAbove + roiBelow);
		if (myROI.x < 0)
			myROI.x = 0;
		if (myROI.y < 0)
			myROI.y = 0;
		if (myROI.x + myROI.width > img.size().width)
			myROI.width = img.size().width - myROI.x;
		if (myROI.y + myROI.height > img.size().height)
			myROI.height = img.size().height - myROI.y;
		Mat croppedImage = img(myROI);

		// Thresholding image
		Mat gray, gray_bgr, bw;
		cvtColor(croppedImage, gray, CV_BGR2GRAY);
		cv::threshold(gray, bw, 60, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		// Plop back into original image
		croppedImage.setTo(Scalar(0, 0, 255), bw);
		result = img;

		//Set next ROI
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//find contours of filtered image using openCV findContours function
		findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

		int largestMomentIndex = 0;
		int largestMoment = 0;
		for (int i = 0; i < hierarchy.size(); i++)
		{
			Moments mm = moments((Mat)contours[i]);
			if (mm.m00 > largestMoment)
			{
				largestMomentIndex = i;
				largestMoment = mm.m00;
			}
		}

		Moments mm = moments((Mat)contours[largestMomentIndex]);
		double m00 = mm.m00;
		double m10 = mm.m10;
		double m01 = mm.m01;
		double centerX = (m10 / m00);
		double centerY = (m01 / m00);

		circle(croppedImage, Point(centerX, centerY), 4, Scalar(255, 0, 0));

		ballCenter.x += centerX - roiSide;
		ballCenter.y += centerY - roiAbove;
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
		Canny(detectedEdges, detectedEdges, val, val * 3, 3);
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
		Canny(detectedEdges, detectedEdges, val, val * 3, 3);

		Mat edgesC;
		cvtColor(detectedEdges, edgesC, CV_GRAY2BGR);

		vector<Vec4i> lines;
		HoughLinesP(detectedEdges, lines, 1, CV_PI / 180, 50, 50, 10);
		result = edgesC;
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
		}
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
		if (val < 1)
			val = 1;

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
		goodFeaturesToTrack(gray, corners, val, qualityLevel,
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
		printf("Location: %d %d\n", x, y);
}

int main(int argc, char** argv)
{
	VideoWriter VOut;
	imageNumber = STARTING_IMG_NUMBER;

	namedWindow("White", CV_WINDOW_AUTOSIZE);
	setMouseCallback("White", mouseCallback, NULL);
	createTrackbar("Threshold", "White", &val, 255, trackbarCallback);
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