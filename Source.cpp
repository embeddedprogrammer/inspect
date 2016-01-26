#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

#define LEFT_ARROW 2424832
#define RIGHT_ARROW 2555904

Mat img;
int imageNumber;
Mat result;

Point2i ballCenter(260, 175);
int roiAbove = 20;
int roiSide = 70;
int roiBelow = 170;

void loadImage(int imageNumber)
{
	char name[20];
	sprintf(name, "image%03d.jpg", imageNumber);
	string str_name(name);
	img = imread(str_name);
}

void pressKey(int key)
{
	if (key == LEFT_ARROW)
	{
		imageNumber -= 2;
		if (imageNumber < 34)
			imageNumber = 72;
		loadImage(imageNumber);
	}
	else if (key == RIGHT_ARROW)
	{
		imageNumber += 2;
		if (imageNumber > 72)
			imageNumber = 34;
		loadImage(imageNumber);
	}
}

void processImage()
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

	// Find contours
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	// Find contour with largest area
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

	// Find center of mass
	Moments mm = moments((Mat)contours[largestMomentIndex]);
	double centerX = (mm.m10 / mm.m00);
	double centerY = (mm.m01 / mm.m00);

	// Move the center of the next frame ROI to the ball's current position
	circle(croppedImage, Point(centerX, centerY), 4, Scalar(255, 0, 0));
	ballCenter.x += centerX - roiSide;
	ballCenter.y += centerY - roiAbove;
	imshow("White", result);
}

int main(int argc, char** argv)
{
	VideoWriter VOut;
	imageNumber = 34;

	namedWindow("White", CV_WINDOW_AUTOSIZE);
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
		if (!VOut.isOpened())
		{
			VOut.open("VideoOut.avi", CV_FOURCC('M', 'P', 'E', 'G'), 10, img.size(), 1);
			printf("Opening video stream with size %d %d\n", img.size().width, img.size().height);
		}
		VOut << result;
	}
}