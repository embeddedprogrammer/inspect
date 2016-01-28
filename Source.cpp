#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int val[3];

enum GrayAction { g_none, g_cvt, g_hue, g_sat, g_val, g_hue_val};
enum BinarizeAction { b_none, b_threshold, b_rgb, b_hsv, b_otsu };
enum FeatureAction { f_none, f_binarize, f_canny, f_action_lines, f_diff, f_harris, f_pretzel, f_otsu, f_hsvHue, f_hsvSat, f_hsvVal, f_contours};
GrayAction g_action = g_none;
BinarizeAction b_action = b_none;
FeatureAction f_action = f_none;
bool displayHist;
bool gaussianSmoothing;
bool erodeDilate;


RNG rng(12345);

#define ENTER 13
#define LEFT_ARROW 2424832
#define UP_ARROW 2490368
#define RIGHT_ARROW 2555904
#define DOWN_ARROW 2621440
#define MIN_IMG_NUMBER 1
#define MAX_IMG_NUMBER 24
#define STARTING_IMG_NUMBER 1

Mat img, imgHsv;
int imageNumber;

Point2i ballCenter(260, 175);
int roiAbove = 20;
int roiSide = 70;
int roiBelow = 170;

void processImage();

void loadImage(Mat& img, int imageNumber)
{
	char name[20];
	sprintf(name, "img (%d).bmp", imageNumber);
	string str_name(name);
	img = imread(str_name);
}

void showHistogram(Mat& img)
{
	int bins = 256;             // number of bins
	int nc = img.channels();    // number of channels

	vector<Mat> hist(nc);       // histogram arrays

								// Initalize histogram arrays
	for (int i = 0; i < hist.size(); i++)
		hist[i] = Mat::zeros(1, bins, CV_32SC1);

	// Calculate the histogram of the image
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				uchar val = nc == 1 ? img.at<uchar>(i, j) : img.at<Vec3b>(i, j)[k];
				hist[k].at<int>(val) += 1;
			}
		}
	}

	// For each histogram arrays, obtain the maximum (peak) value
	// Needed to normalize the display later
	int hmax[3] = { 0,0,0 };
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < bins - 1; j++)
			hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
	}

	const char* wname[3] = { "blue", "green", "red" };
	Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

	vector<Mat> canvas(nc);

	// Display each histogram in a canvas
	for (int i = 0; i < nc; i++)
	{
		canvas[i] = Mat::ones(125, bins, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < bins - 1; j++)
		{
			line(
				canvas[i],
				Point(j, rows),
				Point(j, rows - (hist[i].at<int>(j) * rows / hmax[i])),
				nc == 1 ? Scalar(200, 200, 200) : colors[i],
				1, 8, 0
				);
		}

		imshow(nc == 1 ? "value" : wname[i], canvas[i]);
	}
}

string cmd = "";

void pressKey(int key)
{
	if (key == LEFT_ARROW)
	{
		imageNumber -= 1;
		if (imageNumber < MIN_IMG_NUMBER)
			imageNumber = MAX_IMG_NUMBER;
	}
	else if (key == RIGHT_ARROW)
	{
		imageNumber += 1;
		if (imageNumber > MAX_IMG_NUMBER)
			imageNumber = MIN_IMG_NUMBER;
	}
	else if (key == ENTER)
	{
		// Grayscale commands
		if (cmd == "gn")
			g_action = g_none;
		else if (cmd == "gh")
			g_action = g_hue;
		else if (cmd == "gs")
			g_action = g_sat;
		else if (cmd == "gv")
			g_action = g_val;
		else if (cmd == "gc")
			g_action = g_cvt;
		else if (cmd == "ghv")
			g_action = g_hue_val;

		// Binarization commands
		else if (cmd == "bn")
			b_action = b_none;
		else if (cmd == "bt")
			b_action = b_threshold;
		else if (cmd == "brgb")
			b_action = b_rgb;
		else if (cmd == "bhsv")
			b_action = b_hsv;
		else if (cmd == "bo")
			b_action = b_otsu;

		// Smoothing commands
		else if (cmd == "ed")
			erodeDilate = !erodeDilate;
		else if (cmd == "g")
			gaussianSmoothing = !gaussianSmoothing;

		// Feature commands
		else if (cmd == "none")
			f_action = f_none;
		else if (cmd == "c")
			f_action = f_contours;
		else if (cmd == "t")
			f_action = f_pretzel;
		else if (cmd == "b")
			f_action = f_binarize;
		else if (cmd == "u")
			f_action = f_otsu;
		else if (cmd == "c")
			f_action = f_canny;
		else if (cmd == "l")
			f_action = f_action_lines;
		else if (cmd == "d")
			f_action = f_diff;
		else if (cmd == "h")
			f_action = f_hsvHue;
		else if (cmd == "s")
			f_action = f_hsvSat;
		else if (cmd == "v")
			f_action = f_hsvVal;
		else if (cmd == "save")
			imwrite("test.png", img);
		else if (cmd == "hist")
			displayHist = !displayHist;

		cmd = "";
		cout << endl << ">";
	}
	else if (key >= 'a' && key <= 'z')
	{
		cmd = cmd + ((char)key);
		cout << ((char)key);
	}
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
		Point2f p1 = calcRectIntersection(imgSize, rho, theta, true);
		Point2f p2 = calcRectIntersection(imgSize, rho, theta, false);
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
	Rect roi2(roiX1, 0, roiX2 - roiX1, imgSize.height);
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

void channelDist(Mat& hsv, Mat& dist, int hueVal, int channel)
{
	Mat channelImg;
	getChannel(hsv, channelImg, channel);
	if (channel == 0)
	{
		Mat dist1, dist2;
		absdiff(channelImg, hueVal, dist1);
		absdiff(channelImg, hueVal + ((hueVal < 90) ? 180 : -180), dist2);
		min(dist1, dist2, dist);
	}
	else
		absdiff(channelImg, hueVal, dist);
}

void processImage()
{
	static Mat frame2;
	loadImage(img, imageNumber);

// Crop the image
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

	vector<Point2f> pts;
	Rect roi;
	calcCrop(img.size(), lines, pts, roi);

	Mat img2(img, roi);

	if (imageNumber != 16)
		img = img2;

//Apply grayscale
	if (g_action == g_hue)
	{
		cvtColor(img, imgHsv, CV_BGR2HSV);
		channelDist(imgHsv, img, 16, 0);
		bitwise_not(img, img);
	}
	else if (g_action == g_sat)
	{
		cvtColor(img, imgHsv, CV_BGR2HSV);
		getChannel(imgHsv, img, 1);
	}
	else if (g_action == g_val)
	{
		cvtColor(img, imgHsv, CV_BGR2HSV);
		getChannel(imgHsv, img, 2);
	}
	else if (g_action == g_hue_val)
	{
		Mat d0, d2;
		cvtColor(img, imgHsv, CV_BGR2HSV);
		channelDist(imgHsv, d0, 16, 0);
		bitwise_not(d0, d0);
		getChannel(imgHsv, d2, 2);
		addWeighted(d0, .5, d2, .5, 0, img);
	}

//Gaussian Smoothing
	if(gaussianSmoothing && img.channels() == 1)
		GaussianBlur(img, img, Size(3, 3), 0);

//Binarize
	if(b_action == b_threshold && img.channels() == 1)
		cv::threshold(img, img, val[0], 255, CV_THRESH_BINARY);
	else if (b_action == b_rgb && img.channels() == 3)
		inRange(img, Scalar(val[0], val[1], val[2]), Scalar(255, 255, 255), img);
	else if (b_action == b_hsv && img.channels() == 3)
	{
		cvtColor(img, imgHsv, CV_BGR2HSV);
		inRange(imgHsv, Scalar(val[0], val[1], val[2]), Scalar(255, 255, 255), img);
	}
	else if (b_action == b_otsu && img.channels() == 1)
		cv::threshold(img, img, 60, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

//Erode and Dilate
	if(erodeDilate && img.channels() == 1)
	{
		erode(img, img, getStructuringElement(MORPH_CROSS, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2)));
		dilate(img, img, getStructuringElement(MORPH_CROSS, Size(2 * 4 + 1, 2 * 4 + 1), Point(4, 4)));
	}

//Other actions
	if (f_action == f_pretzel && img.channels() == 1)
	{
		//Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		img = Mat::zeros(img.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
			drawContours(img, contours, i, Scalar(255, 0, 0));

		// Find parent contour
		int pretzelContourIndex = -1;
		int minPretzelSize = 1000;
		int largestContourSize = minPretzelSize;
		for (int i = 0; i < hierarchy.size(); i++)
		{
			if (hierarchy[i][3] == -1)
			{
				Moments mm = moments((Mat)contours[i]);
				if (mm.m00 > largestContourSize)
				{
					pretzelContourIndex = i;
					largestContourSize = mm.m00;
					printf("Size: %d\n", (int)mm.m00);
				}
			}
		}
		int pretzelSize = largestContourSize;

		// Evaluate pretzel based on contour children
		int minHoleSize = 10;
		int pass = -1;
		if (pretzelContourIndex != -1)
		{
			// Find center of mass
			Moments mm = moments((Mat)contours[pretzelContourIndex]);
			double centerX = (mm.m10 / mm.m00);
			double centerY = (mm.m01 / mm.m00);
			circle(img, Point(centerX, centerY), 4, Scalar(0, 255, 0));

			int borderSize = 100;
			if (centerY > borderSize && centerY < img.size().height - borderSize)
			{
				int numberOfHoles = 0;
				int child = hierarchy[pretzelContourIndex][2];
				while (child != -1)
				{
					if (contours[child].size() > minHoleSize)
						numberOfHoles++;
					child = hierarchy[child][0];
				}
				if (numberOfHoles <= 1)
					pass = 2;
				else if (numberOfHoles == 2)
					pass = 1;
				else if (numberOfHoles == 3)
					pass = 0;
			}
		}

		if (pass == -1)
			printf("None\n");
		else if (pass == 0)
			printf("Good\n");
		else if (pass == 1)
			printf("Bad\n");
		else if (pass == 2)
			printf("Ugly\n");
	}
	else if (f_action == f_hsvHue && img.channels() == 3)
		setHsv(img, imgHsv, img, 0);
	else if (f_action == f_hsvSat && img.channels() == 3)
		setHsv(img, imgHsv, img, 1);
	else if (f_action == f_hsvVal && img.channels() == 3)
		setHsv(img, imgHsv, img, 2);
	else if (f_action == f_canny)
	{
		// Convert the image to grayscale
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);

		Mat detectedEdges;
		// Reduce noise with a kernel 3x3
		blur(gray, detectedEdges, Size(3, 3));

		// Canny detector
		Canny(detectedEdges, detectedEdges, val[0], val[0] * 3, 3);
		img = detectedEdges;
	}
	else if (f_action == f_action_lines)
	{
		img = edgesC;
		drawCrop(img, pts, roi);
	}
	else if (f_action == f_diff)
	{
		if (frame2.data)
		{
			img = img.clone(); // Mat::zeros(frame.size(), frame.type());
			absdiff(img, frame2, img);
		}
		frame2 = img.clone();
	}
	else if (f_action == f_harris)
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
		goodFeaturesToTrack(gray, corners, val[0], qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

		/// Draw corners detected
		int r = 4;
		for (int i = 0; i < corners.size(); i++)
		{
			circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255)), -1, 8, 0);
		}
		img = copy;

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
	imshow("White", img);
	if(displayHist)
		showHistogram(img);
}

void trackbarCallback(int, void*)
{
	processImage();

}

void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		if (imgHsv.empty())
		{
			printf("empty\n");
			return;
		}
		Vec3b v = imgHsv.at<Vec3b>(y, x);
		printf("%d %d %d\n", v[0], v[1], v[2]);
		//Mat dist;
		//channelDist(imgHsv, dist, imgHsv.at<Vec3b>(y, x)[0], 0);
		//imshow("White", dist);
	}
}

int main(int argc, char** argv)
{
	VideoWriter VOut;
	imageNumber = STARTING_IMG_NUMBER;

	namedWindow("White", CV_WINDOW_AUTOSIZE);
	setMouseCallback("White", mouseCallback, NULL);
	createTrackbar("H", "White", &val[0], 255, trackbarCallback);
	createTrackbar("S", "White", &val[1], 255, trackbarCallback);
	createTrackbar("V", "White", &val[2], 255, trackbarCallback);
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