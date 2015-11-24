#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class KeyPointProjektor
{
public:
	KeyPointProjektor();
	~KeyPointProjektor();
	vector<cv::RotatedRect> KeyPointProjektor::keyPointsProj( vector<vector<Point>>& ptblobs, vector<double> coordZ, Mat trans, Mat image, string name);
	vector<Rect> bboxes;
};

