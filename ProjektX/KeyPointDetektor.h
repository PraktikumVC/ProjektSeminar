#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

class KeyPointDetektor
{
public:
	KeyPointDetektor();
	~KeyPointDetektor();
	bool KeyPointDetektor::Mser(cv::Mat image);
	std::vector <cv::Rect> bboxes;
};

