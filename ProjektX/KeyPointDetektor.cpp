#include "KeyPointDetektor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;


KeyPointDetektor::KeyPointDetektor()
{
}


KeyPointDetektor::~KeyPointDetektor()
{
}
//MSER

bool  KeyPointDetektor::Mser(cv::Mat image)
{
	cv::Ptr<cv::MSER> mser = cv::MSER::create();
	vector< vector<cv::Point > > ptblobs;
	//MSER Detect wird ausgeführt 
	mser->detectRegions(image, ptblobs, bboxes);
	std::cout << "testo" << endl;
	return true;
}

//SIFT