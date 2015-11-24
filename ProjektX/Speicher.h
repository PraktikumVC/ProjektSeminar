#pragma once
//#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

class Speicher
{
public:
	Speicher();
	~Speicher();
	bool Speicher::Save(cv::Mat img1, std::string ordner, std::string uordner);
	bool Speicher::Save(cv::Mat img1, cv::Mat img2, std::string ordner, std::string uordner);
	bool SetFolder(std::string ordner);
	std::string verzeichnis= "C:\\";
	
};

