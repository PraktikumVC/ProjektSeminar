#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "Mathe.h"


using namespace cv;

class Projektion
{
public:
	
	Projektion();
	~Projektion();
	Mat bildRotieren(Mat image, double alpha, double beta, double gamma);
	Mat bildRotieren(Mat image, double theta, double phi);
	Mat matrixErrechnen(double alpha, double beta, double gamma);
	std::vector<double> Projektion::sizeBerechnen(Mat ol, Mat ul, Mat orr, Mat ur,Mat transL);
	Mat Projektion::PunktVerschieben(Mat punkt,Mat trans);
	Mat Projektion::PunktVerschieben(Mat punkt, Mat trans, Mat transL); // Überladen
	std::vector <Mat> Projektion::PunkteVerschieben(std::vector<Mat> punkte, Mat trans);
	std::vector <Point2f> Projektion::PunkteVerschieben(std::vector<Point> punkte, Mat trans);
	//BildInput
	Mat image;
	//BildOutput
	Mat imageOut;
	//Transformationsmatrix
	Mat trans;

private:	
	//Objekt für mathem. Berechnungen
	Mathe M; 
	//Größe des BildInputs
	Size size; 
	//Größe des BildOutputs
	Size sizeOut;
	double  xmin, xmax, ymin, ymax ; // Minimale und maximale X und Y Werte der Eckpunkte des Bildes

};