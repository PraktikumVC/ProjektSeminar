#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include "Projektion.h"
#include "Speicher.h"
#include "KeyPointDetektor.h"
#include "KeyPointProjektor.h"




using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
		//Variablen für Test erzeugen
		Mat image = imread("adam1.png", 0);
		double alpha = 10; double beta = 90; double gamma = 90; double theta = 10; double phi = 90; string buf;
		Projektion Proj; Speicher Speicher; Mat imageOut = image; Mathe Mathe; KeyPointProjektor KPProj;
		Mat winkels = (Mat_<double>(5, 2) << 0,0, 20,90, 30,180, 30,270, 10,140); // Mat für Winkel

		Mat img;													//Zwischenspeicher
		Mat coord1 =(Mat_ <double>(3,1) << 0, 0, 1);				//Zwischenspeicher
		Mat coord2 = (Mat_ <double>(3, 1) << 0, 0, 1);				//Zwischenspeicher
		Mat coord3 = (Mat_ <double>(3, 1) << 0, 0, 1);				//Zwischenspeicher
		Mat coord4 = (Mat_ <double>(3, 1) << 0, 0, 1);				//Zwischenspeicher
		Mat trans;
		vector<double> coordZ;										//Zwischenspeicher
		vector<Mat> images;											//alle proj. Bildern
		vector<String>imagenName;									//Bildnamen von allen proj. Bildern
		vector<Mat> imgTrans;									//alle Matrizen zu den proj. Bildern		
		int i;  int k=1;								        	//Zählvariablen für Schleifen

		//bildnamen aus Parametern erstellen
		string s1 = Mathe.WinkelZuString(alpha, beta, gamma);
		string s2 = Mathe.WinkelZuString(theta, phi);
		string ad = "adam1";

		
		Ptr<cv::MSER> mser = MSER::create();
		vector< vector<Point > > ptblobs;
		vector<Rect> bboxes;
		vector<RotatedRect>bboxez;
		//MSER Detect wird ausgeführt 
		mser->detectRegions(image, ptblobs, bboxes);

		//für jeden Winkel
		for (i = 0; i <winkels.size().height;i++)
		{
			//bilder drehen, wichtige Teile speichern
			images.push_back( Proj.bildRotieren(image, winkels.at<double>(i, 0), winkels.at<double>(i, 1)) );					
			imgTrans.push_back(Proj.trans.clone());
			imagenName.push_back(Mathe.WinkelZuString(winkels.at<double>(i, 0), winkels.at<double>(i, 1)));
						
			Speicher.Save(images.at(i), "test", imagenName.at(i)); //Abspeichern
		}
		//für jedes proj. Bild
		for (k = 0; k < images.size(); k++)
		{			
			buf = imagenName.at(k);
			coord2.at<double>(1, 0) = images.at(k).size().height;
			coord3.at<double>(0, 0) = images.at(k).size().width;
			coord4.at<double>(0, 0) = images.at(k).size().width;
			coord4.at<double>(1, 0) = images.at(k).size().height;
			bboxez=KPProj.keyPointsProj(ptblobs, Proj.sizeBerechnen(coord1, coord2, coord3, coord4, imgTrans.at(k)) , imgTrans.at(k), images.at(k), buf);
			//für jede BoundingBox
			for (int j = 0; j < bboxez.size(); j++)
			{
				
				ellipse(images.at(k), bboxez.at(j), (255, 255, 255));
			}
			Speicher.Save(images.at(k), "testo", imagenName.at(k));
		}

		
		//bild erzeugen
		//imageOut = Proj.bildRotieren(image, alpha,beta,gamma);
		//imageOut = Proj.bildRotieren(image, theta, phi);


		//Speichern
		//Speicher.Save(imageOut, image,ad,s2);
	
				
		//extra Anzeige erzeugen und Bild darstellen
		namedWindow("image");
		imshow("image", imageOut);
		waitKey(0);			
		/*
		*/

}

