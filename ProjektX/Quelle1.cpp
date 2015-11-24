#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <ctime>

using namespace std;	// Erleichterung durch Weglassen von Header std
using namespace cv;	// Erleichterung durch Weglassen von Header cv


// dandan V1

int main1(int argc, char** argv)
{
	std::cout << "OpenCV Version: " // Ausgabe der Version von OpenCV
		<< CV_MAJOR_VERSION << "."
		<< CV_MINOR_VERSION << "."
		<< CV_SUBMINOR_VERSION
		<< std::endl;

	Mat img = imread("adam1.png"); // Eingelesenes Originalbild

	if (img.empty())	//Abfrage ob eingelesenes Bild leer ist
	{
		cout << "Kann Bild nicht öffnen!" << endl; // Fehlermeldung bei nicht einlesbarem Bild
		return -1;
	}

	

	Mat gray; // Graubild des Eingelesenen Bildes
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);  // Konvertierung von eingelesenem Bild in Graubild

	int minArea = 60;	// MSER Eingabevariable: minimaler Bereich
	int maxArea = 14400;	// MSER Eingabevariable: maximaler Bereich
	int delta = 5;	// MSER Eingabevariable: 
	double variation = 0.25;	// MSER Eingabevariable:
	double diversity = 0.2;	// MSER Eingabevariable:
	int maxEvo = 200;	// MSER Eingabevariable:
	double arTh = 1.01;	// MSER Eingabevariable:
	double minMar = 0.003;	// MSER Eingabevariable:
	int ebs = 5;
	vector< vector< Point > > ptblobs; //Speicherortes für die Punkte der gefundenen Blobs 
		//"a blob is a region of an image in which some properties are constant or approximately constant"
	vector<Rect> bboxes;	// Speicherort für Rechtecke


	static Ptr<MSER> mser = MSER::create(delta, minArea, maxArea, variation, diversity, maxEvo, arTh, minMar, ebs); //  MSER- Detektor
	mser->detectRegions(gray, ptblobs, bboxes);

	Mat img_regions;	// Bild mit markierten MSER Regionen
	
	//ellipse()

		cout << "MSER Berechnung abgeschlossen" << endl;

		namedWindow("Originalbild");
		imshow("Originalbild", img);
		//namedWindow("MSER");
		//imshow("MSER", img_regions);
		waitKey(0);

		return 0;
	}