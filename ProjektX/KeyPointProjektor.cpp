#include "KeyPointProjektor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Projektion.h"
#include "Speicher.h"

using namespace cv;
using namespace std;


KeyPointProjektor::KeyPointProjektor()
{
}


KeyPointProjektor::~KeyPointProjektor()
{
}

/* Übergeben werden muss:
vector<Rect> bboxes;
Mat coordZ = imgProj.at(k).sizeBerechnen(coord1, coord2, coord3, coord4, imgProj.at(k).trans);

*/
vector<cv::RotatedRect> KeyPointProjektor::keyPointsProj( vector<vector<Point>>& ptblobs, vector<double> coordZ, Mat trans,Mat image,string name)
{
	Mat tlt = (Mat_<double>(3, 3) <<
		1, 0, (0 - coordZ.at(0)),
		0, 1, (0 - coordZ.at(2)),
		0, 0, 1);
	string ad = "";
	Mat img;int i;
	RotatedRect box;
	Speicher Speicher;
	Projektion Proj;
	vector<cv::RotatedRect>bboxes;
	vector<vector<Point2f>>transfPtBlobs;
	transfPtBlobs.resize(ptblobs.size());

	Speicher.SetFolder("praktikum");
	Speicher.verzeichnis = "C:\\praktikum\\";
	bboxes.empty();
	//für jedes FeatureFeld
	for (i = 0; i < ptblobs.size(); i++)
	{
		ad = "k"+to_string(i);		
		transfPtBlobs.at(i) = Proj.PunkteVerschieben(ptblobs.at(i), trans); // aus jedem Vector<Punkt> wird eine Vector<Mat> und projeziert diesen
		box = minAreaRect(Mat(transfPtBlobs.at(i))); //Vector<Punkt> werden zu einem RotRect
		bboxes.push_back(box);
		getRectSubPix(image, box.size, box.center, img); //RotRect wird zu Mat		
		Speicher.Save(img, ad, ad + name); //Mat wird gespeichert
		
	}

	return bboxes;

}
 /*
 
		Mat coord1 = (Mat_ <double>(3, 1) << bboxes.at(i).x, bboxes.at(i).y, 1);
		Mat coord2 = (Mat_ <double>(3, 1) << bboxes.at(i).x, bboxes.at(i).y + bboxes.at(i).height, 1);
		Mat coord3 = (Mat_ <double>(3, 1) << bboxes.at(i).x + bboxes.at(i).width, bboxes.at(i).y, 1);
		Mat coord4 = (Mat_ <double>(3, 1) << bboxes.at(i).x + bboxes.at(i).width, bboxes.at(i).y + bboxes.at(i).height, 1);
		
		coord1 = Proj.PunktVerschieben(coord1, trans);
		coord2 = Proj.PunktVerschieben(coord2, trans);
		coord3 = Proj.PunktVerschieben(coord3, trans);
		coord4 = Proj.PunktVerschieben(coord4, trans);
		double EckpunkteX[4] = { coord1.at<double>(0, 0) , coord2.at<double>(0, 0) , coord3.at<double>(0, 0) , coord4.at<double>(0, 0) };
		double EckpunkteY[4] = { coord1.at<double>(1, 0) , coord2.at<double>(1, 0) , coord3.at<double>(1, 0) , coord4.at<double>(1, 0) };
		xmin = EckpunkteX[0];
		xmax = EckpunkteX[0];
		ymin = EckpunkteY[0];
		ymax = EckpunkteY[0];
		for (int i = 1; i < 4; i++)
		{
			// größtes X finden
			xmax = max(EckpunkteX[i], xmax);
			// größtes Y finden
			ymax = max(EckpunkteY[i], ymax);
			// kleinstes X finden
			xmin = min(EckpunkteX[i], xmin);
			// kleinstes Y finden
			ymin = min(EckpunkteY[i], ymin);
		}
		
		cout << "xmin: " << xmin << "ymin: " << ymin << "xamx: " << xmax << "ymax: " << ymax  << endl;
		std::cout << std::endl;
		
		
		cout << "xmin: " << xmin << " ymin: " << ymin << " xamx: " << xmax << " ymax: " << ymax << " xmax-xmin "<< xmax - xmin << " ymax-ymin " << ymax - ymin <<" imagesize: "<<image.size()<< endl;
		img.size() = Size(xmax-xmin,ymax-ymin);
		flag = (xmin >= 0) && (ymin >= 0) && (ymax <= image.size().height) && (xmax <= image.size().width);
		//cout << "Groeße des Zwischenspeichers anpassen" << endl;
		if(flag)
		img = image(Rect(xmin, ymin, xmax - xmin , ymax - ymin ));
		else cout << "KOORDINATEN SIND FEHLERHAFT" << endl;
		std::cout << std::endl;

		
 */