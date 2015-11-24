#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;


// MSER Mit Elipsen - Einzelne_Keypoint-Funktion in den Kommentaren
int main1(int argc, char *argv[])
{   //Bild wird geladen, MSER und Speichervariablen werden erstellt
	Mat image = imread("adam1.png", 0);
	Mat imageOut = imread("adam1.png", 0);
	Ptr<cv::MSER> mser = MSER::create();
	vector< vector<Point > > ptblobs;
	vector<Rect> bboxes;
	//MSER Detect wird ausgeführt 
	mser->detectRegions(image, ptblobs, bboxes);




	//imwrite("adamElipse.png", imageOut);
	//Bild anzeigen
	namedWindow("image");
	imshow("image", imageOut);
	waitKey(0);
	return -1;
};
///Umrechnung von Point -> Point2f für drawKeypoints
/*vector <KeyPoint >keypoints;
vector <Point2f> speicher;
for (int i = 0; i < ptblobs.size();i++)
{
for (int j = 0; j < ptblobs.at(i).size(); j++)
{
Point pt = ptblobs.at(i).at(j);
speicher.push_back(Point2f(pt.x, pt.y)); //hinzufügen von PointKoordinaten x und y zu Zwischenspeicher
}
};


//Umrechnung von Point2f -> Keypoints
//KeyPoint::convert(speicher, keypoints);
//zeichnen der Keypoints
//drawContours(imageOut)
//drawKeypoints(image, keypoints, imageOut);
*/

