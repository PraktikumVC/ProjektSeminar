#include "Projektion.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include "Mathe.h"
#include <iostream>

using namespace cv;

Projektion::Projektion()
{	
}

Projektion::~Projektion()
{
}

/*
input: the image that you want rotated.
output: the Mat object to put the resulting file in.
alpha: the rotation around the x axis
beta: the rotation around the y axis
gamma: the rotation around the z axis (basically a 2D rotation)
*/

Mat Projektion::bildRotieren(Mat img, double alpha, double beta, double gamma )
{
	//Setzen der globalen Variablen
	image = img;
	size = image.size();
	sizeOut = size;

	//Hilfsfunktionen für Berechnung der Matrix und der Bildgröße
	trans = matrixErrechnen(alpha, beta, gamma);
	
	//Anwenden von warpPerspective mit errechneter Matrix
	warpPerspective(image, imageOut, trans, sizeOut, INTER_LANCZOS4);
	//std::cout << "bildRotieren erfolgreich" << std::endl;
	//Bild ausgeben
	//std::cout <<"Projektion: "<< "xmin: " << xmin << "ymin: " << ymin << "xamx: " << xmax << "ymax: " << ymax << std::endl;
	return imageOut;

};

/*
überladene bildRotieren für Kugelkoordinaten
*/
Mat Projektion::bildRotieren(Mat image, double theta, double phi)
{
	//Umrechnung der Koordinaten
	return bildRotieren(image, theta*cos(M.WinkelZuBogen(phi)) +90, theta*sin(M.WinkelZuBogen(phi)) +90, 90);

}

/*
input: the image properties that you want rotated.
output: the Mat object to put the resulting file in. (global 'sizeOut' changed)
alpha: the rotation around the x axis 
beta: the rotation around the y axis
gamma: the rotation around the z axis (Drehung des ganzen Bildes)
*/
Mat Projektion::matrixErrechnen(double alpha = 90, double beta = 90, double gamma = 90)
{
	// get width and height for ease of use in matrices
	double w = (double)size.width;
	double h = (double)size.height;

	// Translationsvariablen errechnen
	double dx = 0; double dy = 0; // werden später errechnet
	double dz = sqrt(w*w + h*h); //Bilddiagonale
	double f = dz; // Entfernung zum Bild == Distanz von Kamera zu Bild
	


	// Winkelumrechnung 
	alpha = M.WinkelZuBogen(alpha - 90);
	beta = M.WinkelZuBogen(beta-90);
	gamma = M.WinkelZuBogen(gamma - 90);
	
	// Projection 2D -> 3D matrix
	Mat A1 = (Mat_<double>(4, 3) <<
		1, 0, -w / 2,
		0, 1, -h / 2,
		0, 0, 0,
		0, 0, 1);

	// Rotation matrices around the X, Y, and Z axis
	Mat RX = (Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(alpha), -sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1);

	Mat RY = (Mat_<double>(4, 4) <<
		cos(beta), 0, -sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1);

	Mat RZ = (Mat_<double>(4, 4) <<
		cos(gamma), -sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix with (RX, RY, RZ)
	Mat R = RX * RY * RZ;

	// Translation matrix
	Mat T = (Mat_<double>(4, 4) <<
		1, 0, 0, dx,
		0, 1, 0, dy,
		0, 0, 1, dz,
		0, 0, 0, 1);

	// 3D -> 2D matrix
	Mat A2 = (Mat_<double>(3, 4) <<
		f, 0, w / 2, 0,
		0, f, h / 2, 0,
		0, 0, 1, 0);

	// Final transformation matrix
	trans = A2 * (T * (R * A1));
		
	//Zwischenschritt: Größe des Bildes berechnen ---------------------------
	 //Obenlinks
	Mat oL = (Mat_<double>(3, 1) << 0, 0, 1);
	 //UntenLinks
	Mat uL = (Mat_<double>(3, 1) << 0, size.height - 1, 1);
	 //ObenRechts
	Mat oR = (Mat_<double>(3, 1) << size.width - 1, 0, 1);
	 //UntenRechts
	Mat uR = (Mat_<double>(3, 1) << size.width - 1, size.height - 1, 1);
	sizeBerechnen(oL,uL,oR,uR,trans);
	//-----------------------------------------------------------------------

	//Translationmatrix (Zurechtrücken des Bildes)
	Mat tlt = (Mat_<double>(3, 3) <<
		1, 0, (0 - xmin),
		0, 1, (0 - ymin),
		0, 0, 1);
	//std::cout << "MatrixErrechnen erfolgreich" << std::endl;
	trans = tlt*trans;
	return trans;
};


/*
Berechnung der Größe des neuen Bildes. Ausgabe: xmin, xmax, ymin, ymax
*/

std::vector<double> Projektion::sizeBerechnen(Mat ol,Mat ul,Mat orr,Mat ur, Mat transL) {
	// Verschiebung der Eckpunkte durch Matrix berechnen.
	Mat oL = PunktVerschieben(ol,transL);
	Mat uL = PunktVerschieben(ul, transL);
	Mat oR = PunktVerschieben(orr, transL);
	Mat uR = PunktVerschieben(ur, transL);
	

	//Eckpunkte, aufgeteilt in Koordinatenachsen, durch z-Wert geteilt und gespeichert
	double EckpunkteX[4] = { oL.at<double>(0, 0) , oR.at<double>(0, 0) , uL.at<double>(0, 0) , uR.at<double>(0, 0) };
	double EckpunkteY[4] = { oL.at<double>(1, 0) , oR.at<double>(1, 0) , uL.at<double>(1, 0) , uR.at<double>(1, 0) };
	
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



	//Ausgabe der einzelnen X,Y,Z-Werte zur Kontrolle
/*
	std::cout << std::endl; 
	std::cout << EckpunkteX[0] << " " << EckpunkteX[1] << " " << EckpunkteX[2] << " " << EckpunkteX[3] << " xmax:" << xmax << " xmin:" << xmin << std::endl;
	std::cout << EckpunkteY[0] << " " << EckpunkteY[1] << " " << EckpunkteY[2] << " " << EckpunkteY[3] << " ymax:" << ymax << " ymin:" << ymin << std::endl;
	std::cout << "z:" << oL.at<double>(2, 0) << " " << oR.at<double>(2, 0) << " " << uL.at<double>(2, 0) << " " << uR.at<double>(2, 0) << " " << std::endl;
*/
	//Errechnen der Bildgröße des zu erzeugenden Bilds
	sizeOut= Size((xmax - xmin + 1), ymax - ymin + 1);
	std::vector<double> VecOut{ xmin,xmax, ymin,ymax }; 
	//std::cout <<"sizeBerechnen erfolgreich" <<std::endl;
	return VecOut;
};
/*
	Überladene PunktVerschieben
	Eingabe: Punkt, Translationsmatrix,Transformationsmatrix
*/
Mat Projektion::PunktVerschieben(Mat punkt, Mat transL, Mat trans) {
	return PunktVerschieben(punkt, trans*transL);
}
/*
PunktVerschieben:
Eingabe: Punkt, Transformationsmatrix
*/
Mat Projektion::PunktVerschieben(Mat punkt,Mat trans){
	Mat punktT = trans*punkt;
	punktT.at<double>(0, 0) = punktT.at<double>(0, 0) / punktT.at<double>(2, 0);
	punktT.at<double>(1, 0) = punktT.at<double>(1, 0) / punktT.at<double>(2, 0);

	return punktT;
}
std::vector <Mat> Projektion::PunkteVerschieben(std::vector<Mat> punkte, Mat trans) {
	std::vector <Mat> punkteT=punkte;
	for (int i = 0; i < punkte.size(); i++){
		punkteT.at(i) = PunktVerschieben(punkteT.at(i),trans);
		
	}
	return punkteT;
}std::vector <Point2f> Projektion::PunkteVerschieben(std::vector<Point> punkte, Mat trans) {
	std::vector <Point2f> punkteT ;
	punkteT.resize(punkte.size());
	Mat buffer;
	for (int i = 0; i < punkte.size(); i++) {
		buffer= PunktVerschieben((Mat_<double>(3, 1) << punkte.at(i).x, punkte.at(i).y, 1), trans);
		punkteT.at(i) = Point2f(buffer.at<double>(0,0),buffer.at<double>(1,0));
	}
	return punkteT;
}
