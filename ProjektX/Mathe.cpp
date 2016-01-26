#include "Mathe.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>
#include <time.h>



Mathe::Mathe()
{
}


Mathe::~Mathe()
{
}

double Mathe::WinkelZuBogen(double winkel)
{
	return winkel *CV_PI / 180.;
}
double Mathe::WinkelZuGrad(double wert)
{
	return wert / CV_PI * 180.;
}
std::string Mathe::WinkelZuString(double winkel,double winkel2)
{
	int x = winkel ; int y = winkel2;
	return "kugel "+std::to_string(x) + "," + std::to_string(y);
}
std::string Mathe::WinkelZuString(double winkel, double winkel2,double winkel3)
{
	int x = winkel - 90; int y = winkel2 - 90; int z = winkel3 - 90;
	return "kartesisch "+std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z);
}
int Mathe::Random(int zahlA,int zahlB, bool echt)
{
		lastRandom = rand()  % ((zahlB-zahlA)+zahlA);
		return lastRandom;
}
int Mathe::Random(int zahl, bool echt)
{
	return Mathe::Random(0, zahl + INT_MAX, echt);
}
int Mathe::Random( bool echt)
{
	//srand(time(0));
	return rand() % INT_MAX;
	//return Mathe::Random(0, INT_MAX, echt);
}
std::string Mathe::VerzeichnisErzeugen(std::string verzeichnis, int random) 
{
	if (verzeichnis.size() > 0)
		return  verzeichnis + "\\object000" + std::to_string(random) + ".view03\\Hessian-Affine\\";
	else
		return Mathe::VerzeichnisErzeugen(random);
}
std::string Mathe::VerzeichnisErzeugen(int random) 
{
	std::string root="J:\\VC\\Training\\MLM";
	std::string rand = std::to_string(random / 100 % 10) + std::to_string(random / 10 % 10) + std::to_string(random % 10);

	return root +  "\\object0" + rand + ".view03\\Hessian-Affine\\";
}