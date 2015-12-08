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
int Mathe::Random(int zahl, bool echt)
{
	if (echt)
	{
		while(lastRandom== rand() % (zahl + 100)) 
			srand(time(0));
		lastRandom = rand() % (zahl + 100);
		return lastRandom;
	}
	else
	{
		counterRandom = counterRandom + 1;
		srand(counterRandom);
		return rand() % (zahl+100);
	}


}