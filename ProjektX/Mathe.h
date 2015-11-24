#pragma once

#include <iostream>

class Mathe
{
public:
	Mathe();
	~Mathe();
	double Mathe::WinkelZuBogen(double winkel);
	double Mathe::WinkelZuGrad(double wert);
	std::string Mathe::WinkelZuString(double winkel, double winkel2);
	std::string Mathe::WinkelZuString(double winkel, double winkel2,double winkel3);
};

