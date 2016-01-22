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
	int Random(int zahlA, int zahlB, bool echt);
	int Random(int zahl, bool echt);
	std::string VerzeichnisErzeugen(std::string verzeichnis, int random);
	std::string VerzeichnisErzeugen(int random);		
	int lastRandom;
	int counterRandom;
};

