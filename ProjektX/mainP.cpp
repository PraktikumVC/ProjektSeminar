#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include "globals.h"
#include "ForestClassification.h"
#include "ForestRegression.h"
#include "ForestSurvival.h"
#include "ForestProbability.h"

#include "Speicher.h"
#include "Mathe.h"

int main(int argc, char **argv) {
	/*
	IN SPEICHER die Ausgabe für n<127 auskommentieren!!!!
	*/
	Speicher Speicher; Mathe Mathe;
	std::string directory1 = "C:\\Users\\mens\\Desktop\\RangerPredict"; std::string directory2 = "C:\\VC\\Ranger"; 
	std::string file1 = "ranger_out.prediction"; std::string file2 = "data16_31_10000_10000_pol.txt"; std::vector<std::string> predicted; std::vector<std::string> produced; 
	uint einseins=0; uint nulleins = 0; uint einsnull = 0; uint nullnull = 0; uint r = 0; uint x = 0; uint y = 0;
	predicted = Speicher.ReadText(directory1, file1);
	produced  = Speicher.ReadText(directory2, file2);
	x = predicted.size(); y = produced.size();
	if (x < y) x = y;
	for (r = 1; r < x; ++r)
		if (predicted.at(r) == produced.at(r))
			if (predicted.at(r) == (std::to_string(1)+" "))
				einseins += 1;
			else 
				nullnull += 1;
		else 
			if (predicted.at(r) == (std::to_string(1) + " "))
				nulleins += 1;
			else
				einsnull += 1;
	std::cout <<" "<< "1 " <<"0"<< std::endl;
	std::cout << "1 " << einseins << " " << nulleins << std::endl;
	std::cout << "0 " << einsnull << " " << nullnull << std::endl;
	system("pause");
	return 0;
}