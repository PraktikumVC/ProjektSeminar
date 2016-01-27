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
	std::string file = "25_31_10000_10000";
	std::string directory1 = "J:\\VC\\exe\\test "+file+"\\"; std::string directory2 = "C:\\VC\\Ranger"; 
	std::string file1 = "ranger_out.prediction"; std::string file2 = "data"+file+"_pol.txt"; std::vector<std::string> predicted; std::vector<std::string> produced; 
	uint einseins=0; uint nulleins = 0; uint einsnull = 0; uint nullnull = 0; uint r = 0; uint x = 0; uint y = 0;
	predicted = Speicher.ReadText(directory1, file1);
	produced  = Speicher.ReadText(directory1, file2);
	x = predicted.size(); y = produced.size();
	if (x < y) x = y;
	for (r = 1; r < x - 1; ++r)
	{
		if (predicted.at(r) == produced.at(r)+" ")
			if (predicted.at(r) == (std::to_string(1) + " "))
				einseins += 1;
			else
				nullnull += 1;
		else
			if (predicted.at(r) == (std::to_string(1) + " "))
				nulleins += 1;
			else
				einsnull += 1;
	}
	float xy = (einsnull + nulleins);
	xy = xy / (einseins + einsnull + nulleins + nullnull);

	std::cout <<"x "<< "1 " <<"0"<< std::endl;
	std::cout << "1 " << einseins << " " << nulleins << std::endl;
	std::cout << "0 " << einsnull << " " << nullnull << std::endl;	
	std::cout << "Fehler: " << std::to_string(xy) << std::endl;

	std::vector<std::string> datfile;
	std::string spacer=" ";
	datfile.push_back("  1"+spacer+" 0");
	datfile.push_back("1 " + std::to_string(einseins) + " " + std::to_string(nulleins));
	datfile.push_back("0 " + std::to_string(einsnull) + " " + std::to_string(nullnull));
	datfile.push_back("Fehler: "+std::to_string(xy) );
	Speicher.WriteText(datfile, "PredictionFile"+file2+".txt",directory1);
	system("pause");
	return 0;
}