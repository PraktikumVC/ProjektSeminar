#include <iostream>
#include <fstream>
#include <string>
#include "Speicher.h"

using namespace std;

int mainxD() {
	Speicher Speicher;
	Speicher.SetFolder("VC\\Training\\MLM\\MLM\\ZuBuD\\object0001.view03\\Hessian-Affine\\0");
	string line;
	Speicher.ReadText("C:\\VC\\TrainingPM","01_02_Hessian - Affine_negative.txt");
	ifstream myfile("0.sift");
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			cout << line << '\n';
		}
		myfile.close();
	}

	else cout << "Unable to open file";

	return 0;
}