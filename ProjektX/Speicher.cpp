#include "Speicher.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sddl.h>
#include <stdio.h>
#include <Windows.h>
#include <aclapi.h>
#include <tchar.h>
#include <Shlwapi.h>
#include <fstream>
#include <vector>


Speicher::Speicher()
{
}
Speicher::~Speicher()
{
}
bool Speicher::Save(cv::Mat img, std::string ordner, std::string uordner)
{
	//Ordner für Eingangsbild
	SetFolder(ordner);
	//Bilder speichern
	std::string buffer = uordner + ".png";
	cv::imwrite(buffer, img);
	return true;
}

/// Bildpfad zwischengespeichert, Unterordner fuer Keypoints erzeugt, Arbeitsumgebung auf erstellten Unterordner gesetzt
bool Speicher::Save(cv::Mat img1, cv::Mat img2, std::string ordner, std::string uordner)
{
	Save(img1, ordner, uordner);
	return true;
}

//Ordner erzeugen und als Arbeitsumgebung setzen
bool Speicher::SetFolder(std::string ordner)
{	
	
	std::string buf = verzeichnis + ordner;
	LPSTR curDirectory = const_cast<char *> (buf.c_str());
	//Ordner erzeugen
	CreateDirectory(curDirectory,NULL);
	//als Arbeitsumgebung setzen
	SetCurrentDirectory(curDirectory);
	std::cout << "SetFolder erfolgreich" << buf << std::endl;
	return true;
}
/**
Einlesen von Textdokumenten, zb SIFT
*/
std::vector<std::string> Speicher::ReadText(std::string directory, std::string file)
{
	std::string buf = "C:\\VC\\Training\\MLM\\MLM\\ZuBuD\\object0001.view03\\Hessian-Affine\\0";
	LPSTR curDirectory = const_cast<char *> (directory.c_str());
	//als Arbeitsumgebung setzen
	SetCurrentDirectory(curDirectory);
	std::vector<std::string> lines;
	std::string line;
	std::string buff = "0.sift";
	std::ifstream myfile(file);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			lines.push_back(line);
		}
		myfile.close();
	}
	else std::cout << "Unable to open file";
	system("pause");

	return lines;
}
bool Speicher::WriteText(std::vector<std::string> lines,std::string file)
{
	std::string buf = "TEXT\\";
	LPSTR curDirectory = const_cast<char *> (buf.c_str());
	//als Arbeitsumgebung setzen
	SetFolder(buf);	
	std::string line;
	std::string buff = "0.sift";
	std::ofstream myfile(file+".txt");
	for (int i = 0; i < lines.size(); ++i) {

		//std::cout << lines.at(i) << std::endl;
		myfile << lines.at(i) <<"\n"<< std::endl;
	}
	myfile.close();
	std::cout << "file " +file+" saved and closed" << std::endl;
	return true;
}


