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
#include <stdlib.h>



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
	//std::cout <<"ReadText von: " <<directory << "|" << file << std::endl;
	//std::string buf = "C:\\VC\\Training\\MLM\\MLM\\ZuBuD\\object0001.view03\\Hessian-Affine\\0";
	LPSTR curDirectory = const_cast<char *> (directory.c_str());
	//als Arbeitsumgebung setzen
	SetCurrentDirectory(curDirectory);
	std::vector<std::string> lines;
	std::string line;
	std::ifstream myfile(file);
	size_t n;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			while (line.find(" 0 ")!=line.npos) {
				//std::cout << line.find(" 0 ") << std::endl;
				line.replace(line.find(" 0 "), 3, " 0.0 ");
			};
			n = std::count(line.begin(), line.end(), ' ');
			if(n!=127&&n>5)std::cout << "Number of Spaces:| " << n << " |---------------------------------------------------" << std::endl;
			lines.push_back(line);
			//std::cout << line << std::endl;
		}
		myfile.close();
	}
	else throw( "Unable to open file "+file); // "\n"<<" in "<<directory <<
	//system("pause");
	return lines;
}

bool Speicher::WriteText(std::vector<std::string> lines,std::string file)
{
	std::string buf = "VC\\Ranger\\";
	LPSTR curDirectory = const_cast<char *> (buf.c_str());
	//als Arbeitsumgebung setzen
	SetFolder(buf);	
	std::ofstream myfile(file);
	for (int i = 0; i < lines.size(); ++i) {
		myfile << lines.at(i) << std::endl;
	}
	myfile.close();
	std::cout << "file " +file+" saved and closed" << std::endl;
	return true;
}
bool Speicher::WriteText(std::vector<std::string> lines, std::string file,std::string directory)
{
	std::string buf = directory;
	LPSTR curDirectory = const_cast<char *> (buf.c_str());
	//als Arbeitsumgebung setzen
	SetFolder(buf);
	std::ofstream myfile(file);
	for (int i = 0; i < lines.size(); ++i) {
		myfile << lines.at(i) << std::endl;
	}
	myfile.close();
	std::cout << "file " + file + " saved and closed" << std::endl;
	return true;
}
std::string Speicher::FindFile(std::string file, std::string path, int random)
{
	std::vector < std::string > Dateien =Speicher::FindFiles(path);
	AnzahlFilesImSpeicher = Dateien.size();
	return Dateien.at(random%Dateien.size());
}

std::vector < std::string > Speicher::FindFiles(std::string path)
{
	HANDLE fHandle;
	WIN32_FIND_DATA wfd;
	LPSTR curDirectory = const_cast<char *> (path.c_str());
	std::vector < std::string >  Dateien;
	fHandle = FindFirstFile(curDirectory, &wfd);
	if (std::to_string(path.back())=="*") {
		FindNextFile(fHandle, &wfd);
		FindNextFile(fHandle, &wfd);
	}
	do
	{
		Dateien.push_back(wfd.cFileName/*[i]*/);
		//			i++;
	} while (FindNextFile(fHandle, &wfd));
	FindClose(fHandle);
	AnzahlFilesImSpeicher = Dateien.size();
	return Dateien;
}