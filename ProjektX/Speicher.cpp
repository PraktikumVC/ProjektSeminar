#include "Speicher.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <sddl.h>
#include <stdio.h>
#include <Windows.h>
#include <aclapi.h>
#include <tchar.h>
#include <Shlwapi.h>


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



