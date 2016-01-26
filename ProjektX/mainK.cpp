/*-------------------------------------------------------------------------------
 This file is part of Ranger.

 Ranger is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Ranger is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Ranger. If not, see <http://www.gnu.org/licenses/>.

 Written by:

 Marvin N. Wright
 Institut für Medizinische Biometrie und Statistik
 Universität zu Lübeck
 Ratzeburger Allee 160
 23562 Lübeck

 http://www.imbs-luebeck.de
 wright@imbs.uni-luebeck.de
 #-------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------
  std::cout << "Options:" << std::endl;
  std::cout << "    " << "--help                        Print this help." << std::endl;
  std::cout << "    " << "--version                     Print version and citation information." << std::endl;
  std::cout << "    " << "--verbose                     Turn on verbose mode." << std::endl;
  std::cout << "    " << "--file FILE                   Filename of input data." << std::endl;
  std::cout << "    " << "--treetype TYPE               Set tree type to:" << std::endl;
  std::cout << "    " << "                              TYPE = 1: Classification." << std::endl;
  std::cout << "    " << "                              TYPE = 3: Regression." << std::endl;
  std::cout << "    " << "                              TYPE = 5: Survival." << std::endl;
  std::cout << "    " << "                              (Default: 1)" << std::endl;
  std::cout << "    " << "--probability                 Grow a Classification forest with probability estimation for the classes." << std::endl;
  std::cout << "    " << "                              Use in combination with --treetype 1." << std::endl;
  std::cout << "    " << "--depvarname NAME             Name of dependent variable. For survival trees this is the time variable." << std::endl;
  std::cout << "    " << "--statusvarname NAME          Name of status variable, only applicable for survival trees." << std::endl;
  std::cout << "    " << "                              Coding is 1 for event and 0 for censored." << std::endl;
  std::cout << "    " << "--ntree N                     Set number of trees to N." << std::endl;
  std::cout << "    " << "                              (Default: 500)" << std::endl;
  std::cout << "    " << "--mtry N                      Number of variables to possibly split at in each node." << std::endl;
  std::cout << "    " << "                              (Default: sqrt(p) for Classification and Survival, p/3 for Regression. " << std::endl;
  std::cout << "    " << "                               p = number of independent variables)" << std::endl;
  std::cout << "    " << "--targetpartitionsize N       Set minimal node size to N." << std::endl;
  std::cout << "    " << "                              For Classification and Regression growing is stopped if a node reaches a size smaller than N." << std::endl;
  std::cout << "    " << "                              For Survival growing is stopped if one child would reach a size smaller than N." << std::endl;
  std::cout << "    " << "                              This means nodes with size smaller N can occur for Classification and Regression." << std::endl;
  std::cout << "    " << "                              (Default: 1 for Classification, 5 for Regression, and 3 for Survival)" << std::endl;
  std::cout << "    " << "--catvars V1,V2,..            Comma separated list of names of (unordered) categorical variables. " << std::endl;
  std::cout << "    " << "                              Categorical variables must contain only positive integer values." << std::endl;
  std::cout << "    " << "--write                       Save forest to file <outprefix>.forest." << std::endl;
  std::cout << "    " << "--predict FILE                Load forest from FILE and predict with new data." << std::endl;
  std::cout << "    " << "--impmeasure TYPE             Set importance mode to:" << std::endl;
  std::cout << "    " << "                              TYPE = 0: none." << std::endl;
  std::cout << "    " << "                              TYPE = 1: Node impurity: Gini for Classification, variance for Regression." << std::endl;
  std::cout << "    " << "                              TYPE = 2: Permutation importance, scaled by standard errors." << std::endl;
  std::cout << "    " << "                              TYPE = 3: Permutation importance, no scaling." << std::endl;
  std::cout << "    " << "                              (Default: 0)" << std::endl;
  std::cout << "    " << "--noreplace                   Sample without replacement." << std::endl;
  std::cout << "    " << "--splitrule RULE              Splitting rule:" << std::endl;
  std::cout << "    " << "                              RULE = 1: Gini for Classification, variance for Regression, logrank for Survival." << std::endl;
  std::cout << "    " << "                              RULE = 2: AUC for Survival, not available for Classification and Regression." << std::endl;
  std::cout << "    " << "                              (Default: 1)" << std::endl;
  std::cout << "    " << "--splitweights FILE           Filename of split select weights file." << std::endl;
  std::cout << "    " << "--alwayssplitvars V1,V2,..    Comma separated list of variable names to be always considered for splitting." << std::endl;
  std::cout << "    " << "--nthreads N                  Set number of parallel threads to N." << std::endl;
  std::cout << "    " << "                              (Default: Number of CPUs available)" << std::endl;
  std::cout << "    " << "--seed SEED                   Set random seed to SEED." << std::endl;
  std::cout << "    " << "                              (Default: No seed)" << std::endl;
  std::cout << "    " << "--outprefix PREFIX            Prefix for output files." << std::endl;
  std::cout << "    " << "--memmode MODE                Set memory mode to:" << std::endl;
  std::cout << "    " << "                              MODE = 0: double." << std::endl;
  std::cout << "    " << "                              MODE = 1: float." << std::endl;
  std::cout << "    " << "                              MODE = 2: char." << std::endl;
  std::cout << "    " << "                              (Default: 0)" << std::endl;
  std::cout << "    " << "--savemem                     Use memory saving (but slower) splitting mode." << std::endl;
  #-------------------------------------------------------------------------------*/
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

int mainK(int argc, char **argv) {	


  //Beschreibung für die Variablen: S.o.
  /*
  // All command line arguments as member: Capital letters
  std::vector<std::string> alwayssplitvars;
  std::string depvarname=""; std::string predict = "";  std::string splitweights = ""; //std::string ordner = "";
  MemoryMode memmode=MEM_DOUBLE;
  bool savemem=false;  
  uint nthreads=DEFAULT_NUM_THREADS;  
  // All command line arguments as member: Small letters
  std::vector<std::string> catvars;
  std::string file=""; std::string outprefix="ranger_out"; std::string statusvarname = "";
  ImportanceMode impmeasure=DEFAULT_IMPORTANCE_MODE;
  uint targetpartitionsize=0; uint mtry=0;  uint ntree=DEFAULT_NUM_TREE; uint seed=0;
  SplitRule splitrule=DEFAULT_SPLITRULE;  
  bool probability=false; bool replace=true;  bool verbose= true;  bool write= true;
  TreeType treetype=TREE_CLASSIFICATION;
 */
  Speicher Speicher; Mathe Mathe;
  std::vector<std::string> datfile;
  std::vector<std::string> polaritaet;
  std::string spacer = "  ";
  std::string filename = "data";
  std::string path = "";
  int c; int r=0; int datfilesize;

  srand(1);
  Speicher.verzeichnis = "C:\\";
 // Speicher.SetFolder("VCtrainingsdaten\\ZuBuD\\object0001.view03\\Hessian-Affine\\0");
    
  //Arbeit über Pixel
  /**
  //cv::Mat image1 =cv::imread("0.jpg",0);  
  //cv::Mat image2 = cv::imread("1.jpg", 0);
 
  for (c = 0; c < image1.cols*image1.rows; ++c)
	  datfile.at(0) = datfile.at(0) + spacer + "Grauwert"+std::to_string(c);
  datfile.at(0) = datfile.at(0) + spacer+ "Fits";
  for (c = 0, r = 0; c < image1.cols, r<image1.rows; ++c, ++r)
	  datfile.at(1) = datfile.at(1) + spacer + std::to_string(image1.at<unsigned char>(c, r));

  for (c = 0, r = 0; c < image2.cols, r<image2.rows; ++c, ++r)
	  datfile.at(1) = datfile.at(1) + spacer + std::to_string(image2.at<unsigned char>(c, r));
	  /**/
  //Arbeit über SIFT-Deskriptor
  std::vector<std::string>lines;
  std::vector<std::string>ordner;
  std::vector<std::string>kontext;
  std::string zwischenspeicher; std::string zwischenspeicher2; std::string zwischenspeicher3;
  std::string wasd;
  std::string verzeichnis = "J:\\VC\\Training\\MLM"; // \\MLM
  std::string rootVerzeichnis=verzeichnis;
  uint random1 = 0; uint random2;  uint random3;  uint random4; uint random5; uint random6; uint randPuffer; uint counter1 = 0; uint counter0 = 0; bool flag = false;
  
  uint zwischenwert = 16; uint anzahlSIFT0 = 10000; uint anzahlSIFT1 = 10000;
  ordner = Speicher.FindFiles( verzeichnis + "\\*");
  uint ordnerSize = 31;//ordner.size();
  std::cout << "Ordnergroesse:"<<ordnerSize << std::endl;
  while ((r < (anzahlSIFT1 + counter1)) && (counter1<10000)) {
	  flag = false;
	  lines.clear();
	  random1 = Mathe.Random(zwischenwert, ordnerSize, true) ;
	  while (random1 == 0)
		  random1 = Mathe.Random(zwischenwert, ordnerSize,true) ;
	  random2 = Mathe.Random(true);
	  random3 = Mathe.Random(true);
	  random4 = Mathe.Random(true);
	  verzeichnis = Mathe.VerzeichnisErzeugen(random1); //random1.view03\MSER\

	  zwischenspeicher = verzeichnis + Speicher.FindFile("", verzeichnis + "*", random2 )+"\\"; //random1.view03\MSER\random2\
	  //_________________________________________________________________________________________________________________________________
	  try {	  
		  zwischenspeicher2 = Speicher.FindFile("", zwischenspeicher + "*.sift", random3);
		  lines.push_back(Speicher.ReadText(zwischenspeicher, zwischenspeicher2 ).at(0));  } //random1.view03\MSER\"random2"\"random3".sift
	  catch (...)  {
		  counter1 += 1;
		  flag =true;
	  }
	  if (flag) continue;
	  //_________________________________________________________________________________________________________________________________
	  try {
		  if (Speicher.AnzahlFilesImSpeicher > 1)
			  while (random3%Speicher.AnzahlFilesImSpeicher == random4%Speicher.AnzahlFilesImSpeicher)
				  random3 = Mathe.Random(1, true);
		  else
			  throw (0);
		zwischenspeicher2 = Speicher.FindFile("", zwischenspeicher + "*.sift", random3);			 
		
		 // std::cout<<zwischenspeicher2 <<"-"<< zwischenspeicher3 <<" || " <<std::ends;
		  lines.push_back(Speicher.ReadText(zwischenspeicher, zwischenspeicher2 ).at(0));
	  }  
	  catch (...) {
		  counter1 += 1;
		  flag = true;
	  }
	  if (flag) continue;
	  datfile.push_back(lines.at(0)+" "+lines.at(1)+" 1");
	  kontext.push_back(zwischenspeicher + "\\" + Speicher.FindFile("", zwischenspeicher + "*.sift", random3));
	  kontext.push_back(zwischenspeicher + "\\" + Speicher.FindFile("", zwischenspeicher + "*.sift", random4));
	  r++;	  
  }
  std::cout << std::endl;
  std::cout << "end of polaritaet =1, r=" + std::to_string(r) << std::endl;
  datfilesize = datfile.size();
  kontext.push_back("end of polaritaet =1, r="+std::to_string(r));
  r = 0;
  random3 = 0; random4 = 0; random5 = 0; random6 = 0;
  while (r < (anzahlSIFT0+counter1 ) ) {
	  flag = false;
	  lines.clear();
	  randPuffer = random1;
	  random1 = 0;
	  while (random1 == 0 || randPuffer == random1)
		  random1 = Mathe.Random(zwischenwert, ordnerSize, true);
	  randPuffer = random2;
	  random2 = random1;
	  while (random1 == random2||randPuffer==random2)
		  random2 = Mathe.Random(zwischenwert, ordnerSize, true);
	  //std::cout << random3 << "|" << random4 << "|" << random5 << "|" << random6 << "||" << std::ends;
	  random3 = Mathe.Random(true); random4 = Mathe.Random(true); random5 = Mathe.Random(true); random6 = Mathe.Random(true); 
	  if (random3 == random5) random5 += 1;
	  if (random4 == random6) random6 += 1;
	  
	  //_________________________________________________________________________________________________________________________________
	  verzeichnis = Mathe.VerzeichnisErzeugen(rootVerzeichnis,random1); //random1.view03\MSER\													
	  zwischenspeicher = verzeichnis + Speicher.FindFile("", verzeichnis + "*", random3); //random1.view03\MSER\random3\ 
	  try {
		  lines.push_back(Speicher.ReadText(zwischenspeicher, Speicher.FindFile("", zwischenspeicher + "\\*.sift", random4)).at(0)); //random1.view03\MSER\random3\random4.sift		  
	  }
	  catch (...) {
		  counter0 += 1;
		  flag = true;
	  }
	  if (flag) continue;
	  kontext.push_back(zwischenspeicher+"\\" +Speicher.FindFile("", zwischenspeicher + "\\*.sift", random4) );
//_________________________________________________________________________________________________________________________________
	  verzeichnis = Mathe.VerzeichnisErzeugen(rootVerzeichnis, random2); //random2.view03\MSER\ 													
	  zwischenspeicher = verzeichnis + Speicher.FindFile("", verzeichnis + "*", random5); //random2.view03\MSER\random5\ 
	  try {
		  lines.push_back(Speicher.ReadText(zwischenspeicher, Speicher.FindFile("", zwischenspeicher + "\\*.sift", random6)).at(0)); //random2.view03\MSER\random5\random6.sift 
	  }
	  catch (...) {
		  counter0 += 1;
		  flag = true;
	  }
	  if (flag) continue;
	  kontext.push_back(zwischenspeicher + "\\" + Speicher.FindFile("", zwischenspeicher + "\\*.sift", random6));
	  datfile.push_back(lines.at(0)+ " " + lines.at(1)+" 0");	  
	  r++;	  
  }
  std::cout << "end of polaritaet =0, r=" + std::to_string(r) << std::endl;
  kontext.push_back("end of polaritaet =0");
  for (c = 0; c < Mathe.Random(5, true)+3;++c)  std::random_shuffle(datfile.begin(), datfile.end());

  std::vector<std::string>::iterator it2;
  it2 = datfile.begin();
  it2 = datfile.insert(it2, "SIFT0");

  for (c = 1; c < 256; ++c)
	  datfile.at(0) = datfile.at(0) + spacer + "SIFT" + std::to_string(c);
  datfile.at(0) = datfile.at(0) + spacer + "polaritaet";

  for (r = 0; r < datfile.size();++r) {
	  wasd = datfile.at(r).back();
	  polaritaet.push_back(wasd );
  }
  filename = filename + std::to_string(zwischenwert) + "_" + std::to_string(ordnerSize) + "_" + std::to_string(anzahlSIFT1) + "_" + std::to_string(anzahlSIFT0);
  kontext.push_back( "Fehlercounter:1: " + std::to_string(counter1) + "|0: " + std::to_string(counter0) );
  kontext.push_back("Groesse von " + filename + ":" + std::to_string(datfile.size()));
  Speicher.WriteText(datfile, filename);
  Speicher.WriteText(kontext, filename+"_kontext.txt");
  Speicher.WriteText(polaritaet, filename+"_pol.txt");
  std::cout << "Fehlercounter:1: " << counter1 << "|0: " << counter0 << std::endl;
  std::cout <<"Groesse von "+filename+":"<< datfile.size()<<std::endl;
  system("pause");

 /* 
  Forest* forest = 0;
  try {


    // Create forest object
    switch (treetype) {
    case TREE_CLASSIFICATION:
      if (probability) {
        forest = new ForestProbability;
      } else {
        forest = new ForestClassification;
      }
      break;
    case TREE_REGRESSION:
      forest = new ForestRegression;
      break;
    case TREE_SURVIVAL:
      forest = new ForestSurvival;
      break;
    case TREE_PROBABILITY:
      forest = new ForestProbability;
      break;
    }

    // Verbose output to logfile if non-verbose mode
    std::ostream* verbose_out;
    if (verbose) {
      verbose_out = &std::cout;
    } else {
      std::ofstream* logfile = new std::ofstream();
      logfile->open(outprefix + ".log");
      if (!logfile->good()) {
        throw std::runtime_error("Could not write to logfile.");
      }
      verbose_out = logfile;
    }

    // Call Ranger
    *verbose_out << "Starting Ranger." << std::endl;
    forest->initCpp(depvarname, memmode,file, mtry,
        outprefix, ntree, verbose_out, seed, nthreads,
        predict, impmeasure, targetpartitionsize, splitweights,
       alwayssplitvars, statusvarname, replace, catvars,
        savemem, splitrule);
	

    forest->run(true);
    if (write) {
      forest->saveToFile();
    }
    forest->writeOutput();
    *verbose_out << "Finished Ranger." << std::endl;

    delete forest;
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << " Ranger will EXIT now." << std::endl;
    delete forest;
    return -1;
  }

  return 0
  */
}
