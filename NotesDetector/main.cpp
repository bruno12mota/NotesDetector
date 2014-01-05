#include <stdio.h>
#include <string>
#include <iostream>

#include "detector.h"

using namespace std;

int main(int argc, char** argv){
	//initModule_nonfree();

	String file;
	String feature_detector = "SIFT";		// SURF || FAST || SIFT || ORB
	String descriptor_extractor = "SIFT";	// SURF || SIFT || ORB || BRIEF || FREAK
	String matcher_type = "FLANN";			// BRUTE || FLANN

    string combinations[11][3] = {
		{"SURF", "SURF",  "FlannBased"},
        {"SURF", "FREAK", "Bruteforce"},
        {"SURF", "SURF",  "Bruteforce"},
        {"SIFT", "SIFT",  "FlannBased"},
		{"FAST", "SURF",  "FlannBased"},
        {"FAST", "SIFT",  "FlannBased"},
        {"FAST", "BRIEF", "Bruteforce"},
        {"FAST", "FREAK", "Bruteforce"},
        {"FAST", "ORB",   "Bruteforce"},
        {"ORB",  "ORB",   "Bruteforce"},
        {"ORB",  "BRIEF", "Bruteforce"}
	};

	cout << "Choose the image scene to use:" << endl;
	//getline(cin, file);
	file = "test/test5.jpg";

	cout << "Choose the feature detector/descriptor extractor/matcher type combination:" << endl;

	for(int i = 0; i < 11 ; i++){
		cout << "[" << i << "] " << combinations[i][0] << " | " << combinations[i][1] << " | " << combinations[i][2] << endl; 
	}

	cout << endl << "The number of the option you want:";

	string number_str;
	int number;
	getline(cin, number_str);

	number = atoi( number_str.c_str() );

	feature_detector = combinations[number][0];
	descriptor_extractor = combinations[number][1];
	matcher_type = combinations[number][2];

	Detector detector = Detector(file, feature_detector, descriptor_extractor, matcher_type);

	system("pause");
	return 0;
}