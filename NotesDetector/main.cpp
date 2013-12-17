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

	cout << "Choose the image scene to use:" << endl;
	getline(cin, file);

	cout << "Choose the feature detector (SURF|FAST|SIFT|ORB):" << endl;
	getline(cin, feature_detector);

	cout << "Choose the descriptor extractor (SURF|SIFT|ORB|BRIEF|FREAK):" << endl;
	getline(cin, descriptor_extractor);

	cout << "Choose the matcher type (FLANN|BRUTE):" << endl;
	getline(cin, matcher_type);

	Detector detector = Detector(file, feature_detector, descriptor_extractor, matcher_type);

	system("pause");
	return 0;
}