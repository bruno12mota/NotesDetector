#include <stdio.h>
#include <string>
#include <iostream>

#include "detector.h"

using namespace std;

int main(int argc, char** argv){
	//initModule_nonfree();

	String feature_detector = "SURF";		// SURF || FAST || SIFT || ORB
	String descriptor_extractor = "SURF";	// SURF || SIFT || ORB || BRIEF || FREAK
	String matcher_type = "BRUTE";			// BRUTE || FLANN

	Detector detector = Detector("test/test3.jpg", feature_detector, descriptor_extractor, matcher_type);

	system("pause");
	return 0;
}