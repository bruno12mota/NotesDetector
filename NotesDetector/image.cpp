#include "image.h"

Image::~Image(void){}

//Constructor
Image::Image( String file_path, FeatureDetector* detector, DescriptorExtractor* extractor ){

	//Load image
	img = imread( file_path, CV_LOAD_IMAGE_GRAYSCALE );

	if( !img.data ){
		cout << " --(!) Error reading an image! " << endl; 
		return;
	}

	this->detector = detector;
	this->extractor = extractor;

	//Processing
	detect_keypoints();
	compute_descriptors();
}

// Detects the image keypoints
void Image::detect_keypoints() {
    detector->detect(img, img_keypoints);
}

// Extracts the descriptors from keypoints
void Image::compute_descriptors() {
    extractor->compute(img, img_keypoints, img_descriptors);
}


Mat Image::get_descriptors(){
	return this->img_descriptors;
}
vector<KeyPoint> Image::get_keypoints(){
	return this->img_keypoints;
}

void Image::set_keypoints( vector<KeyPoint> keypoints ){
	this->img_keypoints.clear();

	for(vector<KeyPoint>::size_type k = 0; k < keypoints.size(); ++k) {
		this->img_keypoints.push_back( keypoints[k] );
	}

	//Reset descriptors
	compute_descriptors();
}