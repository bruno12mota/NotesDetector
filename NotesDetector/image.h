#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <opencv\cv.h>
#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace std;
using namespace cv;

class Image
{

private:

	vector<KeyPoint> img_keypoints;
	Mat img_descriptors;

	FeatureDetector* detector;
	DescriptorExtractor* extractor;

public:
	Mat img;

	Image(String file_path, FeatureDetector* detector, DescriptorExtractor* extractor);
    ~Image(void);

	void detect_keypoints();
	void compute_descriptors();
	
	Mat get_descriptors();
	vector<KeyPoint> get_keypoints();
	
	void set_keypoints(vector<KeyPoint> keypoints);
};

#endif