#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <opencv\cv.h>

#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\nonfree\nonfree.hpp"
#include "image.h"
#include "bill.h"

using namespace std;
using namespace cv;

class Detector
{

private:
	bool has_error;
	FeatureDetector *detector;
	DescriptorExtractor *extractor;
	DescriptorMatcher *matcher;

public:
	Detector(string scene_path, string feature_detector, string descriptor_extractor, string matcher_type);
    ~Detector(void);

	vector< DMatch > get_good_matches(Mat descriptors_obj, Mat descriptors_scene);
	bool bill_found(vector<Point2f> inlier_points, vector<Point2f> scene_corners);
	vector<Bill> make_bills();
};

#endif