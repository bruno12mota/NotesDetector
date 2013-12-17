#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <opencv\cv.h>

#include "opencv2\core\core.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\calib3d\calib3d.hpp"
#include "opencv2\nonfree\nonfree.hpp"

using namespace std;
using namespace cv;

class Detector
{

private:
	bool has_error;
	string feature_detector, descriptor_extractor, matcher_type;

public:
	Detector(string scene_path, string feature_detector, string descriptor_extractor, string matcher_type);
    ~Detector(void);

	vector<KeyPoint> get_key_points(Mat img);
	Mat calculate_descriptors(Mat img, vector<KeyPoint> keypoints);
	vector< DMatch > get_good_matches(Mat descriptors_obj, Mat descriptors_scene);
	bool bill_found(vector<Point2f> inlier_points, vector<Point2f> scene_corners);
};

#endif