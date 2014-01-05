#ifndef BILL_H
#define BILL_H

#include <string>
#include <opencv\cv.h>
#include "image.h"

using namespace std;
using namespace cv;

class Bill : public Image
{

private:
	vector<vector<Point2f>> parts;
public:
	int value;

	Bill(String file_path, FeatureDetector* detector, DescriptorExtractor* extractor, int value);
    ~Bill(void);

	void add_part(int x0, int y0, int x1, int y1);
	void only_key_parts();
	vector<Point2f> get_corners();
};

#endif