#include "bill.h"

Bill::~Bill(void){}

//Constructor
Bill::Bill(String file_path, FeatureDetector* detector, DescriptorExtractor* extractor, int value): Image(file_path, detector, extractor) {

	this->value = value;

}

//Add unique part
void Bill::add_part(int x0, int y0, int x1, int y1){

	//Save highlight area rectangle from 4 points
	vector<Point2f> part(4);
    part[0] = Point(x0, y0);
    part[1] = Point(x1, y0);
    part[2] = Point(x1, y1);
    part[3] = Point(x0, y1);

	parts.push_back( part );
}


void Bill::only_key_parts(){
	if(this->parts.size() == 0)
		return;

	vector<KeyPoint> new_keypoints;

	//For each keypoint check if not inside the highlighted parts
	for(vector<KeyPoint>::size_type i = 0; i < this->get_keypoints().size(); ++i) {
		for(vector<KeyPoint>::size_type k = 0; k < this->parts.size(); ++k) {

            // keypoint is inside a highlight part then it is kep otherwise it is discarted
            if(pointPolygonTest(this->parts[k], this->get_keypoints()[i].pt, false) >= 0) {
				new_keypoints.push_back( this->get_keypoints()[i] );
                break; //Jump to next keypoint
            }

        }
    }

	this->set_keypoints(new_keypoints);
}


vector<Point2f> Bill::get_corners(){
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); 
	obj_corners[1] = cvPoint( img.cols, 0 );
	obj_corners[2] = cvPoint( img.cols, img.rows ); 
	obj_corners[3] = cvPoint( 0, img.rows );

	return obj_corners;
}