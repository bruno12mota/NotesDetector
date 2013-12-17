#include "detector.h"

#include "windows.h"

Detector::~Detector(void) {}

Detector::Detector(string scene_path, string feature_detector, string descriptor_extractor, string matcher_type){
	has_error = false;

	LARGE_INTEGER frequency; // ticks per second
    LARGE_INTEGER begin, end;
    double elapsed_time;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&begin);

	//options
	this->feature_detector = feature_detector;
	this->descriptor_extractor = descriptor_extractor;
	this->matcher_type = matcher_type;

	//Get scene image
	Mat img_scene = imread( scene_path, CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_final = imread( scene_path );

	//Check if images are loaded
	if( !img_scene.data ){ 
		cout << " --(!) Error reading scene image, make sure path is correct! " << endl; 
		has_error = true;
		return; 
	}

	//Scene Key Points
	vector<KeyPoint> scene_keypoints = this->get_key_points(img_scene);

	//Calculate descriptors of scene
	Mat scene_descriptors = this->calculate_descriptors(img_scene, scene_keypoints);


	string bills[8][2] = {
        {"bills/5eu_r.jpg", "5"},
        {"bills/5eu_v.jpg", "5"},
        {"bills/10eu_r.jpg", "10"},
        {"bills/10eu_v.jpg", "10"},
        {"bills/20eu_r.jpg", "20"},
        {"bills/20eu_v.jpg", "20"},
        {"bills/50eu_r.jpg", "50"},
        {"bills/50eu_v.jpg", "50"}
    };

	unsigned int total = 0;

	//For each bill
	for (int i = 0; i < 8; i++) {
		//Load bill
		Mat img_object = imread( bills[i][0] , CV_LOAD_IMAGE_GRAYSCALE );

		//Check if bill is loaded
		if( !img_object.data ){ 
			cout << " --(!) Error reading bill image, make sure path is correct! " << endl; 
			has_error = true;
			return; 
		}

		//Current Bill Key Points
		vector<KeyPoint> object_keypoints = this->get_key_points(img_object);

		//Calculate descriptors of Current Bill
		Mat descriptors_object = this->calculate_descriptors(img_object, object_keypoints);

		//Get Good Matches
		vector< DMatch > good_matches = this->get_good_matches(descriptors_object, scene_descriptors);

		if(good_matches.size() < 4) {
			cout << "4 points are needed to calculate Homography. Found " << good_matches.size() << "." << endl;
		}
		else{
			cout << "Matching" << endl;

			Mat img_matches;
			drawMatches( img_object, object_keypoints, img_scene, scene_keypoints,
						good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

			//-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;

			for( int i = 0; i < good_matches.size(); i++ )
			{
				//-- Get the keypoints from the good matches
				obj.push_back( object_keypoints[ good_matches[i].queryIdx ].pt );
				scene.push_back( scene_keypoints[ good_matches[i].trainIdx ].pt );
			}

			Mat inliers;
			Mat H = findHomography( obj, scene, CV_RANSAC, 3, inliers );

			//-- Get the corners from the image_1 ( the object to be "detected" )
			vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
			obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
			vector<Point2f> scene_corners(4);

			vector<DMatch> inlier_matches;
			vector<Point2f> inlier_points;
			for(int i = 0; i < inliers.rows; ++i) {
				if(inliers.at<uchar>(i, 0) != 0) {
					inlier_matches.push_back(good_matches[i]);
					Point2f point = scene_keypoints[ good_matches[i].trainIdx ].pt; 
					inlier_points.push_back(point);
				}
			}

			perspectiveTransform( obj_corners, scene_corners, H);

			if( bill_found(inlier_points, scene_corners) ){
				cout << "Found bill!" << endl;
				total += atoi( bills[i][1].c_str() );

				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line( img_final, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
				line( img_final, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
				line( img_final, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
				line( img_final, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
			}
			else{
				cout << "Didn't find bill!" << endl;
			}
		}
	}
	
    QueryPerformanceCounter(&end);
	elapsed_time = (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "Elapsed time: " << elapsed_time << " ms\n";
	
	char total_str [50];
	sprintf (total_str, "%d Euros", total);
	putText(img_final, total_str, Point2f(10, 35) , FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,0,0), 2);
	
	imshow( "Euro Detection", img_final );
	waitKey(0);
	cout << "Total: " << total << endl;
}

bool Detector::bill_found(vector<Point2f> inlier_points, vector<Point2f> scene_corners){
	for (unsigned int i = 0; i < inlier_points.size(); ++i) {
		if(pointPolygonTest(scene_corners, inlier_points[i], false) < 0) {
			return false;
		}
	}

	return true;
}

vector< DMatch > Detector::get_good_matches(Mat descriptors_obj, Mat scene_descriptors){
	DescriptorMatcher *matcher;

	if (this->matcher_type == "FLANN") {
        matcher = new FlannBasedMatcher();
    } else if (this->matcher_type == "BRUTE") {
        matcher = new BFMatcher(cv::NORM_HAMMING, true);
    }
	else{
		cout << "Descriptor matcher type incorrect!" << endl;
	}

	vector< DMatch > matches;
	matcher->match( descriptors_obj, scene_descriptors, matches );
	
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( unsigned i = 0; i < matches.size(); i++ ){ 
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
	}

	vector< DMatch > good_matches;

	for( unsigned i = 0; i < matches.size(); i++ ){ 
		if( matches[i].distance < 3*min_dist ){ 
			good_matches.push_back( matches[i] ); 
		}
	}

	return good_matches;
}

vector<KeyPoint> Detector::get_key_points(Mat img){
	FeatureDetector *detector;

	if (this->feature_detector == "FAST") {
        detector = new FastFeatureDetector();
    } else if (this->feature_detector == "SURF") {
        detector = new SurfFeatureDetector(400);
    } else if(this->feature_detector == "SIFT") {
        detector = new SiftFeatureDetector();
    } else if(this->feature_detector == "ORB") {
        detector = new OrbFeatureDetector();
    }
	else{
		cout << "Feature Detector type incorrect!" << endl;
	}

	vector<KeyPoint> keypoints;
	detector->detect( img , keypoints );

	return keypoints;
}

Mat Detector::calculate_descriptors(Mat img, vector<KeyPoint> keypoints){
	DescriptorExtractor *extractor;

	if (this->descriptor_extractor == "SURF") {
        extractor = new SurfDescriptorExtractor();
    } else if (this->descriptor_extractor == "SIFT") {
        extractor = new SiftDescriptorExtractor(400);
    } else if(this->descriptor_extractor == "ORB") {
        extractor = new OrbDescriptorExtractor();
    } else if(this->descriptor_extractor == "BRIEF") {
        extractor = new BriefDescriptorExtractor();
    } else if(this->descriptor_extractor == "FREAK") {
        extractor = new FREAK();
    }
	else{
		cout << "Descriptor Extractor type incorrect!" << endl;
	}

	Mat descriptors;
	extractor->compute( img, keypoints, descriptors );

	return descriptors;
}