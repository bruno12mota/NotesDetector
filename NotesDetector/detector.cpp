#include "detector.h"

#include "windows.h"

Detector::~Detector(void) {}

Detector::Detector(string scene_path, string feature_detector, string descriptor_extractor, string matcher_type){
	
	//Feature Detector Algorithm 
	if (feature_detector == "FAST") {
        detector = new FastFeatureDetector();
    } else if (feature_detector == "SURF") {
        detector = new SurfFeatureDetector(400);
    } else if(feature_detector == "SIFT") {
        detector = new SiftFeatureDetector();
    } else if(feature_detector == "ORB") {
        detector = new OrbFeatureDetector();
    }

	
	//Feature Detector Algorithm 
    if (matcher_type == "FlannBased") {
        matcher = new FlannBasedMatcher();
    } else if (matcher_type == "Bruteforce") {
        matcher = new BFMatcher(NORM_HAMMING, true);
    }

	
	//Feature Detector Algorithm 
    if (descriptor_extractor == "SURF") {
        extractor = new SurfDescriptorExtractor();
        if (matcher_type == "Bruteforce") {
			// NORM_L2 -> without the square root computation which does not introduce error in this matching case and allows less processing to be performed
            matcher = new BFMatcher(NORM_L2, false); 
        }
    } else if (descriptor_extractor == "SIFT") {
        extractor = new SiftDescriptorExtractor();
    } else if (descriptor_extractor == "ORB") {
        extractor = new OrbDescriptorExtractor();
    } else if (descriptor_extractor == "BRIEF") {
        extractor = new BriefDescriptorExtractor();
    } else if (descriptor_extractor == "FREAK") {
        extractor = new FREAK();
    }
	
	
	//Make Image object for scene
	Image scene_image = Image( scene_path, detector, extractor );


	
	//Make bills to test with
	vector<Bill> bills = this->make_bills();
	unsigned int total = 0;

	Mat final_image;
	cvtColor(scene_image.img, final_image, CV_GRAY2RGB);

	for(vector<Bill>::size_type i = 0; i < bills.size(); i++){
		//Bill
		Bill bill = bills[i];

		//Get Good Matches
		vector< DMatch > good_matches = this->get_good_matches( bill.get_descriptors() , scene_image.get_descriptors() );
		
		if(good_matches.size() < 4) {
			cout << "4 points are needed to calculate Homography. Found " << good_matches.size() << "." << endl;
		}
		else{

			vector<Point2f> bill_matched_keypoints;
			vector<Point2f> scene_matched_keypoints;
			
			// Get the keypoints in the scene and bill from the good matches
			for( int i = 0; i < good_matches.size(); i++ )
			{
				bill_matched_keypoints.push_back( bill.get_keypoints()[ good_matches[i].queryIdx ].pt );
				scene_matched_keypoints.push_back( scene_image.get_keypoints()[ good_matches[i].trainIdx ].pt );
			}

			// Calculate Homography
			Mat mask;
			Mat homog = findHomography( bill_matched_keypoints, scene_matched_keypoints, CV_RANSAC, 3, mask );
			
			vector<DMatch> inlier_matches;
			vector<Point2f> inlier_points;
			for(int i = 0; i < mask.rows; ++i) {
				// if the first position in the ith row of the homography mask matrix is different than zero
				// then the scene image point in good_matches[i] is an inlier
				if(mask.at<uchar>(i, 0) != 0) {
					inlier_matches.push_back(good_matches[i]);

					Point2f point = scene_image.get_keypoints()[good_matches[i].trainIdx].pt; 
					inlier_points.push_back(point);
				}
			}

			cout << "\tInlier points: " << inlier_points.size() << "\n";

			Mat testing;
			drawMatches(bill.img, bill.get_keypoints(), scene_image.img, scene_image.get_keypoints(),
					inlier_matches, testing, Scalar::all(-1), Scalar(0,0,255));

			// Apply the homography to the bill corners position to match the one in the scene
			vector<Point2f> bill_corners = bill.get_corners();
			vector<Point2f> bill_on_scene_corners(4);
			perspectiveTransform(bill_corners, bill_on_scene_corners, homog);

			//Draw contour lines of the bill
			Point2f offset((float) bill.img.cols, 0);
			line(testing, bill_on_scene_corners[0] + offset, bill_on_scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
			line(testing, bill_on_scene_corners[1] + offset, bill_on_scene_corners[2] + offset, Scalar(0, 255, 0), 4 );
			line(testing, bill_on_scene_corners[2] + offset, bill_on_scene_corners[3] + offset, Scalar(0, 255, 0), 4 );
			line(testing, bill_on_scene_corners[3] + offset, bill_on_scene_corners[0] + offset, Scalar(0, 255, 0), 4 );


			//Check if all inliners are within the bill on scene corners
			bool bill_exists = true;
			for (unsigned int i = 0; i < inlier_points.size(); ++i) {
				if(pointPolygonTest(bill_on_scene_corners, inlier_points[i], false) < 0) {
					
					//Not within the bill
					bill_exists = false;

					break;
				}
			}

			if(bill_exists){
				int current_value = bill.value;

				//Add to total
				total += current_value;

				//Add to Final Image
				line(final_image, bill_on_scene_corners[0], bill_on_scene_corners[1], Scalar(0, 255, 0), 4 );
				line(final_image, bill_on_scene_corners[1], bill_on_scene_corners[2], Scalar(0, 255, 0), 4 );
				line(final_image, bill_on_scene_corners[2], bill_on_scene_corners[3], Scalar(0, 255, 0), 4 );
				line(final_image, bill_on_scene_corners[3], bill_on_scene_corners[0], Scalar(0, 255, 0), 4 );

				//Max and min x/y
				float max_x = 0, max_y = 0;
				float min_x = scene_image.img.rows, min_y = scene_image.img.rows;
				for(int a = 0; a < bill_on_scene_corners.size(); a++){
					//X
					if(bill_on_scene_corners[a].x > max_x){
						max_x = bill_on_scene_corners[a].x;
					}
					if(bill_on_scene_corners[a].x < min_x){
						min_x = bill_on_scene_corners[a].x;
					}

					//Y
					if(bill_on_scene_corners[a].y > max_y){
						max_y = bill_on_scene_corners[a].y;
					}
					if(bill_on_scene_corners[a].y < min_y){
						min_y = bill_on_scene_corners[a].y;
					}
				}
				cout << "min_x:" << min_x << "     ";
				cout << "min_y:" << min_y << "     ";
				
				cout << "max_x:" << max_x << "     ";
				cout << "max_y:" << max_y << endl;

				//Print value
				string current = static_cast<ostringstream*>( &(ostringstream() << current_value) )->str();

				Size text_size = getTextSize(current, FONT_HERSHEY_COMPLEX, 1, 2, 0);

				Point2f center = Point2f( min_x + (max_x-min_x)/2.0   ,   min_y + (max_y-min_y)/2.0);
				Point2f text_pos = center - Point2f(text_size.width / 2.0, - text_size.height / 2.0);

				putText(final_image, current , text_pos , FONT_HERSHEY_COMPLEX, 1, Scalar(0,255,0), 2);

			}

			//imshow("Test", testing);
			//waitKey(0);
		}
	}


	//Total
	char total_str [50];
	sprintf (total_str, "%d Euros", total);
	putText(final_image, total_str, Point2f(10, 35) , FONT_HERSHEY_COMPLEX, 1, Scalar(255,0,0), 2);
		

	imshow("Final Image", final_image);
	waitKey(0);





	/*LARGE_INTEGER frequency; // ticks per second
    LARGE_INTEGER begin, end;
    double elapsed_time;

	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&begin);


		
	
    QueryPerformanceCounter(&end);
	elapsed_time = (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "Elapsed time: " << elapsed_time << " ms\n";
	
	char total_str [50];
	sprintf (total_str, "%d Euros", total);
	putText(img_final, total_str, Point2f(10, 35) , FONT_HERSHEY_COMPLEX, 1, Scalar(255,0,0), 2);
	
	imshow( "Euro Detection", img_final );
	waitKey(0);
	cout << "Total: " << total << endl;*/
}


// Makes all bills necessary
vector<Bill> Detector::make_bills(){
	vector<Bill> bills;
	string bills_info[8][3] = {
        {"5eu_r", "5"},
        {"5eu_v", "5"},
        {"10eu_r", "10"},
        {"10eu_v", "10"},
        {"20eu_r", "20"},
        {"20eu_v", "20"},
        {"50eu_r", "50"},
        {"50eu_v", "50"}
    };
	
	unsigned int total = 0;
	for (int i = 0; i < 8; i++) {
		string bill_id = bills_info[i][0];
		int bill_value = atoi( bills_info[i][1].c_str() );
		
		//Make bill object
		Bill current_bill = Bill( "bills/"+bill_id+".jpg", detector, extractor, bill_value );

		//Highlights
		if(bill_id == "5eu_r"){
			//5 euro bill front
			current_bill.add_part(6, 5, 22, 25); //Five on the Top left part
			current_bill.add_part(9, 97, 31, 130); //Five on the Bottom left part
			current_bill.add_part(188, 5, 226, 68); //Big Five on the Top right part
			current_bill.add_part(127, 10, 232, 130); //Figure
		}
		else if(bill_id == "5eu_v"){
			//5 euro bill back
			current_bill.add_part(9, 6, 137, 62); //Top left
			current_bill.add_part(7, 101, 24, 129); //Bottom left
			current_bill.add_part(239, 6, 252, 23); //Top right
			current_bill.add_part(242, 107, 257, 129); //bottom right
		}
		else if(bill_id == "10eu_r"){
			//10 euro bill front
			current_bill.add_part(4, 5, 30, 28); //Top left
			current_bill.add_part(6, 106, 36, 133); //Bottom left
			current_bill.add_part(169, 7, 229, 56); //Top right
			current_bill.add_part(136, 30, 232, 132); //Figure
		}
		else if(bill_id == "10eu_v"){
			//10 euro bill back
			current_bill.add_part(7, 7, 150, 79); //Top left + figure
			current_bill.add_part(10, 107, 35, 133); //Bottom left
			current_bill.add_part(233, 5, 255, 26); //Top right
			current_bill.add_part(236, 113, 259, 133); //bottom right
		}
		else if(bill_id == "20eu_r"){
			//20 euro bill front
			current_bill.add_part(7, 4, 29, 27); //Top left
			current_bill.add_part(6, 110, 37, 135); //Bottom left
			current_bill.add_part(161, 5, 223, 57); //Top right
			current_bill.add_part(131, 44, 233, 137); //Figure
		}
		else if(bill_id == "20eu_v"){
			//20 euro bill back
			current_bill.add_part(4, 4, 160, 66); //Top left + figure
			current_bill.add_part(7, 113, 38, 139); //Bottom left
			current_bill.add_part(236, 4, 258, 27); //Top right
			current_bill.add_part(232, 116, 260, 138); //bottom right
		}
		else if(bill_id == "50eu_r"){
			//50 euro bill front
			current_bill.add_part(4, 4, 27, 27); //Top left
			current_bill.add_part(4, 114, 35, 139); //Bottom left
			current_bill.add_part(167, 6, 229, 54); //Top right
			current_bill.add_part(140, 25, 220, 134); //Figure
			current_bill.add_part(224, 80, 257, 114); //Mid right
		}
		else if(bill_id == "50eu_v"){
			//50 euro bill back
			current_bill.add_part(6, 5, 166, 77); //Top left + figure
			current_bill.add_part(10, 114, 44, 141); //Bottom left
			current_bill.add_part(238, 5, 261, 28); //Top right
			current_bill.add_part(220, 112, 256, 139); //bottom right
		}
		
		current_bill.only_key_parts();
		bills.push_back( current_bill );
	}

	return bills;
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