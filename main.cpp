#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
#include <armadillo>
//#include "boost/program_options.hpp"
#include <iostream>
#include <fstream>


using namespace std;

void onMouse(int evt, int x, int y, int flags, void* param) {
	if (evt == CV_EVENT_LBUTTONDOWN) {
		std::vector<cv::Point>* ptPtr = (std::vector<cv::Point>*)param;
		ptPtr->push_back(cv::Point(x, y));
	}
}

int main() {

	ofstream resultfile;
	resultfile.open("2021_07_26_B.txt");

	//TODO read from configfile
	int n_skel = 5;
	double length_tail = 180;

	double radius_max = length_tail / (double)(n_skel - 1);
	// the images are vertical
	arma::vec theta = arma::linspace(-2*M_PI / 3, 2*M_PI / 3, 90); 
	arma::vec radius = arma::linspace(1, radius_max, round(radius_max));

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture cap("2021_07_26_B.avi");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

	// Define some variables
	cv::Mat frame;
	cv::Mat background;
	cv::Mat bckg_sub;
	cv::Mat blur;
	cv::Mat fish_pad;
	cv::Mat skeleton_rgb;

	std::vector<cv::Point> points;
	uint32_t frame_num = 0;

    while (1) {
        // Capture frame-by-frame
        cap >> frame; // NOTE frame is a RGB image
		frame_num++;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

		if (frame_num % 100 == 0)
		{
			printf("Number of frames processed %d \r", frame_num);
			fflush(stdout);
		}
		
		cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
		frame.copyTo(skeleton_rgb);
		cv::Mat frame_gray = cv::Mat(frame.rows, frame.cols, CV_8UC1);
		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);


		if (frame_num == 1) {
			cv::namedWindow("Select start");
			cv::setMouseCallback("Select start", onMouse, (void*)&points);
			while (true) {
				cv::imshow("Select start", frame_gray);
				if (points.size() > 0) {
					break;
				}
				cv::waitKey(16);
			}

			cv::Rect2d fish_rect = cv::selectROI("Select fish", frame_gray, false);
			cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
			mask(fish_rect) = 1;
			cv::inpaint(frame_gray, mask, background, 3, CV_INPAINT_TELEA);

			cv::namedWindow("Background");
			char exit_key_press = 0;
			while (exit_key_press != 'q') {
				cv::imshow("Background", background);
				exit_key_press = cvWaitKey(16);
			}
			cv::destroyAllWindows();

			background.convertTo(background, CV_32F, 1.0 / 255.0);
		}
		

		frame_gray.convertTo(frame_gray, CV_32F, 1.0 / 255.0);
		double sum_image = cv::sum(frame_gray)[0] / (frame_gray.rows * frame_gray.cols);

		cv::absdiff(frame_gray,background, bckg_sub);

		double min_fish, max_fish;
		cv::minMaxLoc(bckg_sub, &min_fish, &max_fish);
		bckg_sub = (bckg_sub - min_fish) / (max_fish - min_fish);

		cv::GaussianBlur(bckg_sub, blur, cv::Size(21, 21), 6, 6);
		cv::copyMakeBorder(blur, fish_pad, round(length_tail), round(length_tail), round(length_tail), round(length_tail), cv::BORDER_CONSTANT, 0);

		//cv::imshow("Bckg sub", fish_pad);

		int x_0 = points[0].x + round(length_tail);
		int y_0 = points[0].y + round(length_tail);
		double best_theta = 0; 
		arma::vec theta_frame(n_skel-1);
		arma::vec skel_x(n_skel);
		arma::vec skel_y(n_skel);
		skel_x(0) = points[0].x;
		skel_y(0) = points[0].y;

		for (int s = 1; s < n_skel; s++) {
			arma::mat Xgrid = arma::round(radius * arma::cos(theta + best_theta).t() + x_0);
			arma::mat Ygrid = arma::round(radius * arma::sin(theta + best_theta).t() + y_0);
			arma::Mat<double> pixels(radius.n_elem, theta.n_elem);
			for (int i = 0; i < radius.n_elem; i++) {
				for (int j = 0; j < theta.n_elem; j++) {
					pixels(i, j) = fish_pad.at<float>(Ygrid(i, j), Xgrid(i, j));
				}
			}

			arma::Row<double> profile = arma::sum(pixels, 0);
			arma::Row<double> gaussian = arma::exp(-arma::pow(arma::linspace(-(int)profile.n_rows / 2, profile.n_rows / 2, profile.n_rows), 2) / 8);
			gaussian = gaussian / arma::sum(gaussian);
			profile = arma::conv(profile, gaussian, "same");
			arma::uword pos_max = arma::index_max(profile);
			double best_theta = theta(pos_max);
			x_0 = x_0 + round(radius_max * cos(best_theta));
			y_0 = y_0 + round(radius_max * sin(best_theta));
			skel_x(s) = x_0 - round(length_tail);
			skel_y(s) = y_0 - round(length_tail);
			theta_frame(s-1) = best_theta;
		}

			

		/*for (int s = 0; s < n_skel; s++) {
			cv::circle(skeleton_rgb, cv::Point(skel_x(s), skel_y(s)), 4, cv::Scalar(0, 0, 255), 1);
		}
		for (int s = 0; s < n_skel - 1; s++) {
			cv::line(skeleton_rgb, cv::Point(skel_x(s), skel_y(s)), cv::Point(skel_x(s + 1), skel_y(s + 1)), cv::Scalar(0, 0, 255));
		}
		cv::resize(skeleton_rgb, skeleton_rgb, cv::Size(), 2, 2);*/

		// write to file
		resultfile << setprecision(17) << frame_num;
		for (int s = 0; s < n_skel - 1; s++) {
			resultfile << "," << setprecision(17) << theta_frame(s);
		}
		resultfile << endl;

		/*
        // Display the resulting frame
			cv::imshow("Skeleton", skeleton_rgb);
			// Press  ESC on keyboard to exit
			char c = (char)cv::waitKey(1);
			if (c == 27)
				break;*/
		
    }

    // When everything done, release the video capture object
    cap.release();
	resultfile.close();

    // Closes all the frames
    cv::destroyAllWindows();

	printf("\n\n=== ENTER TO CLOSE ===\n\n");
	scanf_s("&d");

    return 0;
}
