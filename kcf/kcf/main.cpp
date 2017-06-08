
#include "KCFTracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv/cv.h>
#include <Windows.h>

using namespace std;
using namespace cv;
using namespace kcf;

int keyboard;
void processImagesKCF(char* fistFrameFilename);

LARGE_INTEGER frequency;        // ticks per second
LARGE_INTEGER t1, t2;           // ticks
double elapsedTime;

int main()
{ 
	processImagesKCF("D:\\uav\\CACFT\\Context-Aware-CF-Tracking\\MOSSE\\sequences\\Jumping\\img\\0001.jpg");
	return 0;
}

void processImagesKCF(char* fistFrameFilename) {

	bool use_gray = true; // always
	cv::Mat Im;
	if (use_gray == true)
		Im = imread(fistFrameFilename, CV_LOAD_IMAGE_GRAYSCALE);
	else
		Im = imread(fistFrameFilename);

	if (Im.empty()) { cerr << "Unable to open first image frame: " << fistFrameFilename << endl; exit(EXIT_FAILURE); }
	cv::Mat ImRGBRes; keyboard = 0;
	kcf_tracker mosse;
	bool init = false;
	unsigned char *dataYorR; unsigned char *dataG; unsigned char *dataB;
	string fn(fistFrameFilename);
	while ((char)keyboard != 27)
	{
		cv::Size dsize = cv::Size(Im.cols * 2, Im.rows * 2);
		cv::resize(Im, ImRGBRes, dsize, 0, 0, INTER_LINEAR);
		if (use_gray == true) {}
		else
			cv::cvtColor(Im, Im, CV_BGR2RGB);

		//get the frame number and write it on the current frame
		size_t index = fn.find_last_of("/");
		if (index == string::npos) { index = fn.find_last_of("\\"); }
		size_t index2 = fn.find_last_of("."); string prefix = fn.substr(0, index + 1); string suffix = fn.substr(index2);
		string frameNumberString = fn.substr(index + 1, index2 - index - 1); istringstream iss(frameNumberString);
		int frameNumber = 0; iss >> frameNumber;
		cv::putText(ImRGBRes, "Frame: " + frameNumberString, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(29, 45, 255));
		int w = Im.cols; int h = Im.rows;
		if (init == false) {
			dataYorR = new unsigned char[Im.rows*Im.cols];
			dataG = new unsigned char[Im.rows*Im.cols];
			dataB = new unsigned char[Im.rows*Im.cols];
		}

		if (use_gray == true) {
			for (int j = 0; j < Im.rows; j++)
			{
				uchar* p = Im.ptr<uchar>(j);
				for (int i = 0; i < Im.cols; i++)
				{
					dataYorR[i + j*Im.cols] = p[i];
					dataG[i + j*Im.cols] = p[i];
					dataB[i + j*Im.cols] = p[i];
				}
			}
		}
		else
			for (int j = 0; j < Im.rows; j++)
			{
				for (int i = 0; i < Im.cols; i++)
				{
					dataYorR[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[0];
					dataG[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[1];
					dataB[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[2];
				}
			};

		int cx = 0; int cy = 0; int rw = 0; int rh = 0; float score = 0;

		if (init == false) {	
			if (use_gray == true)
				mosse.initializeTargetModel(164-1, 126-1, 34, 33, w, h, dataYorR); // jumping
			else
				mosse.initializeTargetModel(164 - 1, 126 - 1, 34, 33, w, h, dataYorR, dataG, dataB);   // jumping 																									
		}

		double fps = 0;

		if (init == true) {
			QueryPerformanceFrequency(&frequency);
			QueryPerformanceCounter(&t1);

			if (use_gray == true)
				mosse.findNextLocation(dataYorR);
			else
				mosse.findNextLocation(dataYorR, dataG, dataB);

			QueryPerformanceCounter(&t2);
			fps = double(frequency.QuadPart) / double((t2.QuadPart - t1.QuadPart));
		}

		mosse.getNewLocationCoordinates(cx, cy, rw, rh, score);

		init = true;
		int  l = (cx - rw / 2); int  t = (cy - rh / 2);
		int  r = (cx + rw / 2); int  b = (cy + rh / 2);
		cv::rectangle(ImRGBRes, Point(2 * l, 2 * t), Point(2 * r, 2 * b), cvScalar(0, 255, 0), 1);

		stringstream ss3;
		ss3 << int(fps);

		//show the current frame and the fg masks
		stringstream ss1; ss1 << score; // << ' ' << rh << ' ' << rw;
		cv::putText(ImRGBRes, ss1.str(), cv::Point(2 * (cx - rw / 2), 2 * (cy - rh / 2 - 5)), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 0, 255), 1, 8, false);
		cv::putText(ImRGBRes, "Tracker: " + ss3.str() + "fps", Point(5, 40), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(29, 45, 255));
		cv::imshow("Tracker", ImRGBRes);
		stringstream ostr; ostr << frameNumberString.c_str(); string nums; ostr >> nums;
		string imageToSave = "Frame/frame_" + nums + ".png";
		cv::imwrite(imageToSave, ImRGBRes);
		keyboard = waitKey(1);
		ostringstream oss;
		oss << setfill('0') << setw(4) << (frameNumber + 1);
		string nextFrameNumberString = oss.str();
		string nextFrameFilename = prefix + nextFrameNumberString + suffix;
		if (use_gray == true)
			Im = imread(nextFrameFilename, CV_LOAD_IMAGE_GRAYSCALE);
		else
			Im = imread(nextFrameFilename);
		if (Im.empty()) { cerr << "Unable to open image frame: " << nextFrameFilename << endl; break; }
		else
			fn.assign(nextFrameFilename);
	}

	ImRGBRes.release(); Im.release();
	delete[] dataB; delete[] dataYorR; delete[] dataG;
}




