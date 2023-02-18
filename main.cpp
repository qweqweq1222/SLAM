#include "lib.h"

const int STEP = 5;


void TramTest(const string& left_path, const string& mask_path, Mat& K, 
			  fstream& time, fstream& speeds, const string& output, const Mat& R0)
{
	fs::directory_iterator left_iterator(left_path); 
	fs::directory_iterator next_iterator(left_path);
	fs::directory_iterator mask_iterator(mask_path);


	ofstream otp(output);
	Mat left, right, next, mask;
	int dt, next_time;
	vector<Point2f> kp1, kp2;
	double scale_factor = 0;



	int current_time;
	time >> current_time;

	double speed;
	speeds >> speed;

	Mat GLOBAL_COORD = Mat::eye(Size(4, 4), CV_32FC1);
	advance(next_iterator, START_KEY_FRAME + STEP);

	for (int i = 0; i < NUM_OF_FRAMES; i += STEP)
	{
		Mat local = Mat::eye(Size(4, 4), CV_32FC1);
		// 1) reading images
		left = imread((*left_iterator).path().u8string());
		next = imread((*next_iterator).path().u8string());
		mask = imread((*mask_iterator).path().u8string());

		// 2) get keypoints
		KeyPointMatches kpm = AlignImages(left, next);

		// 3) clear keypoints from dynamic objects

		for (auto& match : kpm.matches) {

			float u = (float(kpm.kp1.at(match.queryIdx).pt.x));
			float v = (float(kpm.kp1.at(match.queryIdx).pt.y));
			kp1.emplace_back(Point2f(u, v));
			kp2.emplace_back(Point2f(float(kpm.kp2.at(match.queryIdx).pt.x), float(kpm.kp1.at(match.queryIdx).pt.y)));
			
		}

		// 4) Estimate R and normalized t 

		Mat E, R, t, useless_mask;
		E = findEssentialMat(kp1, kp2, K, RANSAC, 0.999, 1.0, mask);
		cv::recoverPose(E, kp1, kp2, K, R, t, mask);

		// 5) Read corresponding speed from file
		for (int j = 0; j < STEP - 1; ++j) // заготовка на следующий шаг
			time >> next_time;
		dt = next_time - current_time;
		scale_factor = 0.001 * (speed / dt); // милисекунды
		t *= scale_factor;
		current_time = next_time; // на некст степ
		for (int j = 0; j < STEP - 1; ++j) // заготовка на следующий шаг
			speeds >> speed;

		R = R0 * R * R0.inv(); // переход в глобальную систему координат (пока не учитываем смещение t0)
		R.copyTo(local(Rect(0, 0, 3, 3)));
		t.copyTo(local(Rect(3, 0, 1, 3)));

		GLOBAL_COORD *= local;

		otp << GLOBAL_COORD.at<float>(0, 3) << " " << GLOBAL_COORD.at<float>(1, 3) << " " << GLOBAL_COORD.at<float>(2, 3) << endl;
		// 7) крутим
		advance(next_iterator, START_KEY_FRAME + STEP);
		advance(mask_iterator, START_KEY_FRAME + STEP);
		advance(left_iterator , START_KEY_FRAME + STEP);
	}
}

void GetData(const string& path_to_avi, int& i)
{
	VideoCapture cap(path_to_avi);
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
	}
	cout << path_to_avi << endl;
	Mat frame;
	string name;
	while (1) {
		cap >> frame;
		if (frame.empty())
			break;
		++i;
		cout << i << endl;
		if (i <= 9)
			name = "000" + to_string(i) + ".png";
		if (i >= 10 && i < 100)
			name = "00" + to_string(i) + ".png";
		if (i >= 100 && i < 1000)
			name = "0" + to_string(i) + ".png";
		if (i>=1000)
			name = to_string(i) + ".png";

		cout << name << endl;
		imwrite("D:/get.055/frames/" + name, frame);
	}
	cap.release();

}
int main(void) {

	float P0[] = { 7.0709120e+02, 0.0e+00, 6.018873e+02, 0.00e+00, 0.00e+00, 7.0709120e+02, 1.831104e+02, 0.00e+00, 0.0000e+00, 0.000e+00, 1.00e+00, 0.0000e+00 };
	float P1[] = { 7.0709120e+02, 0.0e+00, 6.018873e+02, -3.7981450e+02, 0.0e+00, 7.0709120e+02, 1.8311040e+02, 0.0e+00, 0.00e+00, 0.0e+00, 1.00e+00, 0.000e+00 };
	float K_data[] = { 1.1510823334853483e+03, 0., 9.7111620950416614e+02, 0., 1.1510823334853483e+03, 6.1461457012426672e+02, 0., 0., 1. };

	Mat K(3, 3, cv::DataType<float>::type, K_data);
	Vec3f rod = { -1.4829183338265911e-01, -1.4999999999999932e-02,
	   -9.2905726938532940e-02 };
	Mat R0;
	Rodrigues(rod, R0);
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	//cout << Decompose(P_right).cameraMatrix << endl;
	std::string folder_left = "D:/TRAMWAY/frames/";
	std::string segmented_left = "D:/TRAMWAY/segmented/";
	std::string output  = "D:/TRAMWAY/xyz.txt";
	std::vector dynamic_classes = { 11,12,13,14,15,16,17,18 };
	fstream speed("D:/TRAMWAY/closest_speed.txt");
	fstream times("D:/TRAMWAY/frame_time.txt");
	TramTest(folder_left, segmented_left, K, times, speed, output, R0);
	return 0;
}
