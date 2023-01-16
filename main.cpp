#include "lib.h"

void Display(Mat img)
{
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
			std::cout << img.at<float>(i, j) << " ";
		cout << endl;
	}
}
int main(void) {

	float P0[] = { 718.856f, 0.0f, 607.1928f, 0.0f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	float P1[] = { 718.856f, 0.0f, 607.1928f, -386.1448f, 0.0f, 718.856f, 185.2157f, 0.0f,0.0f,0.0f,1.0f, 0.0f };
	Mat P_left(3, 4, cv::DataType<float>::type, P0);
	Mat P_right(3, 4, cv::DataType<float>::type, P1);
	std::vector<int> dynamic_classes = {8,9,10};
	std::string folder_left = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_0/";
	std::string folder_right = "C:/Users/Andrey/Desktop/Data/lil_dataset/00/image_1";
	std::string segment_folder = "D:/Kitti_masks/sample_data/kitti_masks";
	std::string input  = "C:/Users/Andrey/Desktop/Data/optimized_results.txt";
	//VisualOdometry(folder_left, folder_right, input , P_left, P_right);
	EstimateAndOptimize(folder_left, folder_right, input, P_left, P_right);
	return 0;
}
