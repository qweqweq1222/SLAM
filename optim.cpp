#include "lib.h"
# define M_PI  3.14159265358979323846
Mat FromPointerToMat(double* pt)
{
	Mat T = Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			T.at<float>(i, j) = pt[3 * i + j];
	T.at<float>(0, 3) = pt[9];
	T.at<float>(1, 3) = pt[10];
	T.at<float>(1, 3) = pt[11];
	return T;
}

void Display(const Mat& mtx)
{
	for (int i = 0; i < mtx.rows; ++i)
	{
		for (int j = 0; j < mtx.cols; ++j)
			std::cout << mtx.at<float>(i, j) << " ";
		std::cout << endl;
	}
}
vector<double> GetAnglesAndVec(const Mat Rt)
{
	double alpha, beta, gamma;
	if(abs(Rt.at<float>(0, 2)) < 1)
		beta = asin(Rt.at<float>(0, 2));
	else if(Rt.at<float>(0,2) == 1)
		beta = M_PI / 2;
	else if(Rt.at<float>(0, 2) == -1)
		beta = -M_PI / 2;

	if(abs(Rt.at<float>(2, 2) / cos(beta)) < 1)
		alpha = acos(Rt.at<float>(2, 2) / cos(beta));
	else if (Rt.at<float>(2, 2) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(2, 2) / cos(beta) == -1)
		beta = M_PI;

	if(abs(Rt.at<float>(0, 0) / cos(beta)) < 1)
		gamma = acos(Rt.at<float>(0, 0) / cos(beta));
	else if (Rt.at<float>(0, 0) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(0, 0) / cos(beta) == -1)
		beta = M_PI;
	return { alpha, beta, gamma, Rt.at<float>(0,3), Rt.at<float>(1,3), Rt.at<float>(2,3) };
}
Mat GetRotation(Mat T)
{
	Mat rotation(Size(3, 3), CV_32F);
	rotation.at<float>(0, 0) = T.at<float>(0, 0);
	rotation.at<float>(0, 1) = T.at<float>(0, 1);
	rotation.at<float>(0, 2) = T.at<float>(0, 2);
	rotation.at<float>(1, 0) = T.at<float>(1, 0);
	rotation.at<float>(1, 1) = T.at<float>(1, 1);
	rotation.at<float>(1, 2) = T.at<float>(1, 2);
	rotation.at<float>(2, 0) = T.at<float>(2, 0);
	rotation.at<float>(2, 1) = T.at<float>(2, 1);
	rotation.at<float>(2, 2) = T.at<float>(2, 2);

	return rotation;
}
Mat ReconstructFromV4(double* alpha_trans)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	answer.at<float>(0, 0) = cos(alpha_trans[0]);
	answer.at<float>(0, 2) = sin(alpha_trans[0]);
	answer.at<float>(2, 0) = -sin(alpha_trans[0]);
	answer.at<float>(2, 2) = cos(alpha_trans[0]);
	answer.at<float>(0, 3) = alpha_trans[1];
	answer.at<float>(1, 3) = alpha_trans[2];
	answer.at<float>(2, 3) = alpha_trans[3];
	return answer;
}
Mat ReconstructFromV6(double* alpha_trans)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	double a = alpha_trans[0];
	double b = alpha_trans[1];
	double g = alpha_trans[2];
	answer.at<float>(0, 0) = cos(b) * cos(g);
	answer.at<float>(0, 1) = -sin(g) * cos(b);
	answer.at<float>(0, 2) = sin(b);
	answer.at<float>(1, 0) = sin(a) * sin(b) * cos(g) + sin(g) * cos(a);
	answer.at<float>(1, 1) = -sin(a) * sin(b) * sin(g) + cos(g) * cos(a);
	answer.at<float>(1, 2) = -sin(a) * cos(b);
	answer.at<float>(2, 0) = sin(a) * sin(g) - sin(b) * cos(a) * cos(g);
	answer.at<float>(2, 1) = sin(a) * cos(g) + sin(b) * sin(g) * cos(a);
	answer.at<float>(2, 2) = cos(a) * cos(b);
	answer.at<float>(0, 3) = alpha_trans[3];
	answer.at<float>(1, 3) = alpha_trans[4];
	answer.at<float>(2, 3) = alpha_trans[5];
	return answer;
}
void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight)
{
	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);
	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);

	std::advance(left_iterator, START_KEY_FRAME);
	std::advance(right_iterator, START_KEY_FRAME);
	std::advance(next_iterator, START_KEY_FRAME + 1);
	std::vector<std::vector<KeyPoint>> vec;
	std::ofstream in(input);
	int buffer = 0;
	int counter = 0;
	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);

	for (int i = 0; i < NUM_OF_FRAMES; i += buffer) // глобальный цикл, в котором бежим по батчам фреймов (локальная оптимизация )
	{
		vector<vector<double>> location;
		fs::directory_iterator copy_left(left_path); 
		fs::directory_iterator copy_next(left_path);

		std::advance(copy_next, START_KEY_FRAME + counter + 1);
		std::advance(copy_left, START_KEY_FRAME + counter);

		vec = GetSamePoints(copy_left, copy_next, SAME_POINTS);
		vector<vector<KeyPoint>> alternative(vec.size(), vector<KeyPoint>(0));
		vector<Vec3f> pts3d;
		counter += vec.size();
		buffer = vec.size();
		Mat P = Mat::eye(4, 4, CV_32F);

		for (int jdx = 0; jdx < vec.size() - 1; ++jdx)
		{

			std::pair<Mat, Mat> buffer = EstimateMotion(imread((*left_iterator).path().u8string()),
				imread((*right_iterator).path().u8string()),
				imread((*next_iterator).path().u8string()), PLeft, PRight);
			if (jdx == 0)
			{
				Mat disparity = CalculateDisparity(imread((*left_iterator).path().u8string()), imread((*next_iterator).path().u8string()));
				float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));

				for (int p = 0; p < vec[0].size(); ++p) {
					float u = (float(vec[0][p].pt.x));
					float v = (float(vec[0][p].pt.y));
					float z = coeff / disparity.at<float>(int(v), int(u));
					if (z > 0 && z < DEPTH_TRASH)
					{
						float x = z * (v - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
						float y = z * (u - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
						for (int idx = 0; idx < vec.size(); ++idx)
							alternative[idx].push_back(vec[idx][p]);
						pts3d.push_back({ x,y,z });
					}

				}
			}
			++left_iterator;
			++right_iterator;
			++next_iterator;
			Mat rot;
			cv::Rodrigues(buffer.first, rot);
			rot.convertTo(rot, CV_32F);
			buffer.second.convertTo(buffer.second, CV_32F);
			Mat L = T(rot, buffer.second);
			P *= L.inv();
			std::vector<double> a_t = GetAnglesAndVec(P.inv());
			location.push_back(a_t);

		}
		ceres::Problem problem;
		const int b = location.size();
		vector<double*> alphas_trans_(location.size());
		for (int i = 0; i < location.size(); ++i)
		{
			alphas_trans_[i] = new double[6];
			alphas_trans_[i][0] = location[i][0];
			alphas_trans_[i][1] = location[i][1];
			alphas_trans_[i][2] = location[i][2];
			alphas_trans_[i][3] = location[i][3];
			alphas_trans_[i][4] = location[i][4];
			alphas_trans_[i][5] = location[i][5];
		}
		std::vector<double*> pts_3d(pts3d.size());
		for (int i = 0; i < pts3d.size(); ++i)
		{
			pts_3d[i] = new double[3];
			pts_3d[i][0] = pts3d[i][0];
			pts_3d[i][1] = pts3d[i][1];
			pts_3d[i][2] = pts3d[i][2];
		}
		
		for (int i = 0; i < alternative.size() - 1; ++i)
		{
			for (int j = 0; j < pts3d.size(); ++j)
			{
			
					/* imbalance checker */

					double P3[3];
					double a = alphas_trans_[i][0];
					double b = alphas_trans_[i][1];
					double g = alphas_trans_[i][2];
					double dx = pts_3d[j][0] - alphas_trans_[i][3];
					double dy = pts_3d[j][1] - alphas_trans_[i][4];
					double dz = pts_3d[j][2] - alphas_trans_[i][5];
					P3[0] = (cos(b) * cos(g)) * dx + (sin(a) * sin(b) * cos(g) + sin(g) * cos(a)) * dy + (sin(a) * sin(g) - sin(b) * cos(a) * cos(g)) * dz;
					P3[1] = (-sin(g) * cos(b)) * dx + (cos(a) * cos(g) - sin(a) * sin(b) * sin(g)) * dy + (sin(a) * cos(g) + sin(b) * sin(g) * cos(a)) * dz;
					P3[2] = (sin(b)) * dx + (-sin(a) * cos(b)) * dy + (cos(a) * cos(b)) * dz;
					double predicted_x = PLeft.at<float>(0, 0) * P3[0] / P3[2] + PLeft.at<float>(0, 2);
					double predicted_y = PLeft.at<float>(0, 0) * P3[1] / P3[2] + PLeft.at<float>(1, 2);

				

					bool positive = predicted_x >= 0 && predicted_y >= 0;
					bool diff = abs(predicted_x - alternative[i + 1][j].pt.y) < 50 && abs(predicted_y - alternative[i + 1][j].pt.x) < 50;
					bool condition = positive && diff;

					if (condition)
					{
						for (int iqwe = 0; iqwe < 6; ++iqwe)
							cout << alphas_trans_[i][iqwe] << " ";
							cout << endl;
							ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(alternative[i + 1][j].pt.y), double(alternative[i + 1][j].pt.x),
									PLeft.at<float>(0, 0), PLeft.at<float>(0, 0), PLeft.at<float>(0, 2), PLeft.at<float>(1, 2), pts3d[j]);
							problem.AddResidualBlock(cost_function, nullptr, alphas_trans_[i] /*pts_3d[j]*/);
						
					}
				
			}
		}
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

	
		for (int unique_i = 0; unique_i < location.size(); ++unique_i)
		{
			Mat copy_GLOBAL = GLOBAL_P.clone();
			copy_GLOBAL *= ReconstructFromV6(alphas_trans_[unique_i]).inv();
			in << copy_GLOBAL.at<float>(0, 3) << " " << copy_GLOBAL.at<float>(1, 3) << " " << copy_GLOBAL.at<float>(2, 3) << "\n";
			if (unique_i == location.size() - 1)
				GLOBAL_P *= ReconstructFromV6(alphas_trans_[unique_i]).inv();
			
		}

		for (auto &p : alphas_trans_)
			delete[] p;
		for (auto& d : pts_3d)
			delete[] d;

	}
}
