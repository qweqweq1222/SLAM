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
Mat Recover(double* pt)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	answer.at<float>(0, 0) = cos(pt[0]);
	answer.at<float>(0, 2) = -sin(pt[0]);
	answer.at<float>(2, 0) = sin(pt[0]);
	answer.at<float>(2, 2) = cos(pt[0]);
	answer.at<float>(0, 3) = -pt[1];
	answer.at<float>(2, 3) = -pt[2];
	return answer;
}
vector<double> GetAnglesAndVec(const Mat Rt) // получаем углы из матрицы поворота и вектор t 
{
	double alpha, beta, gamma;
	if (abs(Rt.at<float>(0, 2)) < 1)
		beta = asin(Rt.at<float>(0, 2));
	else if (Rt.at<float>(0, 2) == 1)
		beta = M_PI / 2;
	else if (Rt.at<float>(0, 2) == -1)
		beta = -M_PI / 2;

	if (abs(Rt.at<float>(2, 2) / cos(beta)) < 1)
		alpha = acos(Rt.at<float>(2, 2) / cos(beta));
	else if (Rt.at<float>(2, 2) / cos(beta) == 1)
		beta = 0;
	else if (Rt.at<float>(2, 2) / cos(beta) == -1)
		beta = M_PI;

	if (abs(Rt.at<float>(0, 0) / cos(beta)) < 1)
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

Mat ReconstructRFromV5(double* alpha_trans, double alpha)
{
	Mat answer = Mat::eye(3, 3, CV_32F);
	double a = alpha;
	double b = alpha_trans[0];
	double g = alpha_trans[1];
	answer.at<float>(0, 0) = cos(b) * cos(g);
	answer.at<float>(0, 1) = -sin(g) * cos(b);
	answer.at<float>(0, 2) = sin(b);
	answer.at<float>(1, 0) = sin(a) * sin(b) * cos(g) + sin(g) * cos(a);
	answer.at<float>(1, 1) = -sin(a) * sin(b) * sin(g) + cos(g) * cos(a);
	answer.at<float>(1, 2) = -sin(a) * cos(b);
	answer.at<float>(2, 0) = sin(a) * sin(g) - sin(b) * cos(a) * cos(g);
	answer.at<float>(2, 1) = sin(a) * cos(g) + sin(b) * sin(g) * cos(a);
	answer.at<float>(2, 2) = cos(a) * cos(b);
	return answer;
}

Mat PACK(Mat& R, float x, float y, float z)
{
	Mat answer = Mat::eye(4, 4, CV_32F);
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			answer.at<float>(i, j) = R.at<float>(i, j);
	answer.at<float>(0, 3) = x;
	answer.at<float>(1, 3) = y;
	answer.at<float>(2, 3) = z;
	return answer;
}
/*void EstimateAndOptimize(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight)
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
				Mat disparity = CalculateDisparity(imread((*left_iterator).path().u8string()), imread((*right_iterator).path().u8string()));
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
				// проверка на мусорные точки
				// т.к. работаем с обратной матрицей R^-1 (обратной относительно той, для которой вычисляем смещение для одометрии), то формулы получаются R^-1x + t 
				Mat R = (ReconstructFromV6(alphas_trans_[i]))(cv::Rect(0, 0, 3, 3)); //вырезаем матрицу поворота
				Mat Rx = R * Vec3f(pts_3d[j][0], pts_3d[j][1], pts_3d[j][2]); // 1x3 
				double predicted_x = 718.856 * (Rx.at<float>(0, 0) + alphas_trans_[i][3]) / (Rx.at<float>(0, 2) + alphas_trans_[i][5]) + 607.1928;
				double predicted_y = 718.856 * (Rx.at<float>(0, 1) + alphas_trans_[i][4]) / (Rx.at<float>(0, 2) + alphas_trans_[i][5]) + 185.2157;


				bool positive = predicted_x >= 0 && predicted_y >= 0; // проверка на мусор 
				bool diff = abs(predicted_x - alternative[i + 1][j].pt.y) < 50 && abs(predicted_y - alternative[i + 1][j].pt.x) < 50;
				bool condition = positive && diff;
				if (condition)
				{
					ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(alternative[i + 1][j].pt.y), double(alternative[i + 1][j].pt.x),
						718.856, 718.856, 607.1928, 185.2157, pts3d[j]);
					problem.AddResidualBlock(cost_function, nullptr, alphas_trans_[i]);

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

		for (auto& p : alphas_trans_)
			delete[] p;
		for (auto& d : pts_3d)
			delete[] d;

		cout << i << endl;
	}
}*/


void OdometryALL(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight,std::vector<int>& dynamic, const std::string& segment)
{
	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);
	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);

	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);
	fs::directory_iterator segment_iterator(segment);
	::advance(left_iterator, START_KEY_FRAME); // если хотим начать не с 0-го кадра 
	::advance(right_iterator, START_KEY_FRAME);
	::advance(segment_iterator, START_KEY_FRAME);
	::advance(next_iterator, START_KEY_FRAME + 10);
	++next_iterator;

	std::ofstream in(input);

	float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));

	for (int i = 0; i < NUM_OF_FRAMES; i+=10)
	{
		ceres::Problem problem;

		Mat left = imread((*left_iterator).path().u8string());
		Mat right = imread((*right_iterator).path().u8string());
		Mat next = imread((*next_iterator).path().u8string());

		Mat mask = imread((*segment_iterator).path().u8string());
		cv::cvtColor(mask, mask, CV_BGR2GRAY); // без этого каста маски неправильно считываются


		pair<Mat, Mat> TR = EstimateNoDynamicMotion(left,right,next,mask,PLeft,PRight,dynamic);
		TR.first.convertTo(TR.first, CV_32F); // здесь были ошибки с типами до без этих строк  - Ransac возвращает в дабле
		TR.second.convertTo(TR.second, CV_32F); // здесь были ошибки с типами до без этих строк  - Ransac возвращает в дабле

		Mat R;
		cv::Rodrigues(TR.first, R);
		R.convertTo(R, CV_32F);
		Mat Trap = TAP(R, TR.second);
		// фул оптимизация 3 угла (не родригес) + 3 координаты
		vector<double> angles_vecs = GetAnglesAndVec(Trap); 
		// получили начальную точку для оптимизации 
		double initial_point[6];
		vector<Vec3f> pts3d; // 3d в референсном фрейме (в данном случае только 2 фрейма, поэтому в СК первого фрейма)
		vector<Vec2f> pts2d; // 2d точки во втором фрейме
		for (int i = 0; i < 6; ++i)
			initial_point[i] = angles_vecs[i];
		// теперь надо получить координаты точек в 3d и 2d 
		KeyPointMatches matcher = AlignImages(left, next);
		Mat disparity = CalculateDisparity(left, right);
		for (auto& match : matcher.matches) {

			float u = (float(matcher.kp1.at(match.queryIdx).pt.x));
			float v = (float(matcher.kp1.at(match.queryIdx).pt.y));
			float z = coeff / disparity.at<float>(int(v), int(u));
			if (z < DEPTH_TRASH && z > 0) {

				float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
				float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
				pts3d.emplace_back(Point3f{ x, y, z });
				pts2d.emplace_back(Vec2f(matcher.kp2.at(match.trainIdx).pt.x, matcher.kp2.at(match.trainIdx).pt.y));
			}
		}
		vector<double* > pts_for;
		for (int m = 0; m < pts3d.size(); ++m)
		{
			double* mat = new double[3];
			mat[0] = pts3d[m][0];
			mat[1] = pts3d[m][1];
			mat[2] = pts3d[m][2];
			pts_for.emplace_back(mat);
			/*pts_for[m][0] = pts3d[m][0];
			pts_for[m][1] = pts3d[m][1];
			pts_for[m][2] = pts3d[m][2];*/
		}
		for (int j = 0; j < pts3d.size(); ++j)
		{
			// проверка на мусорные точки
			double vec_[6];
			for (int k = 0; k < 6; ++k)
				vec_[k] = initial_point[k];
			Mat Res = ReconstructFromV6(vec_)(cv::Rect(0, 0, 3, 3)); //вырезаем матрицу поворота
			Mat Rx = Res * Vec3f(pts3d[j][0], pts3d[j][1], pts3d[j][2]); // 1x3 
			double predicted_x = 707.09119 * (Rx.at<float>(0, 0) + vec_[3]) / (Rx.at<float>(0, 2) + vec_[5]) + 601.88733;
			double predicted_y = 707.09119 * (Rx.at<float>(0, 1) + vec_[4]) / (Rx.at<float>(0, 2) + vec_[5]) + 183.1104;


			//std::cout << predicted_x << " " << pts2d[j][0] << endl;
			//std::cout << predicted_y << " " << pts2d[j][1] << endl;
			bool positive = predicted_x >= 0 && predicted_y >= 0; // проверка на мусор 
			bool diff = abs(predicted_x - pts2d[j][0]) < 50 && abs(predicted_y - pts3d[j][1]) < 50; // более легкая проверка на мусор
			bool condition = positive && diff;
			if (condition)
			{
				ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(pts2d[j][0]), double(pts2d[j][1]),
					707.09119, 707.09119, 601.88733, 183.1104, pts3d[j]);
				problem.AddResidualBlock(cost_function, nullptr, initial_point);

				double start[6];
				for (int p = 0; p < 6; ++p)
					start[p] = initial_point[p];

				for (int p = 0; p < 6; ++p) // ограниения на изменение значения - +- 10% по координате и +-5% по углу
				{
					if (p < 3)
					{
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.80);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.20);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.20);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.80);
						}
					}
					else
					{
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.8);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.20);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.20);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.80);
						}
					}
				}
				for (int p = 0; p < 3; ++p) // ограниения на изменение значения - +- 10% по координате и +-5% по углу
				{
	
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.80);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.20);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.20);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.80);
						}
				}
			}
		}
		for (int p = 0; p < 6; ++p)
			 ::cout << initial_point[p] << ",";
		::cout << endl;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		options.logging_type = ceres::SILENT;
		options.minimizer_progress_to_stdout = false;
		ceres::Solve(options, &problem, &summary);
		for (int p = 0; p < 6; ++p)
			::cout << initial_point[p] << ",";
		::cout << endl;
		::cout << "--------------------------\n";
		Mat eval = ReconstructFromV6(initial_point);
		GLOBAL_P *= eval.inv();
		in << GLOBAL_P.at<float>(0, 3) << " " << GLOBAL_P.at<float>(1, 3) << " " << GLOBAL_P.at<float>(2, 3) << "\n";
		std::cout << "i:" << i << endl;
		advance(left_iterator, START_KEY_FRAME + 10);
		advance(next_iterator, START_KEY_FRAME + 10);
		advance(right_iterator, START_KEY_FRAME + 10);
		advance(segment_iterator, START_KEY_FRAME + 10);
		/*++left_iterator;
		++right_iterator;
		++next_iterator;
		++segment_iterator;
		*/
		cout << "_____________________\n";
	}
}
/*void SimplifiedOdometry(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft, const Mat& PRight,
	std::fstream& myfile_)
{
	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);
	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);

	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);

	//advance(left_iterator, START_KEY_FRAME); // если хотим начать не с начала 
	//advance(right_iterator, START_KEY_FRAME);
	//advance(next_iterator, START_KEY_FRAME + 1);
	++next_iterator;

	std::ofstream in(input);

	float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));

	for (int i = 0; i < NUM_OF_FRAMES; ++i)
	{
		ceres::Problem problem;
		Mat left = imread((*left_iterator).path().u8string());
		Mat right = imread((*right_iterator).path().u8string());
		Mat next = imread((*next_iterator).path().u8string());

		pair<Mat, Mat> TR = EstimateMotion(left, right, next, PLeft, PRight);
		TR.first.convertTo(TR.first, CV_32F); // здесь были ошибки с типами до без этих строк  - Ransac возвращает в дабле
		TR.second.convertTo(TR.second, CV_32F);

		Mat R;
		cv::Rodrigues(TR.first, R);
		R.convertTo(R, CV_32F);
		Mat Trap = TAP(R, TR.second);
		vector<double> angles_vecs = GetAnglesAndVec(Trap); // 3 угла (не родригес) + 3 координаты
		string y_str;
		myfile_ >> y_str;
		double y = -atof(y_str.c_str());
		// получили начальную точку для оптимизации 
		double initial_point[3];
		vector<Vec3f> pts3d; // 3d
		vector<Vec2f> pts2d; // 2d
		initial_point[0] = angles_vecs[1]; // вращение вокруг OY
		initial_point[1] = angles_vecs[3]; // x
		initial_point[2] = angles_vecs[5]; // z
		// теперь надо получить координаты точек в 3d и 2d 
		KeyPointMatches matcher = AlignImages(left, next);
		Mat disparity = CalculateDisparity(left, right);
		for (auto& match : matcher.matches) {

			float u = (float(matcher.kp1.at(match.queryIdx).pt.x));
			float v = (float(matcher.kp1.at(match.queryIdx).pt.y));
			float z = coeff / disparity.at<float>(int(v), int(u));
			if (z < DEPTH_TRASH && z > 0) {

				float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
				float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
				pts3d.emplace_back(Point3f{ x, y, z });
				pts2d.emplace_back(Vec2f(matcher.kp2.at(match.trainIdx).pt.y, matcher.kp2.at(match.trainIdx).pt.x));
			}
		}

		for (int j = 0; j < pts3d.size(); ++j)
		{
			// проверка на мусорные точки
			double vec_[6];
			vec_[0] = 0;
			vec_[1] = initial_point[0];
			vec_[2] = 0;
			vec_[3] = initial_point[1];
			vec_[4] = y;
			vec_[5] = initial_point[2];
			Mat Res = ReconstructFromV6(vec_)(cv::Rect(0, 0, 3, 3)); //вырезаем матрицу поворота
			Mat Rx = R * Vec3f(pts3d[j][0], pts3d[j][1], pts3d[j][2]); // 1x3 
			double predicted_x = 718.856 * (Rx.at<float>(0, 0) + vec_[3]) / (Rx.at<float>(0, 2) + vec_[5]) + 607.1928;
			double predicted_y = 718.856 * (Rx.at<float>(0, 1) + vec_[4]) / (Rx.at<float>(0, 2) + vec_[5]) + 185.2157;

			bool positive = predicted_x >= 0 && predicted_y >= 0; // проверка на мусор 
			bool diff = abs(predicted_x - pts2d[j][0]) < 50 && abs(predicted_y - pts3d[j][1]) < 50;
			bool condition = positive && diff;
			if (condition)
			{
				ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(double(pts2d[j][0]), double(pts2d[j][1]),
					718.856, 718.856, 607.1928, 185.2157, pts3d[j]);
				problem.AddResidualBlock(cost_function, nullptr, initial_point);

				double start[3];
				for (int p = 0; p < 3; ++p)
					start[p] = initial_point[p];

				for (int p = 0; p < 3; ++p) // ограниения на изменение значения - +- 20%
				{
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.95);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.05);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.05);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.95);
						}
				}
			}

			ceres::Solver::Options options;
			options.linear_solver_type = ceres::DENSE_SCHUR;
			options.minimizer_progress_to_stdout = true;
			ceres::Solver::Summary summary;
			options.logging_type = ceres::SILENT;
			options.minimizer_progress_to_stdout = false;
			ceres::Solve(options, &problem, &summary);
			Mat eval = ReconstructFromV6(initial_point);
			GLOBAL_P *= eval.inv();
			in << GLOBAL_P.at<float>(0, 3) << " " << GLOBAL_P.at<float>(, 3) << " " << GLOBAL_P.at<float>(2, 3) << "\n";
			std::cout << "i:" << i << endl;
			++left_iterator;
			++right_iterator;
			++next_iterator;
	}
}*/

void OdometryAXZ(const std::string& left_path, const std::string& right_path, const std::string& input, const Mat& PLeft,
	const Mat& PRight, std::vector<int>& dynamic, const std::string& segment, const std::string& y_coord)
{
	Mat GLOBAL_P = Mat::eye(4, 4, CV_32F);
	CameraInfo cil = Decompose(PLeft);
	CameraInfo cir = Decompose(PRight);

	fs::directory_iterator left_iterator(left_path);
	fs::directory_iterator right_iterator(right_path);
	fs::directory_iterator next_iterator(left_path);
	fs::directory_iterator segment_iterator(segment);
	advance(left_iterator, START_KEY_FRAME); // если хотим начать не с 0-го кадра 
	advance(right_iterator, START_KEY_FRAME);
	advance(segment_iterator, START_KEY_FRAME);
	advance(next_iterator, START_KEY_FRAME + 10);
	std::fstream myfile(y_coord, std::ios_base::in);
	std::ofstream in(input);

	float coeff = cil.cameraMatrix.at<float>(0, 0) * (cir.transVector.at<float>(0) - cil.transVector.at<float>(0));

	for (int i = 0; i < NUM_OF_FRAMES; i += 10)
	{
		ceres::Problem problem;

		Mat left = imread((*left_iterator).path().u8string());
		Mat right = imread((*right_iterator).path().u8string());
		Mat next = imread((*next_iterator).path().u8string());

		Mat mask = imread((*segment_iterator).path().u8string());
		cv::cvtColor(mask, mask, CV_BGR2GRAY); // без этого каста маски неправильно считываются


		pair<Mat, Mat> TR = EstimateNoDynamicMotion(left, right, next, mask, PLeft, PRight, dynamic);
		TR.first.convertTo(TR.first, CV_32F); // здесь были ошибки с типами до без этих строк  - Ransac возвращает в дабле
		TR.second.convertTo(TR.second, CV_32F); // здесь были ошибки с типами до без этих строк  - Ransac возвращает в дабле

		Mat R;
		cv::Rodrigues(TR.first, R);
		R.convertTo(R, CV_32F);
		Mat Trap = TAP(R, TR.second);
		// оптимизируем по углу покруг оси y (вертикальная ось) и
		vector<double> angles_vecs = GetAnglesAndVec(Trap);
		// получили начальную точку для оптимизации 
		double initial_point[3];
		vector<Vec3f> pts3d; // 3d в референсном фрейме (в данном случае только 2 фрейма, поэтому в СК первого фрейма)
		vector<Vec2f> pts2d; // 2d точки во втором фрейме
		initial_point[0] = angles_vecs[1];
		initial_point[1] = angles_vecs[3];
		initial_point[2] = angles_vecs[5];

		double y;
		if (i == 0)
		{
			myfile >> y;
			myfile >> y;
		}
		else
		{
			myfile >> y;
		}
		// теперь надо получить координаты точек в 3d и 2d 
		KeyPointMatches matcher = AlignImages(left, next);
		Mat disparity = CalculateDisparity(left, right);
		for (auto& match : matcher.matches) {

			float u = (float(matcher.kp1.at(match.queryIdx).pt.x));
			float v = (float(matcher.kp1.at(match.queryIdx).pt.y));
			float z = coeff / disparity.at<float>(int(v), int(u));
			if (z < DEPTH_TRASH && z > 0) {

				float x = z * (u - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
				float y = z * (v - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
				pts3d.emplace_back(Point3f{ x, y, z });
				pts2d.emplace_back(Vec2f(matcher.kp2.at(match.trainIdx).pt.x, matcher.kp2.at(match.trainIdx).pt.y));
			}
		}
		for (int j = 0; j < pts3d.size(); ++j)
		{
			// проверка на мусорные точки
			/*
			double vec_[6];
			vec_[0] = 0;
			vec_[1] = initial_point[0];
			vec_[2] = 0;
			vec_[3] = initial_point[1];
			vec_[4] = y;
			vec_[5] = initial_point[2];

			Mat Res = ReconstructFromV6(vec_)(cv::Rect(0, 0, 3, 3)); //вырезаем матрицу поворота
			Mat Rx = Res * Vec3f(pts3d[j][0], pts3d[j][1], pts3d[j][2]); // 1x3 
			double predicted_x = 707.09119 * (Rx.at<float>(0, 0) + vec_[3]) / (Rx.at<float>(0, 2) + vec_[5]) + 601.88733;
			double predicted_y = 707.09119 * (Rx.at<float>(0, 1) + vec_[4]) / (Rx.at<float>(0, 2) + vec_[5]) + 183.1104;


			//std::cout << predicted_x << " " << pts2d[j][0] << endl;
			//std::cout << predicted_y << " " << pts2d[j][1] << endl;
			bool positive = predicted_x >= 0 && predicted_y >= 0; // проверка на мусор 
			bool diff = abs(predicted_x - pts2d[j][0]) < 50 && abs(predicted_y - pts3d[j][1]) < 50; // более легкая проверка на мусор
			bool condition = positive && diff;*/
				ceres::CostFunction* cost_function = SnavelyReprojectionError_::Create(double(pts2d[j][0]), double(pts2d[j][1]),
					707.09119, 707.09119, 601.88733, 183.1104, pts3d[j], y);
				problem.AddResidualBlock(cost_function, nullptr, initial_point);

				/*
				double start[3];
				for (int p = 0; p < 3; ++p)
					start[3] = initial_point[3];

				for (int p = 0; p < 3; ++p) // ограниения на изменение значения - +- 10% по координате и +-5% по углу
				{
					if (p == 0)
					{
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.95);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.05);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.05);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.95);
						}
					}
					else
					{
						if (start[p] >= 0)
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 0.9);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 1.10);
						}
						else
						{
							problem.SetParameterLowerBound(initial_point, p, start[p] * 1.10);
							problem.SetParameterUpperBound(initial_point, p, start[p] * 0.90);
						}
					}
				}*/

		}
		// отрешиваем первую проблему
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		//отрешиваем вторую проблему - много копипасты
		/*
		ceres::Problem problem2;
		vector<Vec3f> pts_on_ground;
		for (int k = 0; k < 100; ++k)
		{

			int i = mask.rows * 0.9 + rand() % int(0.1 * mask.rows);
			int j = mask.cols * 2 / 5 + rand() % int(mask.cols / 5);
			if (int(mask.at<uchar>(i, j) == 0))
			{
				float z = coeff / disparity.at<float>(int(i), int(j));
				if (z < DEPTH_TRASH && z > 0) {

					float x = z * (i - cil.cameraMatrix.at<float>(0, 2)) / cil.cameraMatrix.at<float>(0, 0);
					float y = z * (j - cil.cameraMatrix.at<float>(1, 2)) / cil.cameraMatrix.at<float>(1, 1);
					pts_on_ground.emplace_back(Point3f{ x, y, z });
				}
			}
		}


		for (int j = 0; j < pts_on_ground.size(); ++j)
		{

			cout << " here\n";
			ceres::CostFunction* cost_function = SnavelyReprojectionError__::Create(pts_on_ground[j], y, pts_on_ground[j][2]);
			problem2.AddResidualBlock(cost_function, nullptr, initial_point);

			double start[3];
			for (int p = 0; p < 3; ++p)
				start[3] = initial_point[3];

			for (int p = 0; p < 3; ++p) // ограниения на изменение значения - +- 10% по координате и +-5% по углу
			{
				if (p == 0)
				{
					if (start[p] >= 0)
					{
						problem2.SetParameterLowerBound(initial_point, p, start[p] * 0.90);
						problem2.SetParameterUpperBound(initial_point, p, start[p] * 1.10);
					}
					else
					{
						problem2.SetParameterLowerBound(initial_point, p, start[p] * 1.10);
						problem2.SetParameterUpperBound(initial_point, p, start[p] * 0.90);
					}
				}
				else
				{
					if (start[p] >= 0)
					{
						problem2.SetParameterLowerBound(initial_point, p, start[p] * 0.9);
						problem2.SetParameterUpperBound(initial_point, p, start[p] * 1.10);
					}
					else
					{
						problem2.SetParameterLowerBound(initial_point, p, start[p] * 1.10);
						problem2.SetParameterUpperBound(initial_point, p, start[p] * 0.90);
					}
				}
			}

		} // описали проблему 

		ceres::Solver::Options options2;
		options2.linear_solver_type = ceres::DENSE_SCHUR;
		options2.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary2;
		ceres::Solve(options2, &problem2, &summary2);
		*/
		std::cout << "i:" << i << endl;
		double v6[] = { 0, initial_point[0], 0, initial_point[1], y, initial_point[2] };
		Mat eval = ReconstructFromV6(v6); // восстанавливаем по вектору матрицу 4 на 4 
		eval = eval.inv(); // инвертируем 
		eval.at<float>(1, 3) = y; // заменяем на gt координату по y 

		GLOBAL_P *= eval;
		in << GLOBAL_P.at<float>(0, 3) << " " << y << " " << GLOBAL_P.at<float>(2, 3) << "\n";
		advance(left_iterator, START_KEY_FRAME + 10);
		advance(next_iterator, START_KEY_FRAME + 10);
		advance(right_iterator, START_KEY_FRAME + 10);
		advance(segment_iterator, START_KEY_FRAME + 10);
		for (int i = 0; i < 9; ++i)
			myfile >> y;

	}
}
