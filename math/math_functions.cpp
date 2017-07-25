// Author: Xinshuo
// Email: xinshuow@andrew.cmu.edu

// standard c++ library
#include <random>
//#include <vector>
#include <stdarg.h>

// in-project library
#include "math_functions.h"
#include "type_conversion.h"
#include "debug_tool.h"

/*********************************************** I/O ******************************************************/
void read_matrix(FILE *califp, cv::Mat& m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			int ret;
			ASSERT_WITH_MSG(m.type() == CV_32F || CV_64F, "Current read_matrix function only support CV_32F and CV_64F. Please check the type.");
			if (m.type() == CV_32F)	
				ret = fscanf(califp, "%f", &m.at<float>(i, j));
			else if (m.type() == CV_64F)
				ret = fscanf(califp, "%lf", &m.at<double>(i, j));
			else

			ASSERT_WITH_MSG(ret == 1, "Calibration file not found or the size of parameter to read is not correct. Please check!");
		}
	}
}


/*********************************************** basic ******************************************************/
double get_median(std::vector<double>& a) {
	ASSERT_WITH_MSG(a.size() > 0, "The size of input vector should be bigger than 0 while calculating median.");
	sort(a.begin(), a.end());
	if (a.size() % 2 == 0) {
		return (a[a.size() / 2 - 1] + a[a.size() / 2]) / 2;
	}
	return a[a.size() / 2];
}

bool generate_weighted_randomization(std::vector<double>& weights, std::vector<int>& samples, int seed) {
	ASSERT_WITH_MSG(weights.size() > 0, "The size of input weight vector should be bigger than 0 while generating weighted random number.");
	ASSERT_WITH_MSG(samples.size() >= 1, "The size of random samples request should be larger than 0 while generating weighted random number. \
		Please prereallocate it.");

	bool all_zeros = true;	// if true, all weights are zero
	for (int i = 0; i < weights.size(); i++) {
		ASSERT_WITH_MSG(weights[i] >= 0, "The weights for generating weighted random number should be larger than or equal to 0.");
		if (weights[i] != 0)
			all_zeros = false;
	}
	
	if (all_zeros) {
		//print_vec(samples);
		std::srand(seed);
		samples = random_sample(weights, samples.size(), std::rand());
		//print_vec(samples);
	}
	else {
		//print_vec(weights);
		std::discrete_distribution<int> dist(std::begin(weights), std::end(weights));
		std::mt19937 gen;
		gen.seed(seed);		// set random seed
		//std::vector<int> samples_tmp(size);

		bool partial_out = false;	// if true, part of selected indexes are out of index range, which is impossible when all_zeros is false
		for (int i = 0; i < samples.size(); i++) {
			int new_sample = dist(gen);
			//print_sca(new_sample);
			//print_sca(i);
			samples[i] = new_sample;
			////if (new_sample == weights.size())
			////	partial_out = true;
		}
	}

	//print_vec(samples);
	return all_zeros;

	//print_vec(samples);
	//system("PAUSE");
	//std::cout << all_out << std::endl;
	//std::cout << partial_out << std::endl;
	//ASSERT_WITH_MSG((partial_out && all_zeros) || (!partial_out && !all_zeros), "Error while generating weighted random number!");

}


std::vector<int> random_sample(std::vector<double>& samples, int size, int seed) {
	ASSERT_WITH_MSG(samples.size() > 0, "The size of input weight vector should be bigger than 0 while random sample.");
	ASSERT_WITH_MSG(size >= 1, "The size of random number request should be larger or equal than 1 while random sample.");
	std::srand(seed);		// set random seed
	std::vector<int> selected(size);
	for (auto& i : selected)
		i = std::rand() % samples.size();	// random sample with equal probability
	return selected;
}



/*********************************************** algebra ******************************************************/
double l2_norm(std::vector<double>& vec) {
	ASSERT_WITH_MSG(vec.size() > 0, "The size of input vector should be bigger than 0 while calculating l2 norm!");
	double accum = 0.;
	for (int i = 0; i < vec.size(); ++i) {
		accum += vec[i] * vec[i];
	}
	return sqrt(accum);
}
double l2_norm(std::vector<float>& vec) {
	ASSERT_WITH_MSG(vec.size() > 0, "The size of input vector should be bigger than 0 while calculating l2 norm!");
	std::vector<double> double_vec_tmp = float2double_vec(vec);
	return l2_norm(double_vec_tmp);
}

std::vector<double> cross(std::vector<double>& a, std::vector<double>& b) {
	ASSERT_WITH_MSG(a.size() == b.size(), "The size of vector to do cross product is not equal!");
	ASSERT_WITH_MSG(a.size() == 3, "current cross product function only support vector with size 3!");
	std::vector<double> c(3);
	c[0] = a[1] * b[2] - b[1] * a[2];
	c[1] = a[2] * b[0] - b[2] * a[0];
	c[2] = a[0] * b[1] - b[0] * a[1];
	return c;
}
std::vector<float> cross(std::vector<float>& a, std::vector<float>& b) {
	ASSERT_WITH_MSG(a.size() == b.size(), "The size of vector to do cross product is not equal!");
	ASSERT_WITH_MSG(a.size() == 3, "current cross product function only support vector with size 3!");

	std::vector<double> double_vec_a = float2double_vec(a);
	std::vector<double> double_vec_b = float2double_vec(b);
	std::vector<double> double_vec_tmp = cross(double_vec_a, double_vec_b);
	return double2float_vec(double_vec_tmp);
}

double inner(std::vector<double>& a, std::vector<double>& b) {
	ASSERT_WITH_MSG(a.size() > 0, "The size of input vector should be bigger than 0 while calculating inner product!");
	ASSERT_WITH_MSG(a.size() == b.size(), "The size of vector to do cross product is not equal!");
	double res = 0;
	for (int i = 0; i < a.size(); i++)
		res += a[i] * b[i];
	return res;
}
float inner(std::vector<float>& a, std::vector<float>& b) {
	ASSERT_WITH_MSG(a.size() > 0, "The size of input vector should be larger than 0 while calculating inner product.");
	ASSERT_WITH_MSG(a.size() == b.size(), "The size of two input vector should be equal while calculating inner product.");

	std::vector<double> double_vec_a = float2double_vec(a);
	std::vector<double> double_vec_b = float2double_vec(b);
	return float(inner(double_vec_a, double_vec_b));
}


/*********************************************** geometry ******************************************************/
bool point_triangle_test_3d(std::vector<double>& pts, std::vector<double>& tri_a, std::vector<double>& tri_b, std::vector<double>& tri_c) {
	ASSERT_WITH_MSG(pts.size() == 3, "The input point should be 3d point");
	ASSERT_WITH_MSG(tri_a.size() == 3, "The input point should be 3d point");
	ASSERT_WITH_MSG(tri_b.size() == 3, "The input point should be 3d point");
	ASSERT_WITH_MSG(tri_c.size() == 3, "The input point should be 3d point");

	// transfer to another basis
	// math:
	//       t * b.X + u * c.X + v * n.X = P.X
	//       t * b.Y + u * c.Y + v * n.Y = P.Y
	//       t * b.Z + u * c.Z + v * n.Z = P.Z
	std::vector<double> vec_b, vec_c, pts_new;
	for (int i = 0; i < 3; i++) {
		vec_b.push_back(tri_b[i] - tri_a[i]);
		vec_c.push_back(tri_c[i] - tri_a[i]);
		pts_new.push_back(pts[i] - tri_a[i]);
	}

	std::vector<double> normal = cross(vec_b, vec_c);
	cv::Mat A(3, 3, CV_64F);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (j == 0)
				A.at<double>(i, j) = vec_b[i];
			else if (j == 1)
				A.at<double>(i, j) = vec_c[i];
			else
				A.at<double>(i, j) = normal[i];
		}
	}
	//print_mat(A);
	cv::Mat b(3, 1, CV_64F);
	b = cv::Mat(pts_new);
	//print_mat(b);
	cv::Mat x(3, 1, CV_64F);
	//x = (A.t() * A).inv() * A.t() * b;		// least square
	cv::solve(A, b, x);
	//print_mat(x);

	std::vector<double> vec_x;
	for (int i = 0; i < x.rows; i++) {
		vec_x.push_back(x.at<double>(i, 0));
	}

	// condition:
	//		0 <= t <= 1
	//		0 <= u <= 1
	//		t + u <= 1
	if (vec_x[0] <= 1 && vec_x[0] >= 0 && vec_x[1] <= 1 && vec_x[1] >= 0 && (vec_x[0] + vec_x[1] <= 1))
		return true;
	else
		return false;
}

void get_2d_line(pcl::PointXY& a, pcl::PointXY& b, cv::Mat& line) {
	std::vector<double> line_vec;

	cv::Point2d pts_2d_a = pcl2cv_pts2d(a);
	cv::Point2d pts_2d_b = pcl2cv_pts2d(b);
	get_2d_line(pts_2d_a, pts_2d_b, line_vec);
	ASSERT_WITH_MSG(line_vec.size() == 3, "The size of line vector should be three while getting the line.");
	ASSERT_WITH_MSG(line.type() == CV_32F || line.type() == CV_64F, "Only CV_32F or CV_64F are supported now while getting 2d line.");
	std::vector<double> line_norm = normalize_line_plane(line_vec);
	if (line.type() == CV_32F) {
		line.at<float>(0, 0) = float(line_norm[0]);
		line.at<float>(1, 0) = float(line_norm[1]);
		line.at<float>(2, 0) = float(line_norm[2]);
	}
	else if (line.type() == CV_64F) {
		line.at<double>(0, 0) = line_norm[0];
		line.at<double>(1, 0) = line_norm[1];
		line.at<double>(2, 0) = line_norm[2];
	}
	else
		ASSERT_WITH_MSG(1 == 0, "Fatal error while getting 2d line.");

}


void get_2d_line(cv::Point2d& a, cv::Point2d& b, std::vector<double>& line) {
	print_pts2d(a);
	print_pts2d(b);
	line.push_back(a.y - b.y);
	line.push_back(b.x - a.x);
	line.push_back(a.x * b.y - a.y * b.x);
}


std::vector<double> normalize_line_plane(std::vector<double>& src) {
	ASSERT_WITH_MSG(src.size() == 3 || src.size() == 4, "The size of input line should be vector of 3 or 4 while normalizing.");
	std::vector<double> dst;
	std::vector<double> direction_vec;
	for (int i = 0; i < src.size() - 1; i++) 
		direction_vec.push_back(src[i]);
	double length_direction = l2_norm(direction_vec);	// normalize the line by setting the direction vector as length 1

	bool inv_flag = false;
	for (int i = 0; i < src.size() - 1; i++) {
		if (inv_flag)	// already set the first variable as positive value
			break;
		if (src[i] > 0) {
			inv_flag = true;
			continue;
		}
		else if (src[i] < 0 && src[i] != 0) {
			length_direction = -length_direction;
			inv_flag = true;
			continue;
		}
		else {
			ASSERT_WITH_MSG(std::abs(src[i] - 0) < EPS_SMALL, "The variable should be zero while normalization.");
			continue;
		}	
	}
	for (int i = 0; i < src.size(); i++)
		dst.push_back(src[i] / length_direction);
	return dst;
}


double get_x_from_2d_line(std::vector<double>& line, double y) {
	ASSERT_WITH_MSG(line.size() == 3, "The size of input line should be vector of 3 while getting x and y coordinate.");
	ASSERT_WITH_MSG(!CHECK_SCALAR_EQ(line[0], 0), "The element a of the line is 0. The line is horizontal! No x could be got.");
	return ((-line[1]) * y - line[2]) / line[0];
}
double get_y_from_2d_line(std::vector<double>& line, double x) {
	ASSERT_WITH_MSG(line.size() == 3, "The size of input line should be vector of 3 while getting x and y coordinate.");
	ASSERT_WITH_MSG(!CHECK_SCALAR_EQ(line[1], 0), "The element b of the line is 0. The line is vertical! No y could be got.");
	return ((-line[0]) * x - line[2]) / line[1];
}


double get_3d_plane(cv::Point3d& a, cv::Point3d& b, cv::Point3d& c, std::vector<double>& p) {
	cv::Point3d da, dc;
	da.x = a.x - b.x; da.y = a.y - b.y; da.z = a.z - b.z;
	dc.x = c.x - b.x; dc.y = c.y - b.y; dc.z = c.z - b.z;
	p.resize(4);
	p[0] = da.y * dc.z - da.z * dc.y;
	p[1] = da.z * dc.x - da.x * dc.z;
	p[2] = da.x * dc.y - da.y * dc.x;			// cross product
	long double length = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
	p[0] /= length; //make it easy to compute point to plane distance
	p[1] /= length;
	p[2] /= length;
	p[3] = p[0] * a.x + p[1] * a.y + p[2] * a.z;
	p[3] = -p[3];
	long double residual;
//	residual = std::abs(a.x * p[0] + a.y * p[1] + a.z * p[2] + p[3]);
//	ASSERT_WITH_MSG(residual < EPS_MYSELF, "Point is not on the plane. Please check! The residual for a is " + std::to_string(residual) + ".\na: [" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + "]\nb: [" + std::to_string(b.x) + ", " + std::to_string(b.y) + ", " + std::to_string(b.z) + "]\nc: [" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + "]\np: [" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ", " + std::to_string(p[2]) + ", " + std::to_string(p[3]) + "]\nlength: " + std::to_string(length));
//	residual = std::abs(b.x * p[0] + b.y * p[1] + b.z * p[2] + p[3]);
//	ASSERT_WITH_MSG(residual < EPS_MYSELF, "Point is not on the plane. Please check! The residual for b is " + std::to_string(residual) + ".\na: [" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + "]\nb: [" + std::to_string(b.x) + ", " + std::to_string(b.y) + ", " + std::to_string(b.z) + "]\nc: [" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + "]\np: [" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ", " + std::to_string(p[2]) + ", " + std::to_string(p[3]) + "]\nlength: " + std::to_string(length));
//	residual = std::abs(c.x * p[0] + c.y * p[1] + c.z * p[2] + p[3]);
//	ASSERT_WITH_MSG(residual < EPS_MYSELF, "Point is not on the plane. Please check! The residual for c is " + std::to_string(residual) + ".\na: [" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + "]\nb: [" + std::to_string(b.x) + ", " + std::to_string(b.y) + ", " + std::to_string(b.z) + "]\nc: [" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + "]\np: [" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ", " + std::to_string(p[2]) + ", " + std::to_string(p[3]) + "]\nlength: " + std::to_string(length));
//	residual = std::abs(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - 1);
//	ASSERT_WITH_MSG(residual < EPS_MYSELF, "The parameter of the plane is not unit! The residual is " + std::to_string(residual) + ".\na: [" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.z) + "]\nb: [" + std::to_string(b.x) + ", " + std::to_string(b.y) + ", " + std::to_string(b.z) + "]\nc: [" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) + "]\np: [" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ", " + std::to_string(p[2]) + ", " + std::to_string(p[3]) + "]\nlength: " + std::to_string(length));
	return length;
}

double get_3d_plane(pcl::PointXYZ& a, pcl::PointXYZ& b, pcl::PointXYZ& c, std::vector<double>& p) {
	cv::Point3d pts_3d_a = pcl2cv_pts3d(a);
	cv::Point3d pts_3d_b = pcl2cv_pts3d(b);
	cv::Point3d pts_3d_c = pcl2cv_pts3d(c);

	return get_3d_plane(pts_3d_a, pts_3d_b, pts_3d_c, p);
}

// TODO: test for correctness
bool inside_polygon(cv::Mat &M, pcl::PointXYZ& p, std::vector<pcl::PointXY>& m, int s, int e) {
	float dir = 0;
	cv::Mat line(3, 1, CV_32F);
	cv::Mat point(4, 1, CV_32F);
	point.at<float>(3, 0) = 1;
	for (int i = s; i < e; i++) {
		int next = i + 1;
		if (i + 1 == e) {
			next = s;
		}
		get_2d_line(m[i], m[next], line);
		point.at<float>(0, 0) = p.x;
		point.at<float>(1, 0) = p.y;
		point.at<float>(2, 0) = p.z;
		cv::Mat MP = M * point;
		MP = MP / MP.at<float>(2, 0);
		cv::Mat mul = line.t() * MP / MP.at<float>(2, 0);
		float mult = mul.at<float>(0, 0);
		if (i == s) {
			if (mult == 0) {
				return false;
			}
			dir = mult;
		}
		else {
			if (dir * mult < 0) {
				return false;
			}
		}
	}
	return true;
}

void get_projected_pts_on_2d_line(cv::Point2d& pts_src, std::vector<double>& line, cv::Point2d& pts_dst) {
	ASSERT_WITH_MSG(line.size() == 3, "The size of input line should be 3 while calculating projected point.");
	
	// find the perpendicular line
	std::vector<double> line_orthogonal;
	line_orthogonal.push_back(-1 * line[1]);
	line_orthogonal.push_back(line[0]);
	line_orthogonal.push_back(-1 * (pts_src.x * line_orthogonal[0] + pts_src.y * line_orthogonal[1]));

	get_intersection_pts_from_2d_lines(line, line_orthogonal, pts_dst);
}

void get_intersection_pts_from_2d_lines(std::vector<double>& line1, std::vector<double>& line2, cv::Point2d& pts_dst) {
	ASSERT_WITH_MSG(line1.size() == 3, "The size of input line should be 3 while calculating intersection point.");
	ASSERT_WITH_MSG(line2.size() == 3, "The size of input line should be 3 while calculating intersection point.");
	pts_dst.x = (line2[1] * line1[2] - line1[1] * line2[2]) / (line2[0] * line1[1] - line1[0] * line2[1]);
	pts_dst.y = (line1[0] * line2[2] - line2[0] * line1[2]) / (line2[0] * line1[1] - line1[0] * line2[1]);
}

/*********************************************** algorithm ******************************************************/
// TODO: test for correctness
void mean_shift(std::vector<std::vector<double>>& pts_estimate, pcl::PointXYZ& out) {
	std::vector<double> now(3, 0);
	double threshold = 10; //mm
	for (int i = 0; i < 3; i++) {
		now[i] = get_median(pts_estimate[i]);
		//now[i] = pts_estimate[i][0];
	}
	fprintf(stderr, "b4: %g %g %g ", now[0], now[1], now[2]);
	int pts = pts_estimate[0].size();
	while (1) {
		std::vector<double> acc(3);
		int accpts = 0;
		for (int i = 0; i < pts; i++) {
			double dist = 0;
			for (int j = 0; j < 3; j++) {
				double diff = pts_estimate[j][i] - now[j];
				dist += diff * diff;
			}
			if (sqrt(dist) > threshold) {
				continue;
			}
			for (int j = 0; j < 3; j++) {
				acc[j] += pts_estimate[j][i];
			}
			accpts++;
		}
		if (accpts == 0) {
			break;
		}
		for (int j = 0; j < 3; j++) {
			acc[j] /= accpts;
		}

		int same = 1;
		for (int j = 0; j < 3; j++) {
			if (std::abs(now[j] - acc[j]) > EPS_MYSELF) {
				same = 0;
			}
			now[j] = acc[j];
		}
		if (same == 1) {
			if (threshold < 1) {
				break;
			}
			threshold /= 1.1;
		}
	}
	out.x = now[0];
	out.y = now[1];
	out.z = now[2];
	fprintf(stderr, ". meanshift: %g %g %g\n", now[0], now[1], now[2]);
}


