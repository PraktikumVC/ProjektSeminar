/**
* @brief ImageWarper.cpp
* @author Lukas Roth
*/

#include "ImageWarper.hpp"

// Constructor
ImageWarper::ImageWarper(const cv::Mat& src, float theta, float phi, float scale, int method, int antialiasing, float sigma)
	: theta(theta), phi(phi), scale(scale), method(method), antialiasing(antialiasing), sigma(sigma)
{
	size = cv::Size(src.cols, src.rows);
	if(method == WARP_PERSPECTIVE)
	{
		float f = sqrt(static_cast<float>(src.cols * src.cols + src.rows * src.rows));
		H = calculatePerspectiveHomography(- theta, 0, phi, 0, 0, f, scale * f, src.cols, src.rows);
		//H_inv = H.inv();

		std::vector<cv::Point2f> warpedCorners;
		warpRect(cv::Rect(0, 0, src.cols - 1, src.rows - 1), warpedCorners);

		calculateOutputLimits(warpedCorners, warpedSize, warpedTopLeftCorner, warpedBottomRightCorner);

		cv::Mat T = (cv::Mat_<double>(3, 3) <<
			1, 0, - warpedTopLeftCorner.x,
			0, 1, - warpedTopLeftCorner.y,
			0, 0, 1);
		H = T * H;

		//cv::Mat T_inv = (cv::Mat_<double>(3, 3) <<
		//	1, 0, warpedTopLeftCorner.x,
		//	0, 1, warpedTopLeftCorner.y,
		//	0, 0, 1);
		//H_inv = H_inv * T_inv;

		H_inv = H.inv();
	}
	if(method == WARP_AFFINE)
	{
		H = calculateAffineHomography(theta, phi, scale);
		//cv::invertAffineTransform(H, H_inv);

		std::vector<cv::Point2f> warpedCorners;
		warpRect(cv::Rect(0, 0, src.cols - 1, src.rows - 1), warpedCorners);

		calculateOutputLimits(warpedCorners, warpedSize, warpedTopLeftCorner, warpedBottomRightCorner);

		H.at<double>(0, 2) = H.at<double>(0, 2) - warpedTopLeftCorner.x;
		H.at<double>(1, 2) = H.at<double>(1, 2) - warpedTopLeftCorner.y;

		//H_inv.at<double>(0, 2) = H_inv.at<double>(0, 2) + warpedTopLeftCorner.x;
		//H_inv.at<double>(1, 2) = H_inv.at<double>(1, 2) + warpedTopLeftCorner.y;

		cv::invertAffineTransform(H, H_inv);
	}
}

//bool ImageWarper::comp_point_x(cv::Point2f i, cv::Point2f j)
//{
//	return (i.x < j.x);
//}

//bool ImageWarper::comp_point_y(cv::Point2f i, cv::Point2f j)
//{
//	return (i.y < j.y);
//}

cv::Point2f ImageWarper::warpPoint(cv::Mat T, cv::Point2f pt)
{
	cv::Point2f wpt;
	// WARP_PERSPECTIVE
	//if(T.size() == cv::Size(3, 3))
	//{
	if(method == WARP_PERSPECTIVE)
	{
		double w = T.at<double>(2, 0) * pt.x + T.at<double>(2, 1) * pt.y + T.at<double>(2, 2);
		if(fabs(w) > FLT_EPSILON)
		{
			w = 1.0 / w;
			wpt.x = static_cast<float>((T.at<double>(0, 0) * pt.x + T.at<double>(0, 1) * pt.y + T.at<double>(0, 2)) * w);
			wpt.y = static_cast<float>((T.at<double>(1, 0) * pt.x + T.at<double>(1, 1) * pt.y + T.at<double>(1, 2)) * w);
		}
		else
		{
			wpt.x = 0.0f;
			wpt.y = 0.0f;
		}
	}
	// WARP_AFFINE
	//if(T.size() == cv::Size(3, 2))
	//{
	if(method == WARP_AFFINE)
	{
		wpt.x = static_cast<float>(T.at<double>(0, 0) * pt.x + T.at<double>(0, 1) * pt.y + T.at<double>(0, 2));
		wpt.y = static_cast<float>(T.at<double>(1, 0) * pt.x + T.at<double>(1, 1) * pt.y + T.at<double>(1, 2));
	}
	return wpt;
}

void ImageWarper::warpRect(cv::Rect rect, std::vector<cv::Point2f>& wpts)
{
	wpts.clear();
	wpts.resize(4);
	// Order: top-left, top-right, bottom-right, bottom-left
	wpts[0] = warpPoint(H, cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)));
	wpts[1] = warpPoint(H, cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)));
	wpts[2] = warpPoint(H, cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)));
	wpts[3] = warpPoint(H, cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)));
	//std::vector<cv::Point2f> pts(4, cv::Point2f());
	//// Order: top-left, top-right, bottom-right, bottom-left
	//pts[0] = cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y));
	//pts[1] = cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y));
	//pts[2] = cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height));
	//pts[3] = cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height));
	//cv::perspectiveTransform(pts, wpts, H);
}

void ImageWarper::calculateOutputLimits(const std::vector<cv::Point2f>& wpts, cv::Size& wsz, cv::Point2f& wtlc, cv::Point2f& wbrc)
{
	float x = wpts[0].x;
	float y = wpts[0].y;
	float left = x;
	float right = x;
	float top = y;
	float bottom = y;
	for(int i = 1; i < wpts.size(); ++i)
	{
		x = wpts[i].x;
		y = wpts[i].y;
		if(left > x)
		{
			left = x;
		}
		else
		{
			if(right < x)
			{
				right = x;
			}
		}
		if(top > y)
		{
			top = y;
		}
		else
		{
			if(bottom < y)
			{
				bottom = y;
			}
		}
	}
	//float left = (*std::min_element(wpts.begin(), wpts.end(), comp_point_x)).x;
	//float right = (*std::max_element(wpts.begin(), wpts.end(), comp_point_x)).x;
	//float top = (*std::min_element(wpts.begin(), wpts.end(), comp_point_y)).y;
	//float bottom = (*std::max_element(wpts.begin(), wpts.end(), comp_point_y)).y;
	int width = static_cast<int>(ceil(right) - floor(left)) + 1;
	int height = static_cast<int>(ceil(bottom) - floor(top)) + 1;
	wsz = cv::Size(width, height);
	wtlc = cv::Point2f(left, top);
	wbrc = cv::Point2f(right, bottom);
}

cv::Mat ImageWarper::calculatePerspectiveHomography(float alpha, float beta, float gamma, float dx, float dy, float dz, float scale, int width, int height)
{
	// Inspiration: http://jepsonsblog.blogspot.de/2012/11/rotation-in-3d-using-opencvs.html

	// Projection 2D -> 3D matrix
	cv::Mat P1 = (cv::Mat_<double>(4, 3) <<
		1, 0, - width / 2,
		0, 1, - height / 2,
		0, 0, 0,
		0, 0, 1);

	// Rotation matrices around the X, Y, and Z axis
	cv::Mat RX = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, 0,
		0, cos(alpha), - sin(alpha), 0,
		0, sin(alpha), cos(alpha), 0,
		0, 0, 0, 1);
	cv::Mat RY = (cv::Mat_<double>(4, 4) <<
		cos(beta), 0, - sin(beta), 0,
		0, 1, 0, 0,
		sin(beta), 0, cos(beta), 0,
		0, 0, 0, 1);
	cv::Mat RZ = (cv::Mat_<double>(4, 4) <<
		cos(gamma), - sin(gamma), 0, 0,
		sin(gamma), cos(gamma), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1);

	// Composed rotation matrix
	cv::Mat R = RX * RY * RZ;

	// Translation matrix
	cv::Mat T = (cv::Mat_<double>(4, 4) <<
		1, 0, 0, dx,
		0, 1, 0, dy,
		0, 0, 1, dz,
		0, 0, 0, 1);

	// Projection 3D -> 2D matrix
	cv::Mat P2 = (cv::Mat_<double>(3, 4) <<
		scale, 0, width / 2, 0,
		0, scale, height / 2, 0,
		0, 0, 1, 0);

	// Final transformation matrix
	cv::Mat M = P2 * (T * (R * P1));
	M = M / M.at<double>(2, 2);
	return M;
}

cv::Mat ImageWarper::calculateAffineHomography(float theta, float phi, float scale)
{
	double t = 1.0 / cos(theta);
	
	// Rotation matrix
	cv::Mat R = (cv::Mat_<double>(3, 3) <<
		cos(phi), - sin(phi), 0,
		sin(phi), cos(phi), 0,
		0, 0, 1);
	
	// Tilt and scale matrix
	cv::Mat TS = (cv::Mat_<double>(3, 3) <<
		scale, 0, 0,
		0, scale / t, 0,
		0, 0, 1);

	// Final transformation matrix
	cv::Mat M = TS * R;
	M.pop_back();
	return M;
}

void ImageWarper::doEllipticalWeightedAverageFiltering(const cv::Mat& src, cv::Mat& dst)
{
	// src has to be of type CV_32F or CV_32FC3

	dst = cv::Mat(warpedSize, src.type());

	// Calculate coordinates in src and dst
	cv::Mat dst_x(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat dst_y(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat src_x(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat src_y(warpedSize.height, warpedSize.width, CV_32F);

	cv::Mat x(1, warpedSize.width, CV_32F);
	float* ptr_x = x.ptr<float>();
	for(int i = 0; i < warpedSize.width; ++i)
	{
		*ptr_x++ = static_cast<float>(i);
	}
	cv::repeat(x, warpedSize.height, 1, dst_x);

	cv::Mat y(warpedSize.height, 1, CV_32F);
	float* ptr_y = y.ptr<float>();
	for(int i = 0; i < warpedSize.height; ++i)
	{
		*ptr_y++ = static_cast<float>(i);
	}
	cv::repeat(y, 1, warpedSize.width, dst_y);

	float* ptr_dst_x = dst_x.ptr<float>();
	float* ptr_dst_y = dst_y.ptr<float>();
	float* ptr_src_x = src_x.ptr<float>();
	float* ptr_src_y = src_y.ptr<float>();
	for(int i = 0; i < warpedSize.width * warpedSize.height; ++i)
	{
		cv::Point2f pt, wpt;
		pt.x = *ptr_dst_x++;
		pt.y = *ptr_dst_y++;
		wpt = warpPointInverse(pt);
		*ptr_src_x++ = wpt.x;
		*ptr_src_y++ = wpt.y;
	}

	// Calculate derivatives (du_dx, du_dy, dv_dx, dv_dy represent the rate of change of the texture coordinates relative to changes in screen space)
	cv::Mat du_dx(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat du_dy(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat dv_dx(warpedSize.height, warpedSize.width, CV_32F);
	cv::Mat dv_dy(warpedSize.height, warpedSize.width, CV_32F);

	cv::Mat kernel_x = (cv::Mat_<float>(1, 3) <<
		- 0.5, 0, 0.5);
	cv::Mat kernel_y = (cv::Mat_<float>(3, 1) <<
		- 0.5, 0, 0.5);

	cv::filter2D(src_x, du_dx, - 1, kernel_x);
	cv::filter2D(src_x, du_dy, - 1, kernel_y);
	cv::filter2D(src_y, dv_dx, - 1, kernel_x);
	cv::filter2D(src_y, dv_dy, - 1, kernel_y);

	// Compute the ellipse's coefficients (A * x * x + B * x * y + C * y * y = F)
	cv::Mat A = dv_dx.mul(dv_dx) + dv_dy.mul(dv_dy); // + 1.0f would be wrong (because of the definition of the coordinate system)
	cv::Mat B = - 2.0f * (du_dx.mul(dv_dx) + du_dy.mul(dv_dy));
	cv::Mat C = du_dx.mul(du_dx) + du_dy.mul(du_dy); // + 1.0f would be wrong (because of the definition of the coordinate system)
	cv::Mat F = A.mul(C) - 0.25f * B.mul(B);

	// Compute the ellipse's bounding box
	cv::Mat temp0 = - B.mul(B) + 4.0f * C.mul(A);
	cv::Mat temp1, temp2;
	cv::sqrt(temp0.mul(C).mul(F), temp1);
	cv::sqrt(temp0.mul(A).mul(F), temp2);

	cv::Mat u_min = src_x - 2.0f * (temp1 / temp0);
	cv::Mat u_max = src_x + 2.0f * (temp1 / temp0);
	cv::Mat v_min = src_y - 2.0f * (temp2 / temp0);
	cv::Mat v_max = src_y + 2.0f * (temp2 / temp0);

	float* ptr_u_min = u_min.ptr<float>();
	float* ptr_u_max = u_max.ptr<float>();
	float* ptr_v_min = v_min.ptr<float>();
	float* ptr_v_max = v_max.ptr<float>();
	for(int i = 0; i < warpedSize.width * warpedSize.height; ++i)
	{
		*ptr_u_min++ = floor(*ptr_u_min);
		*ptr_u_max++ = ceil(*ptr_u_max);
		*ptr_v_min++ = floor(*ptr_v_min);
		*ptr_v_max++ = ceil(*ptr_v_max);
	}

	// Iterate over the ellipse's bounding box and calculate A * x * x + B * x * y + C * y * y (when this value is less than F, we're inside of the ellipse)
	float* ptr_dst = dst.ptr<float>();

	ptr_src_x = src_x.ptr<float>();
	ptr_src_y = src_y.ptr<float>();

	float* ptr_A = A.ptr<float>();
	float* ptr_B = B.ptr<float>();
	float* ptr_C = C.ptr<float>();
	float* ptr_F = F.ptr<float>();

	ptr_u_min = u_min.ptr<float>();
	ptr_u_max = u_max.ptr<float>();
	ptr_v_min = v_min.ptr<float>();
	ptr_v_max = v_max.ptr<float>();

	float sigma_square_inv = 1.0f / (sigma * sigma);
	float pi_inv = static_cast<float>(1.0 / CV_PI);

	float U, ddq, num, den, V, dq, q, d, weight, num0, num1, num2;

	for(int i = 0; i < warpedSize.width * warpedSize.height; ++i)
	{
		U = *ptr_u_min - *ptr_src_x;
		ddq = 2.0f * *ptr_A;
		num = 0;
		den = 0;
		num0 = 0;
		num1 = 0;
		num2 = 0;

		for(int v = static_cast<int>(*ptr_v_min); v <= static_cast<int>(*ptr_v_max); ++v)
		{
			V = v - *ptr_src_y;
			dq = *ptr_A * (2.0f * U + 1.0f) + *ptr_B * V;
			q = (*ptr_C * V + *ptr_B * U) * V + *ptr_A * U * U;

			for(int u = static_cast<int>(*ptr_u_min); u <= static_cast<int>(*ptr_u_max); ++u)
			{
				if(q < *ptr_F)
				{
					d = q / *ptr_F;

					weight = 0.5f * pi_inv * sigma_square_inv * exp(- 0.5f * d * sigma_square_inv);

					if(u >= 0 && u < src.cols && v >= 0 && v < src.rows)
					{
						if(src.type() == CV_32F)
						{
							num += weight * src.ptr<float>(v)[u];
						}
						if(src.type() == CV_32FC3)
						{
							num0 += weight * src.ptr<cv::Vec3f>(v)[u][0];
							num1 += weight * src.ptr<cv::Vec3f>(v)[u][1];
							num2 += weight * src.ptr<cv::Vec3f>(v)[u][2];
						}
					}

					den += weight;
				}

				q += dq;
				dq += ddq;
			}
		}

		if(src.type() == CV_32F)
		{
			*ptr_dst++ = num / den;
		}
		if(src.type() == CV_32FC3)
		{
			*ptr_dst++ = num0 / den;
			*ptr_dst++ = num1 / den;
			*ptr_dst++ = num2 / den;
		}

		++ptr_src_x;
		++ptr_src_y;
		++ptr_A;
		++ptr_B;
		++ptr_C;
		++ptr_F;
		++ptr_u_min;
		++ptr_u_max;
		++ptr_v_min;
		++ptr_v_max;
	}
}

//int ImageWarper::getMethod()
//{
//	return method;
//}

//void ImageWarper::setMethod(int method)
//{
//	this->method = method;
//}

void ImageWarper::warpImage(const cv::Mat& src, cv::Mat& dst)
{
	// Check if src.size() equals size (member variable)

	if(method == WARP_PERSPECTIVE)
	{
		//float f = sqrt(static_cast<float>(src.cols * src.cols + src.rows * src.rows));
		//H = calculatePerspectiveHomography(- theta, 0, phi, 0, 0, f, scale * f, src.cols, src.rows);

		//std::vector<cv::Point2f> warpedCorners = warpRect(H, cv::Rect(0, 0, src.cols - 1, src.rows - 1));

		//cv::Size warpedSize;
		//cv::Point2f warpedTopLeftCorner, warpedBottomRightCorner;
		//calculateOutputLimits(warpedCorners, warpedSize, warpedTopLeftCorner, warpedBottomRightCorner);

		//cv::Mat T = (cv::Mat_<float>(3, 3) <<
		//	1, 0, - warpedTopLeftCorner.x,
		//	0, 1, - warpedTopLeftCorner.y,
		//	0, 0, 1);
		//H = T * H;

		if(antialiasing == AA_ON)
		{
			doEllipticalWeightedAverageFiltering(src, dst);
		}
		else
		{
			cv::warpPerspective(src, dst, H, warpedSize, CV_INTERPOLATION);
		}
	}
	if(method == WARP_AFFINE)
	{
		//H = calculateAffineHomography(theta, phi, scale);

		//std::vector<cv::Point2f> warpedCorners = warpRect(H, cv::Rect(0, 0, src.cols - 1, src.rows - 1));

		//cv::Size warpedSize;
		//cv::Point2f warpedTopLeftCorner, warpedBottomRightCorner;
		//calculateOutputLimits(warpedCorners, warpedSize, warpedTopLeftCorner, warpedBottomRightCorner);

		//H.at<float>(0, 2) = H.at<float>(0, 2) - warpedTopLeftCorner.x;
		//H.at<float>(1, 2) = H.at<float>(1, 2) - warpedTopLeftCorner.y;

		if(antialiasing == AA_ON)
		{
			doEllipticalWeightedAverageFiltering(src, dst);
		}
		else
		{
			cv::warpAffine(src, dst, H, warpedSize, CV_INTERPOLATION);
		}
	}
}

cv::Point2f ImageWarper::warpPointForward(cv::Point2f pt)
{
	cv::Point2f wpt;
	if(method == WARP_PERSPECTIVE)
	{
		//cv::Mat S = (cv::Mat_<double>(3, 3) <<
		//	scale, 0, 0,
		//	0, scale, 0,
		//	0, 0, 1);
		//wpt = warpPoint(H * S, pt);
		//std::vector<cv::Point2f> ptv, wptv;
		//ptv.push_back(pt);
		//cv::perspectiveTransform(ptv, wptv, H * S);
		//wpt = wptv[0];
	}
	if(method == WARP_AFFINE)
	{
		//pt.x = pt.x * scale;
		//pt.y = pt.y * scale;
		//wpt = warpPoint(H, pt);
		//std::vector<cv::Point2f> ptv, wptv;
		//ptv.push_back(pt);
		//ptv[0].x = ptv[0].x * scale;
		//ptv[0].y = ptv[0].y * scale;
		//cv::perspectiveTransform(ptv, wptv, H);
		//wpt = wptv[0];
	}
	//pt.x = pt.x * scale;
	//pt.y = pt.y * scale;
	wpt = warpPoint(H, pt);
	return wpt;
}

cv::Point2f ImageWarper::warpPointInverse(cv::Point2f pt)
{
	cv::Point2f wpt;
	if(method == WARP_PERSPECTIVE)
	{
		//cv::Mat S_inv = (cv::Mat_<double>(3, 3) <<
		//	1 / scale, 0, 0,
		//	0, 1 / scale, 0,
		//	0, 0, 1);
		//wpt = warpPoint(S_inv * H_inv, pt);
		//std::vector<cv::Point2f> ptv, wptv;
		//ptv.push_back(pt);
		//cv::perspectiveTransform(ptv, wptv, S_inv * H_inv);
		//wpt = wptv[0];
	}
	if(method == WARP_AFFINE)
	{
		//wpt = warpPoint(H_inv, pt);
		//wpt.x = wpt.x / scale;
		//wpt.y = wpt.y / scale;
		//std::vector<cv::Point2f> ptv, wptv;
		//ptv.push_back(pt);
		//cv::perspectiveTransform(ptv, wptv, H);
		//wptv[0].x = wptv[0].x / scale;
		//wptv[0].y = wptv[0].y / scale;
		//wpt = wptv[0];
	}
	//pt.x = pt.x + warpedTopLeftCorner.x;
	//pt.y = pt.y + warpedTopLeftCorner.y;
	wpt = warpPoint(H_inv, pt);
	//wpt.x = wpt.x / scale;
	//wpt.y = wpt.y / scale;
	return wpt;
}

cv::Mat ImageWarper::warpCovarianceMatrixForward(cv::Point2f pt, cv::Mat C)
{
	cv::Mat wC;
	cv::Mat A;
	//double s = scale * scale;
	double s = 1.0;

	if(method == WARP_PERSPECTIVE)
	{
		//cv::Mat S = (cv::Mat_<double>(3, 3) <<
		//	scale, 0, 0,
		//	0, scale, 0,
		//	0, 0, 1);
		//cv::Mat T = H * S;

		cv::Mat T = H;

		//pt.x = pt.x * scale;
		//pt.y = pt.y * scale;

		double w = 1.0 / pow((T.at<double>(2, 0) * pt.x + T.at<double>(2, 1) * pt.y + T.at<double>(2, 2)), 2);

		// Jacobian matrix
		A = (cv::Mat_<float>(2, 2) <<
			((T.at<double>(0, 0) * T.at<double>(2, 1) - T.at<double>(0, 1) * T.at<double>(2, 0)) * pt.y - T.at<double>(0, 2) * T.at<double>(2, 0) + T.at<double>(0, 0) * T.at<double>(2, 2)) * w, ((T.at<double>(0, 1) * T.at<double>(2, 0) - T.at<double>(0, 0) * T.at<double>(2, 1)) * pt.x - T.at<double>(0, 2) * T.at<double>(2, 1) + T.at<double>(0, 1) * T.at<double>(2, 2)) * w,
			((T.at<double>(1, 0) * T.at<double>(2, 1) - T.at<double>(1, 1) * T.at<double>(2, 0)) * pt.y - T.at<double>(1, 2) * T.at<double>(2, 0) + T.at<double>(1, 0) * T.at<double>(2, 2)) * w, ((T.at<double>(1, 1) * T.at<double>(2, 0) - T.at<double>(1, 0) * T.at<double>(2, 1)) * pt.x - T.at<double>(1, 2) * T.at<double>(2, 1) + T.at<double>(1, 1) * T.at<double>(2, 2)) * w);
	}
	if(method == WARP_AFFINE)
	{
		// Jacobian matrix
		A = (cv::Mat_<float>(2, 2) <<
			H.at<double>(0, 0), H.at<double>(0, 1),
			H.at<double>(1, 0), H.at<double>(1, 1));
	}

	// Propagation of uncertainty
	wC = A * C * A.t();
	wC = wC * s;

	return wC;
}

cv::Mat ImageWarper::warpCovarianceMatrixInverse(cv::Point2f pt, cv::Mat C)
{
	cv::Mat wC;
	cv::Mat A;
	//double s = 1.0 / (scale * scale);
	double s = 1.0;

	if(method == WARP_PERSPECTIVE)
	{
		//cv::Mat S_inv = (cv::Mat_<double>(3, 3) <<
		//	1 / scale, 0, 0,
		//	0, 1 / scale, 0,
		//	0, 0, 1);
		//cv::Mat T = S_inv * H_inv;

		cv::Mat T = H_inv;

		//pt.x = pt.x + warpedTopLeftCorner.x;
		//pt.y = pt.y + warpedTopLeftCorner.y;

		double w = 1.0 / pow((T.at<double>(2, 0) * pt.x + T.at<double>(2, 1) * pt.y + T.at<double>(2, 2)), 2);

		// Jacobian matrix
		A = (cv::Mat_<float>(2, 2) <<
			((T.at<double>(0, 0) * T.at<double>(2, 1) - T.at<double>(0, 1) * T.at<double>(2, 0)) * pt.y - T.at<double>(0, 2) * T.at<double>(2, 0) + T.at<double>(0, 0) * T.at<double>(2, 2)) * w, ((T.at<double>(0, 1) * T.at<double>(2, 0) - T.at<double>(0, 0) * T.at<double>(2, 1)) * pt.x - T.at<double>(0, 2) * T.at<double>(2, 1) + T.at<double>(0, 1) * T.at<double>(2, 2)) * w,
			((T.at<double>(1, 0) * T.at<double>(2, 1) - T.at<double>(1, 1) * T.at<double>(2, 0)) * pt.y - T.at<double>(1, 2) * T.at<double>(2, 0) + T.at<double>(1, 0) * T.at<double>(2, 2)) * w, ((T.at<double>(1, 1) * T.at<double>(2, 0) - T.at<double>(1, 0) * T.at<double>(2, 1)) * pt.x - T.at<double>(1, 2) * T.at<double>(2, 1) + T.at<double>(1, 1) * T.at<double>(2, 2)) * w);
	}
	if(method == WARP_AFFINE)
	{
		// Jacobian matrix
		A = (cv::Mat_<float>(2, 2) <<
			H_inv.at<double>(0, 0), H_inv.at<double>(0, 1),
			H_inv.at<double>(1, 0), H_inv.at<double>(1, 1));
	}

	// Propagation of uncertainty
	wC = A * C * A.t();
	wC = wC * s;

	return wC;
}