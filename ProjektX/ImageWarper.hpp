#ifndef IMAGEWARPER_HPP
#define IMAGEWARPER_HPP

/**
* @brief ImageWarper.hpp
* @author Lukas Roth
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <algorithm>

#define WARP_PERSPECTIVE 0
#define WARP_AFFINE 1
#define AA_OFF 0
#define AA_ON 1
#define CV_INTERPOLATION 2

class ImageWarper
{
private:
	float theta; // Latitude or tilt
	float phi; // Longitude
	float scale;
	int method;
	int antialiasing;
	float sigma;
	cv::Size size;
	cv::Size warpedSize;
	cv::Point2f warpedTopLeftCorner, warpedBottomRightCorner;
	cv::Mat H;
	cv::Mat H_inv;
	
	//static bool comp_point_x(cv::Point2f i, cv::Point2f j);
	//static bool comp_point_y(cv::Point2f i, cv::Point2f j);

	cv::Point2f warpPoint(cv::Mat T, cv::Point2f pt);

	void warpRect(cv::Rect rect, std::vector<cv::Point2f>& warpedCorners);
	void calculateOutputLimits(const std::vector<cv::Point2f>& warpedCorners, cv::Size& warpedSize, cv::Point2f& upperLeftCorner, cv::Point2f& lowerRightCorner);

	cv::Mat calculatePerspectiveHomography(float alpha, float beta, float gamma, float dx, float dy, float dz, float scale, int width, int height);
	cv::Mat calculateAffineHomography(float theta, float phi, float scale);

	void doEllipticalWeightedAverageFiltering(const cv::Mat& src, cv::Mat& dst);
public:
	ImageWarper(const cv::Mat& image, float theta, float phi, float scale, int method = WARP_PERSPECTIVE, int antialiasing = AA_ON, float sigma = 1.0f);

	//int getMethod();
	//void setMethod(int method);

	void warpImage(const cv::Mat& src, cv::Mat& dst);

	cv::Point2f warpPointForward(cv::Point2f pt);
	cv::Point2f warpPointInverse(cv::Point2f pt);

	cv::Mat warpCovarianceMatrixForward(cv::Point2f pt, cv::Mat C);
	cv::Mat warpCovarianceMatrixInverse(cv::Point2f pt, cv::Mat C);
};

#endif // IMAGEWARPER_HPP