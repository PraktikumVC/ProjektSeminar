#include <fstream>
#include <stdexcept>
#include <string>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "globals.h"
#include "Speicher.h"
/**
* @brief tmlm.cpp
* @author Lukas Roth
*/

#include "ImageWarper.hpp"

#include "hesaff/pyramid.h"
#include "hesaff/affine.h"
#include "hesaff/siftdesc.h"
#include "mser/mser.h"
#include "mser/sift.h"
#include "mser/affine.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/case_conv.hpp>
//#include <libconfig.h++>

#include <iostream>

struct MODSParams
{
	float sigma; // Smoothing
	std::vector<float> tilts;
	std::vector<float> scales;
	float angle; // In degree

	MODSParams()
		: sigma(1.0f), tilts(std::vector<float>(0, 0.0f)), scales(std::vector<float>(0, 0.0f)), angle(0.0f)
	{
	}
	MODSParams(float sigma, std::vector<float> tilts, std::vector<float> scales, float angle)
		: sigma(sigma), tilts(tilts), scales(scales), angle(angle)
	{
	}
};

struct WarpConfiguration
{
	std::vector<float> theta; // In radian
	std::vector<float> phi; // In radian
	std::vector<float> scale;

	WarpConfiguration()
		: theta(std::vector<float>(0, 0.0f)), phi(std::vector<float>(0, 0.0f)), scale(std::vector<float>(0, 0.0f))
	{
	}
	WarpConfiguration(std::vector<float> theta, std::vector<float> phi, std::vector<float> scale)
		: theta(theta), phi(phi), scale(scale)
	{
	}
};
/*
struct MSERParams
{
	int delta;
	int minArea; // Absolute number (number of pixels)
	float maxArea; // Relative number (percentage of image area)
	float maxVariation;
	float minDiversity;

	MSERParams()
		: delta(2), minArea(3), maxArea(0.01f), maxVariation(0.25f), minDiversity(0.2f)
	{
	}
	MSERParams(int delta, int minArea, float maxArea, float maxVariation, float minDiversity)
		: delta(delta), minArea(minArea), maxArea(maxArea), maxVariation(maxVariation), minDiversity(minDiversity)
	{
	}
};
*/

struct HessianAffineParams
{
	int firstOctave;
	int octaveResolution;
	float peakThreshold;
	float edgeThreshold;
	float laplacianPeakThreshold;

	HessianAffineParams()
		: firstOctave(0), octaveResolution(3), peakThreshold(3.0f), edgeThreshold(10.0f), laplacianPeakThreshold(0.01f)
	{
	}
	HessianAffineParams(int firstOctave, int octaveResolution, float peakThreshold, float edgeThreshold, float laplacianPeakThreshold)
		: firstOctave(firstOctave), octaveResolution(octaveResolution), peakThreshold(peakThreshold), edgeThreshold(edgeThreshold), laplacianPeakThreshold(laplacianPeakThreshold)
	{
	}
};

struct SIFTParams
{
	int method;
	int firstOctave;
	int octaveResolution;
	int patchResolution;
	float patchRelativeExtent;
	float patchRelativeSmoothing;

	SIFTParams()
		: method(0), firstOctave(0), octaveResolution(3), patchResolution(15), patchRelativeExtent(7.5f), patchRelativeSmoothing(1.0f)
	{
	}
	SIFTParams(int method, int firstOctave, int octaveResolution, int patchResolution, float patchRelativeExtent, float patchRelativeSmoothing)
		: method(method), firstOctave(firstOctave), octaveResolution(octaveResolution), patchResolution(patchResolution), patchRelativeExtent(patchRelativeExtent), patchRelativeSmoothing(patchRelativeSmoothing)
	{
	}
};

struct Descriptor
{
	float x; // Center (x, y, s)
	float y; // Center (x, y, s)
	float s; // Center (x, y, s)
	float a; // Covariance matrix (of an unoriented ellipse [a b; b c] or an oriented ellipse [a, b; 0, c])
	float b; // Covariance matrix (of an unoriented ellipse [a b; b c] or an oriented ellipse [a, b; 0, c])
	float c; // Covariance matrix (of an unoriented ellipse [a b; b c] or an oriented ellipse [a, b; 0, c])
	std::vector<float> data; // Descriptor data

	Descriptor()
		: x(0.0f), y(0.0f), s(0.0f), a(0.0f), b(0.0f), c(0.0f), data(std::vector<float>(0, 0.0f))
	{
	}
	Descriptor(float x, float y, float s, float a, float b, float c, std::vector<float> data)
		: x(x), y(y), s(s), a(a), b(b), c(c), data(data)
	{
	}
};

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
	const cv::Mat image;
	SIFTDescriptor sift;
	std::vector<Descriptor> descriptors;
	float mrSize;

public:
	AffineHessianDetector(const cv::Mat& image, const PyramidParams& pp, const AffineShapeParams& asp, const SIFTDescriptorParams& sp)
		: HessianDetector(pp), AffineShape(asp), image(image), sift(sp), mrSize(asp.mrSize)
	{
		this->setHessianKeypointCallback(this);
		this->setAffineShapeCallback(this);
	}

	void onHessianKeypointDetected(const cv::Mat& blur, float x, float y, float s, float pixelDistance, int type, float response)
	{
		findAffineShape(blur, x, y, s, pixelDistance, type, response);
	}

	void onAffineShapeFound(const cv::Mat& blur, float x, float y, float s, float pixelDistance, float a11, float a12, float a21, float a22, int type, float response, int iters)
	{
		// Convert shape into an up-is-up frame
		rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

		// Sample the patch
		if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
		{
			// Compute SIFT descriptor
			sift.computeSiftDescriptor(this->patch);

			float sc = this->mrSize * s;
			cv::Mat A = (cv::Mat_<float>(2, 2) << a11, a12, a21, a22); // A = [a, b; 0, c] (oriented ellipse)
			A = A * A.t(); // A = [a, b; b, c] (unoriented ellipse)
			A = sc * A;

			// Store the keypoint
			descriptors.push_back(Descriptor());
			Descriptor& k = descriptors.back();
			k.x = x;
			k.y = y;
			k.s = s;
			k.a = A.at<float>(0, 0);
			k.b = A.at<float>(0, 1);
			k.c = A.at<float>(1, 1);
			k.data = sift.vec;
		}
	}
};

int loadImage(const std::string& file, cv::Mat& image_gray, cv::Mat& image_color)
{
	cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_UNCHANGED);

	if (image.channels() == 3)
	{
		image_color = image;

		image.convertTo(image, CV_32FC3);
		std::vector<cv::Mat> image_split(3, cv::Mat());
		cv::split(image, image_split);
		image_gray = (image_split[0] + image_split[1] + image_split[2]) / 3.0;
	}
	else
	{
		image_gray = image;
		image_gray.convertTo(image_gray, CV_32F);

		cv::cvtColor(image, image_color, CV_GRAY2BGR);
	}

	return EXIT_SUCCESS;
}

void createWarpConfiguration(const MODSParams& params, WarpConfiguration& warpConfiguration)
{
	warpConfiguration.theta.clear();
	warpConfiguration.phi.clear();
	warpConfiguration.scale.clear();

	int k = 0;
	float phi = 0.0f;

	float theta = 0.0f;
	float rotationStep = 0.0f;

	for (int i = 0; i < params.scales.size(); ++i)
	{
		for (int j = 0; j < params.tilts.size(); ++j)
		{
			k = 0;
			phi = 0.0f;

			theta = acos(1 / params.tilts[j]);
			rotationStep = params.angle / params.tilts[j] / 180 * static_cast<float>(CV_PI); // From degree to radian

			while (phi < 2 * static_cast<float>(CV_PI))
			{
				warpConfiguration.theta.push_back(theta);
				warpConfiguration.phi.push_back(phi);
				warpConfiguration.scale.push_back(params.scales[i]);

				++k;
				phi = k * rotationStep;
			}
		}
	}
}
/*
void detectMSER(const cv::Mat& image, std::vector<Descriptor>& descriptors, const MSERParams& params1, const SIFTParams& params2)
{
	// Convert image from CV_32F to CV_8U
	cv::Mat tempImage(image.size(), CV_8U);
	image.convertTo(tempImage, CV_8U);

	int imageArea = tempImage.cols * tempImage.rows;
	int maxArea = static_cast<int>(params1.maxArea * imageArea);

	// Return if minArea > maxArea
	if (params1.minArea > maxArea)
	{
		return;
	}

	// Create object
	MSER mser(params1.delta, params1.minArea, maxArea, static_cast<double>(params1.maxVariation), static_cast<double>(params1.minDiversity), false);

	std::vector<MSER::Region> regions = mser((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows);

	// Invert image
	cv::bitwise_not(tempImage, tempImage);

	std::vector<MSER::Region> tempRegions = mser((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows);
	regions.insert(regions.end(), tempRegions.begin(), tempRegions.end());

	// Invert image
	cv::bitwise_not(tempImage, tempImage);

	// Create object
	SIFT sift(params2.patchResolution * 2 + 1, params2.patchRelativeExtent);

	std::vector<SIFT::Descriptor> tempDescriptors = sift((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows, regions, false);

	descriptors.clear();
	descriptors.resize(tempDescriptors.size());
	std::vector<float> data(128, 0.0f);
	for (int i = 0; i < tempDescriptors.size(); ++i)
	{
		for (int j = 0; j < 128; ++j)
		{
			data[j] = static_cast<float>(tempDescriptors[i].data[j]);
		}

		descriptors[i].x = static_cast<float>(tempDescriptors[i].x) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].y = static_cast<float>(tempDescriptors[i].y) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].a = static_cast<float>(tempDescriptors[i].a);
		descriptors[i].b = static_cast<float>(tempDescriptors[i].b);
		descriptors[i].c = static_cast<float>(tempDescriptors[i].c);
		descriptors[i].data = data;
	}

	// Erase improper features
	for (int i = 0; i < descriptors.size(); )
	{
		if (descriptors[i].a * descriptors[i].c - descriptors[i].b * descriptors[i].b < 1.0f)
		{
			descriptors.erase(descriptors.begin() + i);
		}
		else
		{
			++i;
		}
	}
}*/

void detectHessianAffine(const cv::Mat& image, std::vector<Descriptor>& descriptors, const HessianAffineParams& params1, const SIFTParams& params2)
{
	PyramidParams pp;
	pp.threshold = params1.edgeThreshold;

	AffineShapeParams asp;
	asp.maxIterations = 16;
	asp.patchSize = params2.patchResolution * 2 + 1;
	asp.mrSize = params2.patchRelativeExtent;

	SIFTDescriptorParams sp;
	sp.patchSize = params2.patchResolution * 2 + 1;

	AffineHessianDetector detector(image, pp, asp, sp);
	detector.detectPyramidKeypoints(image);

	descriptors.clear();
	descriptors = detector.descriptors;
}

void describeSIFTPatchInPatch(const cv::Mat& image, std::vector<Descriptor>& descriptors, const SIFTParams& params)
{
	cv::Mat tempImage(image.size(), CV_8U);
	if (image.channels() == 1)
	{
		image.convertTo(tempImage, CV_8U);
	}
	if (image.channels() == 3)
	{
		cv::Mat temp(image.size(), CV_32F);
		std::vector<cv::Mat> image_split(3, cv::Mat());
		cv::split(image, image_split);
		temp = (image_split[0] + image_split[1] + image_split[2]) / 3.0;
		temp.convertTo(tempImage, CV_8U);
	}

	// Create object
	SIFT sift(params.patchResolution * 2 + 1, 1.0f);

	float a = static_cast<float>(tempImage.cols) / 2;
	float b = static_cast<float>(tempImage.rows) / 2;

	std::vector<std::vector<float>> vec_xyabc(1, std::vector<float>(5, 0.0f));
	vec_xyabc[0][0] = a - 0.5f; // - 0.5f because of the definition of the coordinate system
	vec_xyabc[0][1] = b - 0.5f; // - 0.5f because of the definition of the coordinate system
	vec_xyabc[0][2] = a * a;
	vec_xyabc[0][3] = 0.0f;
	vec_xyabc[0][4] = b * b;

	std::vector<SIFT::Descriptor> tempDescriptors = sift.describe((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows, vec_xyabc);

	descriptors.clear();
	descriptors.resize(tempDescriptors.size());
	std::vector<float> data(128, 0.0f);
	for (int i = 0; i < tempDescriptors.size(); ++i)
	{
		for (int j = 0; j < 128; ++j)
		{
			data[j] = static_cast<float>(tempDescriptors[i].data[j]);
		}

		descriptors[i].x = static_cast<float>(tempDescriptors[i].x) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].y = static_cast<float>(tempDescriptors[i].y) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].a = static_cast<float>(tempDescriptors[i].a);
		descriptors[i].b = static_cast<float>(tempDescriptors[i].b);
		descriptors[i].c = static_cast<float>(tempDescriptors[i].c);
		descriptors[i].data = data;
	}

	// Erase improper features
	for (int i = 0; i < descriptors.size(); )
	{
		if (descriptors[i].a * descriptors[i].c - descriptors[i].b * descriptors[i].b < 1.0f)
		{
			descriptors.erase(descriptors.begin() + i);
		}
		else
		{
			++i;
		}
	}
}

void describeSIFTPatchInImage(const cv::Mat& image, const Descriptor &descriptor, std::vector<Descriptor>& descriptors, const SIFTParams& params)
{
	// Create object
	SIFT sift(params.patchResolution * 2 + 1, params.patchRelativeExtent);

	std::vector<std::vector<float>> vec_xyabc(1, std::vector<float>(5, 0.0f));
	vec_xyabc[0][0] = descriptor.x - 0.5f; // - 0.5f because of the definition of the coordinate system
	vec_xyabc[0][1] = descriptor.y - 0.5f; // - 0.5f because of the definition of the coordinate system
	vec_xyabc[0][2] = descriptor.a;
	vec_xyabc[0][3] = descriptor.b;
	vec_xyabc[0][4] = descriptor.c;

	std::vector<SIFT::Descriptor> tempDescriptors = sift.describe((uint8_t*)image.data, image.cols, image.rows, vec_xyabc);

	descriptors.clear();
	descriptors.resize(tempDescriptors.size());
	std::vector<float> data(128, 0.0f);
	for (int i = 0; i < tempDescriptors.size(); ++i)
	{
		for (int j = 0; j < 128; ++j)
		{
			data[j] = static_cast<float>(tempDescriptors[i].data[j]);
		}

		descriptors[i].x = static_cast<float>(tempDescriptors[i].x) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].y = static_cast<float>(tempDescriptors[i].y) + 0.5f; // + 0.5f because of the definition of the coordinate system
		descriptors[i].a = static_cast<float>(tempDescriptors[i].a);
		descriptors[i].b = static_cast<float>(tempDescriptors[i].b);
		descriptors[i].c = static_cast<float>(tempDescriptors[i].c);
		descriptors[i].data = data;
	}

	// Erase improper features
	for (int i = 0; i < descriptors.size(); )
	{
		if (descriptors[i].a * descriptors[i].c - descriptors[i].b * descriptors[i].b < 1.0f)
		{
			descriptors.erase(descriptors.begin() + i);
		}
		else
		{
			++i;
		}
	}
}

bool isOutsideImageBoundary(const cv::Mat& image, Descriptor& descriptor, int margin)
{
	float dx = sqrt(descriptor.a);
	float dy = sqrt(descriptor.c);
	return ((0 > descriptor.x - dx - margin) || (0 > descriptor.y - dy - margin) || (image.cols < descriptor.x + dx + margin) || (image.rows < descriptor.y + dy + margin));
}

bool isTooElongated(Descriptor& descriptor)
{
	float a = descriptor.a;
	float b = descriptor.b;
	float c = descriptor.c;
	float d = sqrt((a - c) * (a - c) + 4.0f * b * b);
	return ((a + c + d) / (a + c - d) > 25.0f);
}

bool hasNonPositiveDeterminant(Descriptor& descriptor)
{
	float a = descriptor.a;
	float b = descriptor.b;
	float c = descriptor.c;
	return !((a * c - b * b) > 0.0f);
}

cv::RotatedRect getRotatedRectFromCovarianceMatrix(cv::Point2f pt, cv::Mat C, float magnificationFactor)
{
	// Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(C,eigenvalues, eigenvectors);

	// Calculate the angle between the largest eigenvector and the x-axis
	float angle = atan2(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(0, 0));

	// Shift the angle to [0, 2 * pi] instead of [- pi, pi]
	//if(angle < 0)
	//{
	//	angle += static_cast<float>(2 * CV_PI);
	//}

	// Convert to degrees instead of radians
	angle = angle / static_cast<float>(CV_PI) * 180;

	// Calculate the size of the minor and major axes
	float a = sqrt(eigenvalues.at<float>(0));
	float b = sqrt(eigenvalues.at<float>(1));
	//float a = sqrt(0.5 * (C.at<float>(0, 0) + C.at<float>(1, 1)) + sqrt(pow(0.5 * (C.at<float>(0, 0) - C.at<float>(1, 1)), 2) + C.at<float>(0, 1) * C.at<float>(0, 1))); // This is identical to the above
	//float b = sqrt(0.5 * (C.at<float>(0, 0) + C.at<float>(1, 1)) - sqrt(pow(0.5 * (C.at<float>(0, 0) - C.at<float>(1, 1)), 2) + C.at<float>(0, 1) * C.at<float>(0, 1))); // This is identical to the above

	cv::RotatedRect rr = cv::RotatedRect(pt, cv::Size2f(magnificationFactor * 2 * a, magnificationFactor * 2 * b), angle);
	return rr;
}

bool isOutsideImageBoundary(cv::Mat& image, cv::RotatedRect& rr, int margin)
{
	cv::Rect br = rr.boundingRect();
	return ((0 > br.tl().x - margin) || (0 > br.tl().y - margin) || (image.cols < br.br().x + margin) || (image.rows < br.br().y + margin));
}

cv::Mat extractNormalizedAndOrientedPatch(cv::Mat& src, cv::RotatedRect rr, cv::Size sl, cv::Point2f up, bool adjustOrientation)
{
	// Rotate image by rr.angle around rr.center
	double angle1 = rr.angle / 180 * CV_PI;

	double alpha1 = cos(angle1);
	double beta1 = sin(angle1);

	cv::Mat M1(3, 3, CV_64F);
	double* m1 = M1.ptr<double>();

	m1[0] = alpha1;
	m1[1] = beta1;
	m1[2] = (1 - alpha1) * rr.center.x - beta1 * rr.center.y;
	m1[3] = -beta1;
	m1[4] = alpha1;
	m1[5] = beta1 * rr.center.x + (1 - alpha1) * rr.center.y;
	m1[6] = 0;
	m1[7] = 0;
	m1[8] = 1;

	// Normalize image (from ellipse to circle, from rr.size to sl)
	double sx = sl.width / rr.size.width;
	double sy = sl.height / rr.size.height;

	cv::Mat M2(3, 3, CV_64F);
	double* m2 = M2.ptr<double>();

	m2[0] = sx;
	m2[1] = 0;
	m2[2] = (1 - sx) * rr.center.x;
	m2[3] = 0;
	m2[4] = sy;
	m2[5] = (1 - sy) * rr.center.y;
	m2[6] = 0;
	m2[7] = 0;
	m2[8] = 1;

	// Temporary transformation matrix
	cv::Mat T_temp = M2 * M1;

	cv::Mat T_final;
	if (adjustOrientation)
	{
		// Calculate the point at the end of the semi-major axis
		//cv::Point2f a = cv::Point2f(rr.center.x + cos(- rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f, rr.center.y - sin(- rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f);
		cv::Point3f a = cv::Point3f(rr.center.x + cos(-rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f, rr.center.y - sin(-rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f, 1.0f);

		// Apply temporary transformation matrix
		cv::Mat a_trans = T_temp * cv::Mat(static_cast<cv::Point3d>(a), false);
		cv::Mat up_trans = T_temp * cv::Mat(cv::Point3d(up.x, up.y, 1.0f), false);

		// Compute the angle between a and up
		//cv::Vec2f v1 = cv::Vec2f(a.x - rr.center.x, a.y - rr.center.y);
		//cv::Vec2f v2 = cv::Vec2f(up.x - rr.center.x, up.y - rr.center.y);
		cv::Vec2f v1 = cv::Vec2f(static_cast<float>(a_trans.at<double>(0, 0)) - rr.center.x, static_cast<float>(a_trans.at<double>(1, 0)) - rr.center.y);
		cv::Vec2f v2 = cv::Vec2f(static_cast<float>(up_trans.at<double>(0, 0)) - rr.center.x, static_cast<float>(up_trans.at<double>(1, 0)) - rr.center.y);
		float angle = static_cast<float>(acos(v1.dot(v2) / (cv::norm(v1) * cv::norm(v2)))) / static_cast<float>(CV_PI) * 180;

		// Compute Hesse normal equation
		cv::Vec2f p = cv::Vec2f(rr.center.x, rr.center.y);
		cv::Vec2f n = cv::Vec2f(v1[1], -v1[0]);

		float pn = p.dot(n);

		cv::Vec2f n0;
		if (pn < 0)
		{
			n0 = -n / cv::norm(n);
		}
		else
		{
			n0 = n / cv::norm(n);
		}

		float d0 = p.dot(n0);

		float d = (v2 + p).dot(n0) - d0;

		if (d < 0)
		{
			angle = -angle;
		}

		if (d < 1.0e-03f && angle != angle)
		{
			if (fabs(a.x - up.x) < 1.0e-03f && fabs(a.y - up.y) < 1.0e-03f)
			{
				angle = 0.0f;
			}
			else
			{
				angle = 180.0f;
			}
		}

		double angle2 = static_cast<double>(angle) / 180 * CV_PI;

		double alpha2 = cos(angle2);
		double beta2 = sin(angle2);

		cv::Mat M3(3, 3, CV_64F);
		double* m3 = M3.ptr<double>();

		m3[0] = alpha2;
		m3[1] = beta2;
		m3[2] = (1 - alpha2) * rr.center.x - beta2 * rr.center.y;
		m3[3] = -beta2;
		m3[4] = alpha2;
		m3[5] = beta2 * rr.center.x + (1 - alpha2) * rr.center.y;
		m3[6] = 0;
		m3[7] = 0;
		m3[8] = 1;

		// Final transformation matrix
		T_final = M3 * (M2 * M1);
		T_final.pop_back();
	}
	else
	{
		// Final transformation matrix
		T_final = T_temp;
		T_final.pop_back();
	}

	cv::Mat rotated;
	cv::warpAffine(src, rotated, T_final, src.size(), CV_INTERPOLATION);

	cv::Mat cropped;
	cv::getRectSubPix(rotated, sl, rr.center, cropped);

	return cropped;
}

cv::Mat drawMatches(const std::string& title, const int delay, const cv::Mat& image1, const cv::Mat& image2, 
	const std::vector<Descriptor>& descriptors1, const std::vector<Descriptor>& descriptors2, float magnificationFactor)
{
	CV_Assert(image1.type() == image2.type() && descriptors1.size() == descriptors2.size());
	cv::Mat drawing = cv::Mat::zeros(cv::Size(image1.cols + image2.cols, std::max(image1.rows, image2.rows)), CV_8UC3);
	cv::Rect rect1(0, 0, image1.cols, image1.rows);
	cv::Rect rect2(image1.cols, 0, image2.cols, image2.rows);

	// Allow color drawing
	if (image1.channels() == 1)
	{
		cv::Mat image1Copy(image1.size(), CV_8U), image2Copy(image2.size(), CV_8U);
		image1.convertTo(image1Copy, CV_8U);
		image2.convertTo(image2Copy, CV_8U);
		cv::cvtColor(image1Copy, drawing(rect1), CV_GRAY2BGR);
		cv::cvtColor(image2Copy, drawing(rect2), CV_GRAY2BGR);
	}
	if (image1.channels() == 3)
	{
		image1.convertTo(drawing(rect1), CV_8UC3);
		image2.convertTo(drawing(rect2), CV_8UC3);
	}

	cv::RotatedRect rr1, rr2;
	float angle1, angle2; std::vector<uint>::iterator  it; uint found=1;
	cv::Scalar color1 = cv::Scalar(255, 0, 0);
	cv::Scalar color2 = cv::Scalar(0, 255, 0);
	for (int i = 0; i < descriptors1.size(); ++i)
		//for (int j = 0; j < descriptors2.size();++j)
		{		
			rr1 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors1[i].x, descriptors1[i].y), (cv::Mat_<float>(2, 2) << descriptors1[i].a, descriptors1[i].b, descriptors1[i].b, descriptors1[i].c), magnificationFactor);
			rr2 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors2[i].x, descriptors2[i].y), (cv::Mat_<float>(2, 2) << descriptors2[i].a, descriptors2[i].b, descriptors2[i].b, descriptors2[i].c), magnificationFactor);
			rr2.center.x = rr2.center.x + image1.cols;
			angle1 = -rr1.angle / 180 * static_cast<float>(CV_PI);
			angle2 = -rr2.angle / 180 * static_cast<float>(CV_PI);

			cv::line(drawing, rr1.center, rr2.center, color1, 1, CV_AA, 0);
			cv::ellipse(drawing, rr1, color2, 1, CV_AA);
			cv::line(drawing, rr1.center, cv::Point2f(rr1.center.x + cos(angle1) * rr1.size.width * 0.5f, rr1.center.y - sin(angle1) * rr1.size.width * 0.5f), color2, 1, CV_AA, 0);
			//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr1.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color2);
			cv::ellipse(drawing, rr2, color2, 1, CV_AA);
			cv::line(drawing, rr2.center, cv::Point2f(rr2.center.x + cos(angle2) * rr2.size.width * 0.5f, rr2.center.y - sin(angle2) * rr2.size.width * 0.5f), color2, 1, CV_AA, 0);
			//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr2.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color2);
			
		}
	cv::imshow(title, drawing);
	return drawing;
}

std::vector<std::string> floatToString(std::vector<float> floats) {
	std::vector<std::string> buffer;
	for (uint i = 0; i < floats.size(); ++i)
		buffer.push_back(std::to_string(floats.at(i)));
	return buffer;
}
//returns predicted NONmatches from 'descriptor' in 'descriptors' and writes matches in 'descriptors'
void rangerCheck(std::vector<Descriptor> descriptors1, std::vector<Descriptor> descriptors2)
{
	Speicher Speicher;
	std::vector<Descriptor> rangerPredicted1;
	std::vector<Descriptor> rangerPredicted2;
	std::vector<std::string>rangerSetUp;
	std::string speicher;
	//construct ranger input
	//für jeden Deskriptor
	for (uint x = 0; x < descriptors1.size(); ++x)
		//für zum Deskriptor x gehörende Deskriptoren
		for (uint y = 0; y < descriptors2.size();++y)
		{	
			//jeden Deskriptor aus y einlesen
			for (uint m = 0; m < descriptors2.at(y).data.size(); ++m)
				speicher = speicher + std::to_string(descriptors2.at(y).data.at(m))+" ";
			//mit Deskriptor x abspeichern
			for (uint m = 0; m < descriptors1.at(x).data.size(); ++m)
				speicher = speicher + std::to_string(descriptors1.at(x).data.at(m)) + " ";
		speicher.pop_back();
		rangerSetUp.push_back(speicher);
		speicher.clear();
	}
	

	Speicher.WriteText(rangerSetUp, "data.dat", "J:\\VC\\Ranger\\");
	//run Ranger 
	system("J:\\VC\\Ranger\\Ranger.exe");
	rangerSetUp.clear();	
	//fill rangerSetUp with "ranger_out.prediction"
	rangerSetUp = Speicher.ReadText("J:\\VC\\Ranger\\", "ranger_out.prediction");
	//cv::Mat rangerMatch= cv::Mat((rangerSetUp.size(),2),CV_64F,0.0);
	//find "1 " in rangerSetUp and set flag
	for (uint x = 0; x < rangerSetUp.size() + 1; ++x)
		if (rangerSetUp.at(x-1) == "1")
		{
			rangerPredicted1.push_back(descriptors1.at(x / descriptors1.size()));
			rangerPredicted1.push_back(descriptors2.at(x % descriptors1.size()));
		}
	descriptors1 = rangerPredicted1;
	descriptors2 = rangerPredicted2;
}

int mainV(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
		return EXIT_FAILURE;
	}
	/*
	// Read config file
	libconfig::Config cfg;
	try
	{
		cfg.readFile(argv[1]);
	}
	catch (const libconfig::FileIOException &fioex)
	{
		std::cerr << "I/O error while reading file." << std::endl;
		return EXIT_FAILURE;
	}
	catch (const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
		return EXIT_FAILURE;
	}*/

	// Set MODS parameters
	std::vector<float> tilts;
	std::vector<float> scales;
	std::vector<MODSParams> params_mods;
	// MSER
	//tilts.clear();
	//tilts.push_back(1.0f);
	//tilts.push_back(3.0f);
	//tilts.push_back(6.0f);
	//tilts.push_back(9.0f);
	//scales.clear();
	//scales.push_back(1.0f);
	//scales.push_back(0.25f);
	//scales.push_back(0.125f);
	//params_mods.push_back(MODSParams(1.0f, tilts, scales, 360.0f));
	tilts.clear();
	tilts.resize(4);//cfg.lookup("MODSParams.MSER.tilts").getLength());
	/*
	for (int i = 0; i < tilts.size(); i++)
	{
		tilts[i] = static_cast<float>(cfg.lookup("MODSParams.MSER.tilts")[i]);
	}*/
	tilts[0] = 1;
	tilts[1] = 3;
	tilts[2] = 6;
	tilts[3] = 9;
	scales.clear();
	scales.resize(3);//cfg.lookup("MODSParams.MSER.scales").getLength());
	/*for (int i = 0; i < scales.size(); i++)
	{
		scales[i] = static_cast<float>(cfg.lookup("MODSParams.MSER.scales")[i]);
	}*/
	scales.push_back(1);
	scales.push_back(0.25);
	scales.push_back(0.125);
	//params_mods.push_back(MODSParams(cfg.lookup("MODSParams.MSER.sigma"), tilts, scales, cfg.lookup("MODSParams.MSER.angle")));
	params_mods.push_back(MODSParams(1, tilts, scales, 360));
	// Hessian-Affine
	tilts.clear();
	tilts.push_back(1.0f);
	tilts.push_back(2.0f);
	tilts.push_back(4.0f);
	tilts.push_back(6.0f);
	tilts.push_back(8.0f);
	scales.clear();
	scales.push_back(1.0f);
	params_mods.push_back(MODSParams(1.0f, tilts, scales, 60.0f));

	// Set MSER and Hessian-Affine parameters
	//MSERParams params_mser(2, 30, 0.01f, 0.25f, 0.2f);
	//HessianAffineParams params_ha(0, 3, 16.0f / 3.0f, 10.0f, 0.01f);
	//SIFTParams params_sift(0, - 1, 3, 20, 3.0f * sqrt(3.0f), 1.0f);
	/*
	MSERParams params_mser(
		static_cast<int>(2),//cfg.lookup("MSERParams.delta")),
		static_cast<int>(30),//cfg.lookup("MSERParams.minArea")),
		static_cast<float>(0.01),//cfg.lookup("MSERParams.maxArea")),
		static_cast<float>(0.25),//cfg.lookup("MSERParams.maxVariation")),
		static_cast<float>(0.2)//cfg.lookup("MSERParams.minDiversity"))
		);
		*/
	HessianAffineParams params_ha(
		static_cast<int>(0),//cfg.lookup("HessianAffineParams.firstOctave")),
		static_cast<int>(3),//cfg.lookup("HessianAffineParams.octaveResolution")),
		static_cast<float>(5.3333333333333333333333333333333),//cfg.lookup("HessianAffineParams.peakThreshold")),
		static_cast<float>(10),//cfg.lookup("HessianAffineParams.edgeThreshold")),
		static_cast<float>(0.01)//cfg.lookup("HessianAffineParams.laplacianPeakThreshold"))
		);
	SIFTParams params_sift(
		static_cast<int>(0),//cfg.lookup("SIFTParams.method")),
		static_cast<int>(-1),//cfg.lookup("SIFTParams.firstOctave")),
		static_cast<int>(3),//cfg.lookup("SIFTParams.octaveResolution")),
		static_cast<int>(20),//cfg.lookup("SIFTParams.patchResolution")),
		static_cast<float>(5.1961524227066318805823390245176),//cfg.lookup("SIFTParams.patchRelativeExtent")),
		static_cast<float>(1)//cfg.lookup("SIFTParams.patchRelativeSmoothing"))
		);
	// Set other parameters
	std::string fileExtension = static_cast<std::string>(".jpg");// cfg.lookup("fileExtension").c_str());
	bool useColor = static_cast<bool>(true);// cfg.lookup("useColor"));
	int method = static_cast<int>(0);// cfg.lookup("method"));
	int antialiasing = static_cast<int>(1);//cfg.lookup("antialiasing"));
	float magnificationFactor = static_cast<float>(1);//cfg.lookup("magnificationFactor"));
	float minimumPatchSize = static_cast<float>(3);//cfg.lookup("minimumPatchSize"));
	int patchSize = static_cast<int>(31);//cfg.lookup("patchSize"));
	bool adjustOrientation = static_cast<bool>(true);//cfg.lookup("adjustOrientation"));
	bool describePatchWithSIFT = static_cast<bool>(true);//cfg.lookup("describePatchWithSIFT"));
	int whereToDescribe = static_cast<int>(0);//cfg.lookup("whereToDescribe"));
	bool writeToDisk = static_cast<bool>(true);//cfg.lookup("writeToDisk"));
	bool displayPatch = static_cast<bool>(true);//cfg.lookup("displayPatch"));
	bool displayAffineRegions = static_cast<bool>(true);//cfg.lookup("displayAffineRegions"));

	// Define input path
	boost::filesystem::path inputPath("C:\VC\\in");//cfg.lookup("inputPath").c_str());

	// Check if input path exists
	if (!boost::filesystem::exists(inputPath) || !boost::filesystem::is_directory(inputPath))
	{
		perror(reinterpret_cast<const char*>(inputPath.c_str()));
		return EXIT_FAILURE;
	}

	// Define output path
	boost::filesystem::path outputPath("C:\VC\\out");//cfg.lookup("outputPath").c_str());
	boost::filesystem::create_directory(outputPath);

	// Find all images
	std::vector<boost::filesystem::path> files;
	for (boost::filesystem::directory_iterator it(inputPath); it != boost::filesystem::directory_iterator(); ++it)
	{
		if (boost::filesystem::is_regular_file(it->status()) && boost::algorithm::to_lower_copy(it->path().extension().string()) == fileExtension)
		{
			files.push_back(it->path());
		}
	}
	
	//storage for image for later use
	cv::Mat image_last;
	// Detected features for later use
	std::vector<Descriptor> descriptorsLast;
	Speicher Speicher;
	// Iterate through images
	for (int image_iter = 0; image_iter < files.size(); ++image_iter)
	{
		// Create directory for each image
		boost::filesystem::path imagePath = outputPath;
		imagePath += "/";
		imagePath += files[image_iter].stem();
		boost::filesystem::create_directory(imagePath);

		// Load image
		cv::Mat image_gray, image_color;
		loadImage(files[image_iter].string(), image_gray, image_color);
		image_color.convertTo(image_color, CV_32FC3);		
		
		
		std::cout << "Processing image " << image_iter + 1 << " of " << files.size() << " (" << files[image_iter].stem().string() << ")" << std::endl;
		// Detect features
		std::vector<Descriptor> descriptors;
		bool flag;
		for (int detector_iter = 0; detector_iter < params_mods.size(); ++detector_iter)
		{
			// Create directory for each detector
			boost::filesystem::path detectorPath = imagePath;

			/*
			//MSER
			if (detector_iter == 0)
			{
				detectMSER(image_gray, descriptors, params_mser, params_sift);

				std::cout << "Found " << descriptors.size() << " affine regions (MSER)" << std::endl;

				// Create directory for MSER
				detectorPath += "/MSER";
				boost::filesystem::create_directory(detectorPath);
			}*/
			//Hessian-Affine
			if (detector_iter == 1)
			{
				detectHessianAffine(image_gray, descriptors, params_ha, params_sift);

				std::cout << "Found " << descriptors.size() << " affine regions (Hessian-Affine)" << std::endl;

				// Create directory for Hessian-Affine
				detectorPath += "/Hessian-Affine";
				boost::filesystem::create_directory(detectorPath);
			}

			// Remove improper features
			for (int i = 0; i < descriptors.size(); )
			{
				if (isOutsideImageBoundary(image_gray, descriptors[i], 0) || hasNonPositiveDeterminant(descriptors[i])) //isTooElongated(descriptors[i]) || 
				{
					descriptors.erase(descriptors.begin() + i);
				}
				else
				{
					++i;
				}
			}
			/*
			// Create warp configuration
			WarpConfiguration warpConfiguration;
			createWarpConfiguration(params_mods[detector_iter], warpConfiguration);

			int numWarps = static_cast<int>(warpConfiguration.theta.size());

			std::vector<cv::Mat> warpedImages(numWarps, cv::Mat());
			for (int i = 0; i < numWarps; ++i)
			{
				std::cout << "Processing warp " << i + 1 << " of " << numWarps << " -> ";

				ImageWarper iw(image_gray, warpConfiguration.theta[i], warpConfiguration.phi[i], warpConfiguration.scale[i], method, antialiasing, params_mods[detector_iter].sigma);
				iw.warpImage(useColor ? image_color : image_gray, warpedImages[i]);

				cv::Mat tempImage(warpedImages[i].size(), CV_8U);
				if (useColor)
				{
					cv::Mat temp(warpedImages[i].size(), CV_32F);
					std::vector<cv::Mat> image_split(3, cv::Mat());
					cv::split(warpedImages[i], image_split);
					temp = (image_split[0] + image_split[1] + image_split[2]) / 3.0;
					temp.convertTo(tempImage, CV_8U);
				}

				std::vector<Descriptor> wdesc(descriptors.size());

				// Iterate through features
				cv::KeyPoint kpt, wkpt, wwkpt;
				cv::Mat C, wC, wwC;
				cv::RotatedRect rr, wrr;
				cv::Point2f a, wa;
				cv::Mat patch, temp;
				std::vector<Descriptor> patch_desc;
				int count = 0;
				for (int j = 0; j < descriptors.size(); ++j)
				{
					kpt.pt = cv::Point2f(descriptors[j].x, descriptors[j].y);
					wkpt.pt = iw.warpPointForward(kpt.pt);
					//wwkpt.pt = iw.warpPointInverse(wkpt.pt); // Check if everything works

					C = (cv::Mat_<float>(2, 2) <<
						descriptors[j].a, descriptors[j].b,
						descriptors[j].b, descriptors[j].c);
					wC = iw.warpCovarianceMatrixForward(kpt.pt, C);
					//wwC = iw.warpCovarianceMatrixInverse(wkpt.pt, wC); // Check if everything works

					wdesc[j].x = wkpt.pt.x;
					wdesc[j].y = wkpt.pt.y;
					wdesc[j].a = wC.at<float>(0, 0);
					wdesc[j].b = wC.at<float>(0, 1);
					wdesc[j].c = wC.at<float>(1, 1);

					rr = getRotatedRectFromCovarianceMatrix(kpt.pt, C, magnificationFactor);
					wrr = getRotatedRectFromCovarianceMatrix(wkpt.pt, wC, magnificationFactor);

					if (!isOutsideImageBoundary(image_gray, rr, 0) && wrr.size.height > minimumPatchSize && wrr.size.width > minimumPatchSize)
					{
						a = cv::Point2f(kpt.pt.x + cos(-rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f, kpt.pt.y - sin(-rr.angle / 180 * static_cast<float>(CV_PI)) * rr.size.width * 0.5f);
						wa = iw.warpPointForward(a);

						// Extract patch
						patch = extractNormalizedAndOrientedPatch(warpedImages[i], wrr, cv::Size(patchSize, patchSize), wa, adjustOrientation);

						// Display patch
						if (displayPatch)
						{
							patch.convertTo(temp, CV_8UC3);
							cv::namedWindow("Display patch", CV_WINDOW_AUTOSIZE & CV_WINDOW_FREERATIO & CV_GUI_EXPANDED);
							cv::imshow("Display patch", temp);
							cv::waitKey(1);
						}

						// Describe patch with SIFT
						if (describePatchWithSIFT)
						{
							if (whereToDescribe == 0)
							{
								describeSIFTPatchInImage(tempImage, wdesc[j], patch_desc, params_sift);
							}
							if (whereToDescribe == 1)
							{
								describeSIFTPatchInPatch(patch, patch_desc, params_sift);
							}
						}

						// Save patch and SIFT descriptor
						boost::filesystem::path featurePath = detectorPath;
						featurePath += "/";
						featurePath += std::to_string(static_cast<long double>(j));
						boost::filesystem::create_directory(featurePath);

						std::stringstream filename1, filename2;
						filename1 << featurePath.string() << "/" << i << ".jpg";
						filename2 << featurePath.string() << "/" << i << ".sift";

						if (writeToDisk)
						{
							cv::imwrite(filename1.str(), patch);

							if (describePatchWithSIFT)
							{
								std::ofstream file;
								file.open(filename2.str());
								for (int k = 0; k < patch_desc.size(); ++k)
								{
									for (int l = 0; l < 127; ++l)
									{
										file << patch_desc[k].data[l] << " ";
									}
									file << patch_desc[k].data[127] << std::endl;
								}
								file.close();
							}
						}

						count++;
					}
				}//für Deskriptor.size

				std::cout << "Saved " << count << " patches" << std::endl;

				// Display affine regions
				if (displayAffineRegions)
				{
					//drawMatches("Display affine regions", 1, image_color, warpedImages[i], descriptors, wdesc, magnificationFactor);
				}
			}//für jedes warp
		}*/
		//für jeden Detektor
			if (image_iter > 0)
			{
				cv::Mat image = image_color;

				rangerCheck(descriptorsLast, descriptors);

				///rangerPredicted aufräumen

				//einspeichern der positiven Matches

			//Matches einzeichnen
				image = drawMatches("", 1, image, image_last, descriptorsLast, descriptors,  magnificationFactor);
				//Bild mit Matches abspeichern / anzeigen				
				cv::imwrite("Image.jpg", image);
				cv::imshow("Image.jpg", image);

			}
			else
			{
				// Detect features from picture before
				std::vector<Descriptor> descriptorsLast = descriptors;
				//save image for later use
				cv::Mat image_last = image_color;
			}
		}
	}//für jedes Bild
}/**/