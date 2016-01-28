/**
* @brief tpm.cpp
* @author Lukas Roth
*/
#include <fstream>
#include <stdexcept>
#include <string>
#include <windows.h>
#include "speicher.h"

#include "hesaff/pyramid.h"
#include "hesaff/affine.h"
#include "hesaff/siftdesc.h"
#include "mser/mser.h"
#include "mser/sift.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <boost/filesystem.hpp>
//#include <libconfig.h++>

#include <iostream>
#include <fstream>

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
		if(!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
		{
			// Compute SIFT descriptor
			sift.computeSiftDescriptor(this->patch);

			float sc = this->mrSize * s;
			cv::Mat A = (cv::Mat_<float>(2,2) << a11, a12, a21, a22); // A = [a, b; 0, c] (oriented ellipse)
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

	if(image.channels() == 3)
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

void detectMSER(const cv::Mat& image, std::vector<Descriptor>& descriptors, const MSERParams& params1, const SIFTParams& params2)
{
	// Convert image from CV_32F to CV_8U
	cv::Mat tempImage(image.size(), CV_8U);
	image.convertTo(tempImage, CV_8U);

	int imageArea = tempImage.cols * tempImage.rows;
	int maxArea = static_cast<int>(params1.maxArea * imageArea);
	
	// Return if minArea > maxArea
	if(params1.minArea > maxArea)
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

	std::vector<SIFT::Descriptor> tempDescriptors = sift((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows, regions);
	
	descriptors.clear();
	descriptors.resize(tempDescriptors.size());
	std::vector<float> data(128, 0.0f);
	for(int i = 0; i < tempDescriptors.size(); ++i)
	{
		for(int j = 0; j < 128; ++j)
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
	for(int i = 0; i < descriptors.size(); )
	{
		if(descriptors[i].a * descriptors[i].c - descriptors[i].b * descriptors[i].b < 1.0f)
		{
			descriptors.erase(descriptors.begin() + i);
		}
		else
		{
			++i;
		}
	}
}

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

void describeSIFT(const cv::Mat& image, std::vector<Descriptor>& descriptors, const SIFTParams& params)
{
	// Convert image from CV_32F to CV_8U
	cv::Mat tempImage(image.size(), CV_8U);
	image.convertTo(tempImage, CV_8U);
	
	// Create object
	SIFT sift(params.patchResolution * 2 + 1, params.patchRelativeExtent);
	
	std::vector<std::vector<float>> vec_xyabc(descriptors.size(), std::vector<float>(5, 0.0f));
	for(int i = 0; i < descriptors.size(); ++i)
	{
		vec_xyabc[i][0] = descriptors[i].x - 0.5f; // - 0.5f because of the definition of the coordinate system
		vec_xyabc[i][1] = descriptors[i].y - 0.5f; // - 0.5f because of the definition of the coordinate system
		vec_xyabc[i][2] = descriptors[i].a;
		vec_xyabc[i][3] = descriptors[i].b;
		vec_xyabc[i][4] = descriptors[i].c;
	}

	std::vector<SIFT::Descriptor> tempDescriptors = sift.describe((uint8_t*)tempImage.data, tempImage.cols, tempImage.rows, vec_xyabc);
	
	descriptors.clear();
	descriptors.resize(tempDescriptors.size());
	std::vector<float> data(128, 0.0f);
	for(int i = 0; i < tempDescriptors.size(); ++i)
	{
		for(int j = 0; j < 128; ++j)
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
	for(int i = 0; i < descriptors.size(); )
	{
		if(descriptors[i].a * descriptors[i].c - descriptors[i].b * descriptors[i].b < 1.0f)
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

void collapseDescriptors(const std::vector<Descriptor>& src1, const std::vector<Descriptor>& src2, cv::Mat& dst1, cv::Mat& dst2)
{
	int size = src1[0].data.size();
	dst1.release();
	dst2.release();
	for(int i = 0; i < src1.size(); ++i)
	{
		dst1.push_back(cv::Mat(1, size, CV_32F, (float*)&src1[i].data[0]));
	}
	for(int i = 0; i < src2.size(); ++i)
	{
		dst2.push_back(cv::Mat(1, size, CV_32F, (float*)&src2[i].data[0]));
	}
}

void transformSIFTToRootSIFT(cv::Mat& descriptors)
{
	cv::Mat temp;
	double sum;
	for(int i = 0; i < descriptors.rows; ++i)
	{
		if(!descriptors.row(i).empty())
		{
			cv::normalize(descriptors.row(i), descriptors.row(i), 1.0, cv::NORM_L1);
			cv::reduce(descriptors.row(i), temp, 1, CV_REDUCE_SUM);
			sum = 1.0 / static_cast<double>(temp.at<float>(0, 0));
			if(sum != 0)
			{
				//cv::divide(sum, descriptors.row(i), descriptors.row(i));
				descriptors.row(i) = static_cast<float>(sum) * descriptors.row(i);
				cv::sqrt(descriptors.row(i), descriptors.row(i));
			}
		}
	}
}

std::vector<cv::DMatch> doSNNDistanceRatioTest(const std::vector<std::vector<cv::DMatch>>& srcMatches, float ratio)
{
	std::vector<cv::DMatch> dstMatches;
	ratio = ratio * ratio; // Avoid sqrt(...)
	for(int i = 0; i < srcMatches.size(); ++i)
	{
		if(srcMatches[i][0].distance < ratio * srcMatches[i][1].distance)
		{
			dstMatches.push_back(srcMatches[i][0]);
		}
	}
	return dstMatches;
}

std::vector<cv::DMatch> doSNNDistanceRatioTest(const cv::Mat& indices, const cv::Mat& dists, float ratio, float descdist)
{
	std::vector<cv::DMatch> matches;
	ratio = ratio * ratio; // Avoid sqrt(...)
	descdist = descdist * descdist; // Avoid sqrt(...)
	for(int i = 0; i < indices.rows; ++i)
	{
		const int* ptr_indices = indices.ptr<const int>(i);
		const float* ptr_dists = dists.ptr<const float>(i);

		if(ptr_dists[0] < ratio * ptr_dists[1] && ptr_dists[0] < descdist)
		{
			matches.push_back(cv::DMatch(i, ptr_indices[0], ptr_dists[0]));
		}
	}
	return matches;
}

std::vector<cv::DMatch> doFGINNDistanceRatioTest(const std::vector<std::vector<cv::DMatch>>& srcMatches, const std::vector<Descriptor>& descriptors1, const std::vector<Descriptor>& descriptors2, float distance, float ratio)
{
	distance = distance * distance; // Avoid sqrt(...)
	ratio = ratio * ratio; // Avoid sqrt(...)
	std::vector<cv::DMatch> dstMatches;
	float x1, y1, x2, y2;
	for(int i = 0; i < srcMatches.size(); ++i)
	{
		x1 = descriptors2[srcMatches[i][0].trainIdx].x;
		y1 = descriptors2[srcMatches[i][0].trainIdx].y;

		for(int j = 0; j < srcMatches[i].size(); ++j)
		{
			x2 = descriptors2[srcMatches[i][j].trainIdx].x;
			y2 = descriptors2[srcMatches[i][j].trainIdx].y;

			// Find first geometrically inconsistent nearest neighbor
			if((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) > distance)
			{
				if(srcMatches[i][0].distance < ratio * srcMatches[i][j].distance)
				{
					dstMatches.push_back(srcMatches[i][0]);
				}
				break;
			}
		}
	}
	return dstMatches;
}

std::vector<cv::DMatch> doFGINNDistanceRatioTest(const cv::Mat& indices, const cv::Mat& dists, const std::vector<Descriptor>& descriptors1, const std::vector<Descriptor>& descriptors2, float geomdist, float ratio, float descdist)
{
	geomdist = geomdist * geomdist; // Avoid sqrt(...)
	ratio = ratio * ratio; // Avoid sqrt(...)
	descdist = descdist * descdist; // Avoid sqrt(...)
	std::vector<cv::DMatch> matches;
	float x1, y1, x2, y2;
	for(int i = 0; i < indices.rows; ++i)
	{
		const int* ptr_indices = indices.ptr<const int>(i);
		const float* ptr_dists = dists.ptr<const float>(i);

		x1 = descriptors2[ptr_indices[0]].x;
		y1 = descriptors2[ptr_indices[0]].y;

		for(int j = 1; j < indices.cols; ++j)
		{
			x2 = descriptors2[ptr_indices[j]].x;
			y2 = descriptors2[ptr_indices[j]].y;

			// Find first geometrically inconsistent nearest neighbor
			if((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) > geomdist)
			{
				if(ptr_dists[0] < ratio * ptr_dists[j] && ptr_dists[0] < descdist)
				{
					matches.push_back(cv::DMatch(i, ptr_indices[0], ptr_dists[0]));
				}
				break;
			}
		}
	}
	return matches;
}

bool comp_matches_distance(cv::DMatch i, cv::DMatch j)
{
	return (i.distance < j.distance);
}

bool comp_matches_queryidx(cv::DMatch i, cv::DMatch j)
{
	return (i.queryIdx < j.queryIdx);
}

bool comp_matches_trainidx(cv::DMatch i, cv::DMatch j)
{
	return (i.trainIdx < j.trainIdx);
}

bool pred_matches_queryidx(cv::DMatch i, cv::DMatch j)
{
	return (i.queryIdx == j.queryIdx);
}

bool pred_matches_trainidx(cv::DMatch i, cv::DMatch j)
{
	return (i.trainIdx == j.trainIdx);
}

void sortMatchesQueryIdx(std::vector<cv::DMatch>& matches)
{
	// Sort by id
	std::sort(matches.begin(), matches.end(), comp_matches_queryidx);
	// Sort by distance
	std::vector<cv::DMatch> temp1, temp2;
	temp1.push_back(matches[0]);
	for(int i = 1; i < matches.size(); ++i)
	{
		if(matches[i].queryIdx == temp1.back().queryIdx)
		{
			temp1.push_back(matches[i]);
		}
		else
		{
			std::sort(temp1.begin(), temp1.end(), comp_matches_distance);
			temp2.insert(temp2.end(), temp1.begin(), temp1.end());
			temp1.clear();
			temp1.push_back(matches[i]);
		}
	}
	std::sort(temp1.begin(), temp1.end(), comp_matches_distance);
	temp2.insert(temp2.end(), temp1.begin(), temp1.end());
	matches = temp2;
}

void sortMatchesTrainIdx(std::vector<cv::DMatch>& matches)
{
	// Sort by id
	std::sort(matches.begin(), matches.end(), comp_matches_trainidx);
	// Sort by distance
	std::vector<cv::DMatch> temp1, temp2;
	temp1.push_back(matches[0]);
	for(int i = 1; i < matches.size(); ++i)
	{
		if(matches[i].trainIdx == temp1.back().trainIdx)
		{
			temp1.push_back(matches[i]);
		}
		else
		{
			std::sort(temp1.begin(), temp1.end(), comp_matches_distance);
			temp2.insert(temp2.end(), temp1.begin(), temp1.end());
			temp1.clear();
			temp1.push_back(matches[i]);
		}
	}
	std::sort(temp1.begin(), temp1.end(), comp_matches_distance);
	temp2.insert(temp2.end(), temp1.begin(), temp1.end());
	matches = temp2;
}

std::vector<cv::DMatch> mergeMatches(const std::vector<cv::DMatch>& matches12, const std::vector<cv::DMatch>& matches21)
{
	std::vector<cv::DMatch> matches, temp;
	matches = matches12;
	temp = matches21;
	int id;
	for(int i = 0; i < temp.size(); ++i)
	{
		id = temp[i].trainIdx;
		temp[i].trainIdx = temp[i].queryIdx;
		temp[i].queryIdx = id;
	}
	matches.insert(matches.end(), temp.begin(), temp.end());
	sortMatchesTrainIdx(matches);
	std::vector<cv::DMatch>::iterator itTrainIdx = std::unique(matches.begin(), matches.end(), pred_matches_trainidx);
	matches.erase(itTrainIdx, matches.end());
	sortMatchesQueryIdx(matches);
	std::vector<cv::DMatch>::iterator itQueryIdx = std::unique(matches.begin(), matches.end(), pred_matches_queryidx);
	matches.erase(itQueryIdx, matches.end());
	return matches;
}

void filterMatchesUsingEpipolarGeometry(const std::vector<Descriptor>& descriptors1, const std::vector<Descriptor>& descriptors2, const std::vector<cv::DMatch>& srcMatches, std::vector<cv::DMatch>& dstMatches, cv::Mat& F)
{
	std::vector<int> queryIdxs(srcMatches.size(), 0), trainIdxs(srcMatches.size(), 0);
	for(int i = 0; i < srcMatches.size(); ++i)
	{
		queryIdxs[i] = srcMatches[i].queryIdx;
		trainIdxs[i] = srcMatches[i].trainIdx;
	}
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	for(int i = 0; i < descriptors1.size(); ++i)
	{
		keypoints1.push_back(cv::KeyPoint(descriptors1[i].x, descriptors1[i].y, 0.0f));
	}
	for(int i = 0; i < descriptors2.size(); ++i)
	{
		keypoints2.push_back(cv::KeyPoint(descriptors2[i].x, descriptors2[i].y, 0.0f));
	}
	std::vector<cv::Point2f> points1, points2;
	cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
	cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
	std::vector<unsigned char> inliers(srcMatches.size(), 0);
	F = cv::findFundamentalMat(points1, points2, inliers, CV_FM_RANSAC, 3.0, 0.99);
	for(int i = 0; i < srcMatches.size(); ++i)
	{
		if(inliers[i] > 0)
		{
			dstMatches.push_back(srcMatches[i]);
		}
	}
	//cv::Mat inliers;
	//F = cv::findHomography(points1, points2, CV_FM_RANSAC, 0.99, inliers);
	//for(int i = 0; i < srcMatches.size(); ++i)
	//{
	//	if(inliers.at<uchar>(i))
	//	{
	//		dstMatches.push_back(srcMatches[i]);
	//	}
	//}
}

cv::RotatedRect getRotatedRectFromCovarianceMatrix(cv::Point2f pt, cv::Mat C, float magnificationFactor)
{
	// Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(C,  eigenvalues, eigenvectors);

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

void drawMatches(const std::string& title, const int delay, const cv::Mat& image1,  const std::vector<Descriptor>& descriptors1, const std::vector<Descriptor>& descriptors2)
{
	//CV_Assert(image1.type() == image2.type() && descriptors1.size() == descriptors2.size());
	cv::Mat drawing = cv::Mat::zeros(cv::Size(image1.cols , image1.rows), CV_8UC3);
	cv::Rect rect1(0, 0, image1.cols, image1.rows);
	//cv::Rect rect2(image1.cols, 0, image2.cols, image2.rows);

	// Allow color drawing
	if(image1.channels() == 1)
	{
		cv::Mat image1Copy(image1.size(), CV_8U);
		image1.convertTo(image1Copy, CV_8U);		
		cv::cvtColor(image1Copy, drawing(rect1), CV_GRAY2BGR);		
	}
	if(image1.channels() == 3)
	{
		image1.convertTo(drawing(rect1), CV_8UC3);
	}

	cv::RotatedRect rr1, rr2;
	float angle1, angle2;
	cv::Scalar color1 = cv::Scalar(255, 0, 0);
	cv::Scalar color2 = cv::Scalar(0, 255, 0);
	cv::Scalar color3 = cv::Scalar(0, 0, 255);
		for(int i = 0; i < descriptors2.size(); ++i)
	{
		rr1 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors2[i].x, descriptors2[i].y), (cv::Mat_<float>(2, 2) << descriptors2[i].a, descriptors2[i].b, descriptors2[i].b, descriptors2[i].c), 1.0f);
		//rr2 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors4[i].x, descriptors4[i].y), (cv::Mat_<float>(2, 2) << descriptors4[i].a, descriptors4[i].b, descriptors4[i].b, descriptors4[i].c), 1.0f);
		//rr2.center.x = rr2.center.x + image1.cols;
		angle1 = - rr1.angle / 180 * static_cast<float>(CV_PI);
		//angle2 = - rr2.angle / 180 * static_cast<float>(CV_PI);

		//cv::line(drawing, rr1.center, rr2.center, color1, 1, CV_AA, 0);
		cv::ellipse(drawing, rr1, color3, 1, CV_AA);
		cv::line(drawing, rr1.center, cv::Point2f(rr1.center.x + cos(angle1) * rr1.size.width * 0.5f, rr1.center.y - sin(angle1) * rr1.size.width * 0.5f), color3, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr1.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color3);
		//cv::ellipse(drawing, rr2, color3, 1, CV_AA);
		//cv::line(drawing, rr2.center, cv::Point2f(rr2.center.x + cos(angle2) * rr2.size.width * 0.5f, rr2.center.y - sin(angle2) * rr2.size.width * 0.5f), color3, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr2.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color3);
	}
	/*
	for(int i = 0; i < descriptors1.size(); ++i)
	{
		//rr1 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors3[i].x, descriptors3[i].y), (cv::Mat_<float>(2, 2) << descriptors3[i].a, descriptors3[i].b, descriptors3[i].b, descriptors3[i].c), 1.0f);
		rr2 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors1[i].x, descriptors1[i].y), (cv::Mat_<float>(2, 2) << descriptors4[i].a, descriptors4[i].b, descriptors4[i].b, descriptors4[i].c), 1.0f);
		rr2.center.x = rr2.center.x + image1.cols;
		//angle1 = - rr1.angle / 180 * static_cast<float>(CV_PI);
		angle2 = - rr2.angle / 180 * static_cast<float>(CV_PI);

		//cv::line(drawing, rr1.center, rr2.center, color1, 1, CV_AA, 0);
		//cv::ellipse(drawing, rr1, color3, 1, CV_AA);
		//cv::line(drawing, rr1.center, cv::Point2f(rr1.center.x + cos(angle1) * rr1.size.width * 0.5f, rr1.center.y - sin(angle1) * rr1.size.width * 0.5f), color3, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr1.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color3);
		cv::ellipse(drawing, rr2, color3, 1, CV_AA);
		cv::line(drawing, rr2.center, cv::Point2f(rr2.center.x + cos(angle2) * rr2.size.width * 0.5f, rr2.center.y - sin(angle2) * rr2.size.width * 0.5f), color3, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr2.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color3);
	}
	*/
	for(int i = 0; i < descriptors1.size(); ++i)
	{
		rr1 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors1[i].x, descriptors1[i].y), (cv::Mat_<float>(2, 2) << descriptors1[i].a, descriptors1[i].b, descriptors1[i].b, descriptors1[i].c), 1.0f);
		//rr2 = getRotatedRectFromCovarianceMatrix(cv::Point2f(descriptors2[i].x, descriptors2[i].y), (cv::Mat_<float>(2, 2) << descriptors2[i].a, descriptors2[i].b, descriptors2[i].b, descriptors2[i].c), 1.0f);
		//rr2.center.x = rr2.center.x + image1.cols;
		angle1 = - rr1.angle / 180 * static_cast<float>(CV_PI);
		//angle2 = - rr2.angle / 180 * static_cast<float>(CV_PI);

		//cv::line(drawing, rr1.center, rr2.center, color1, 1, CV_AA, 0);
		cv::ellipse(drawing, rr1, color2, 1, CV_AA);
		cv::line(drawing, rr1.center, cv::Point2f(rr1.center.x + cos(angle1) * rr1.size.width * 0.5f, rr1.center.y - sin(angle1) * rr1.size.width * 0.5f), color2, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr1.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color2);
		//cv::ellipse(drawing, rr2, color2, 1, CV_AA);
		//cv::line(drawing, rr2.center, cv::Point2f(rr2.center.x + cos(angle2) * rr2.size.width * 0.5f, rr2.center.y - sin(angle2) * rr2.size.width * 0.5f), color2, 1, CV_AA, 0);
		//cv::putText(drawing, std::to_string(static_cast<long long>(i)), rr2.center, CV_FONT_HERSHEY_COMPLEX, 0.5, color2);
	}

	cv::imshow(title, drawing);
	cv::waitKey(delay);
}
std::vector<Descriptor> rangerCheck(std::vector<Descriptor> descriptors)
{
	Speicher Speicher;
	std::vector<Descriptor> rangerPredicted;
	std::vector<Descriptor> rangerMissMatch;
	std::vector<std::string>rangerSetUp;
	//construct ranger input
	for (uint j = 0; j < descriptors.size(); ++j)
		rangerSetUp.push_back(std::to_string(descriptors.at(j).data.at(0)));
	//header
	std::vector<std::string>::iterator it2;
	it2 = rangerSetUp.begin();
	it2 = rangerSetUp.insert(it2, "SIFT0");
	for (uint c = 1; c < 128; ++c)
		rangerSetUp.at(0) = rangerSetUp.at(0) + " " + "SIFT" + std::to_string(c);
	//save to data.dat
	Speicher.WriteText(rangerSetUp, "data.dat", "J:\\VC\\Ranger\\");
	//run Ranger 
	WinExec("J:\\VC\\Ranger\\Ranger.exe", SW_SHOWNORMAL);
	rangerSetUp.clear();
	//fill rangerSetUp with "ranger_out.prediction"
	rangerSetUp = Speicher.ReadText("J:\\VC\\Ranger\\", "ranger_out.prediction");
	//find "1 " in rangerSetUp and set flag

	for (uint k = 0; k < rangerSetUp.size(); ++k)
		//rangerPredicted hat Einträge von allen zu descriptor.at(i) passenden
		if (rangerSetUp.at(k) == "1 ")
			rangerPredicted.push_back(descriptors.at(k));
		else
			rangerMissMatch.push_back(descriptors.at(k));
	descriptors = rangerPredicted;
	return rangerMissMatch;
}
int main(int argc, char** argv)
{
	if(argc != 4)
	{
		std::cout << "Usage: " << argv[0] << " <image 1> <image 2> <config file>" << std::endl;
		return EXIT_FAILURE;
	}

	// Load images
	cv::Mat image1_gray, image1_color;
	//boost::filesystem::path path1(argv[1]);  ----------------------------------------------------------------
	Speicher Speicher;
	Speicher.SetFolder("C:\\VC\\");
	loadImage("basel_000062_mv0.jpg", image1_gray, image1_color);
	//boost::filesystem::path path2(argv[2]); 
	//loadImage("C:\\VC", image2_gray, image2_color);

	if(!image1_gray.data )
	{
		std::cout << "Could not open or find one of the images" << std::endl;
		return EXIT_FAILURE;
	}
	/*
	// Read config file
	libconfig::Config cfg;
	try
	{
		cfg.readFile(argv[3]);
	}
	catch(const libconfig::FileIOException &fioex)
	{
		std::cerr << "I/O error while reading file." << std::endl;
		return EXIT_FAILURE;
	}
	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
		return EXIT_FAILURE;
	}
	*/
	// Set MSER and Hessian-Affine parameters
	//MSERParams params_mser(2, 30, 0.01f, 0.25f, 0.2f);
	//HessianAffineParams params_ha(0, 3, 16.0f / 3.0f, 10.0f, 0.01f);
	//SIFTParams params_sift(0, - 1, 3, 20, 3.0f * sqrt(3.0f), 1.0f);
	MSERParams params_mser(
		static_cast<int>(2),//cfg.lookup("MSERParams.delta")),
		static_cast<int>(30),//cfg.lookup("MSERParams.minArea")),
		static_cast<float>(0.01),//cfg.lookup("MSERParams.maxArea")),
		static_cast<float>(0.25),//cfg.lookup("MSERParams.maxVariation")),
		static_cast<float>(0.2)//cfg.lookup("MSERParams.minDiversity"))
		);
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
	int imageMargin = static_cast<int>(0);//cfg.lookup("imageMargin"));
	bool useRootSIFT = static_cast<bool>(false);//cfg.lookup("useRootSIFT"));
	int distanceRatioTest = static_cast<int>(1);//cfg.lookup("distanceRatioTest"));
	float minimumGeometricDistance = static_cast<float>(10);//cfg.lookup("minimumGeometricDistance"));
	float maximumDistanceRatio = static_cast<float>(0.8);//cfg.lookup("maximumDistanceRatio"));
	float maximumDescriptorDistance = static_cast<float>(200);//cfg.lookup("maximumDescriptorDistance"));
	int minimumNumberMatches = static_cast<int>(50);// cfg.lookup("minimumNumberMatches"));
	bool filterMatchesWithEpipolarGeometry = static_cast<bool>(true);//cfg.lookup("filterMatchesWithEpipolarGeometry"));
	int displayDelay = static_cast<int>(1);// cfg.lookup("displayDelay"));

	for(int detector_iterator = 0; detector_iterator < 2; ++detector_iterator)
	{
		// Detect and describe features
		std::vector<Descriptor> descriptors1, descriptors2;
		if(detector_iterator == 0)
		{
			detectMSER(image1_gray, descriptors1, params_mser, params_sift);
			//detectMSER(image2_gray, descriptors2, params_mser, params_sift);

			std::cout << "Found " << descriptors1.size() << " affine regions (MSER) in image 1" << std::endl;
			std::cout << "Found " << descriptors2.size() << " affine regions (MSER) in image 2" << std::endl;
		}
		if(detector_iterator == 1)
		{
			detectHessianAffine(image1_gray, descriptors1, params_ha, params_sift);
			describeSIFT(image1_gray, descriptors1, params_sift);
			//detectHessianAffine(image2_gray, descriptors2, params_ha, params_sift);
			//describeSIFT(image2_gray, descriptors2, params_sift);

			std::cout << "Found " << descriptors1.size() << " affine regions (Hessian-Affine) in image 1" << std::endl;
			std::cout << "Found " << descriptors2.size() << " affine regions (Hessian-Affine) in image 2" << std::endl;
		}

		// Remove improper features
		for(int i = 0; i < descriptors1.size(); )
		{
			if(isOutsideImageBoundary(image1_gray, descriptors1[i], imageMargin) || /*isTooElongated(descriptors1[i]) || */hasNonPositiveDeterminant(descriptors1[i]))
			{
				descriptors1.erase(descriptors1.begin() + i);
			}
			else
			{
				++i;
			}
		}
		/*
		for(int i = 0; i < descriptors2.size(); )
		{
			if(isOutsideImageBoundary(image2_gray, descriptors2[i], imageMargin) || hasNonPositiveDeterminant(descriptors2[i])) //isTooElongated(descriptors2[i]) || 
			{
				descriptors2.erase(descriptors2.begin() + i);
			}
			else
			{
				++i;
			}
		}
	*/
		// Display matches
		drawMatches("Display matches", displayDelay, image1_color, descriptors1, descriptors2);

		cv::Mat desc1, desc2;
		collapseDescriptors(descriptors1, descriptors2, desc1, desc2);

		// Transform descriptors from SIFT to RootSIFT
		if(useRootSIFT)
		{
			transformSIFTToRootSIFT(desc1);
			transformSIFTToRootSIFT(desc2);
		}


		/* RANGER _______________________________________-----------------______________________________*/
		std::vector<Descriptor>descriptorsGood = descriptors1;
		//use Ranger to predict
		std::vector<Descriptor>descriptorsBad = rangerCheck(descriptorsGood);
		//draw Matches
		drawMatches("Display matched", displayDelay, image1_color, descriptorsGood, descriptorsBad);
		cv::imwrite("image1.jpg",image1_color);
		/*
		// Matching
		cv::flann::Index tree1 = cv::flann::Index(desc1, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);
		cv::flann::Index tree2 = cv::flann::Index(desc2, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);

		cv::Mat indices12, indices21;
		cv::Mat dists12, dists21;

		cv::flann::SearchParams SearchParams(128);
		tree1.knnSearch(desc2, indices21, dists21, 48, SearchParams);
		tree2.knnSearch(desc1, indices12, dists12, 48, SearchParams);

		// Do distance ratio test
		std::vector<cv::DMatch> matches12, matches21;
		if(distanceRatioTest)
		{
			matches12 = doFGINNDistanceRatioTest(indices12, dists12, descriptors1, descriptors2, minimumGeometricDistance, maximumDistanceRatio, maximumDescriptorDistance);
			matches21 = doFGINNDistanceRatioTest(indices21, dists21, descriptors2, descriptors1, minimumGeometricDistance, maximumDistanceRatio, maximumDescriptorDistance);
		}
		else
		{
			matches12 = doSNNDistanceRatioTest(indices12, dists12, maximumDistanceRatio, maximumDescriptorDistance);
			matches21 = doSNNDistanceRatioTest(indices21, dists21, maximumDistanceRatio, maximumDescriptorDistance);
		}

		// Merge matches
		std::vector<cv::DMatch> matchesMerged;
		matchesMerged = mergeMatches(matches12, matches21);
		if(matchesMerged.size() < minimumNumberMatches)
		{
			std::cout << "Error: Not enough matches were found" << std::endl;
			return EXIT_FAILURE;
		}

		// Filter matches using epipolar geometry
		std::vector<cv::DMatch> finalMatches;
		if(filterMatchesWithEpipolarGeometry)
		{
			cv::Mat F;
			filterMatchesUsingEpipolarGeometry(descriptors1, descriptors2, matchesMerged, finalMatches, F);
		}
		else
		{
			finalMatches = matchesMerged;
		}

		// Separate descriptors in positive and negative ones
		std::vector<Descriptor> desc_pos, desc_neg, desc_pos_1, desc_pos_2, desc_neg_1, desc_neg_2;
		for(int i = 0; i < finalMatches.size(); ++i)
		{
			desc_pos_1.push_back(descriptors1[finalMatches[i].queryIdx]);
			desc_pos_2.push_back(descriptors2[finalMatches[i].trainIdx]);
		}
		sortMatchesQueryIdx(finalMatches);
		for(int i = finalMatches.size() - 1; i > - 1; --i)
		{
			descriptors1.erase(descriptors1.begin() + finalMatches[i].queryIdx);
			//descriptors2.erase(descriptors2.begin() + finalMatches[i].trainIdx);
		}
		sortMatchesTrainIdx(finalMatches);
		for(int i = finalMatches.size() - 1; i > - 1; --i)
		{
			//descriptors1.erase(descriptors1.begin() + finalMatches[i].queryIdx);
			descriptors2.erase(descriptors2.begin() + finalMatches[i].trainIdx);
		}
		desc_pos = desc_pos_1;
		desc_pos.insert(desc_pos.end(), desc_pos_2.begin(), desc_pos_2.end());
		desc_neg = descriptors1;
		desc_neg.insert(desc_neg.end(), descriptors2.begin(), descriptors2.end());
		desc_neg_1 = descriptors1;
		desc_neg_2 = descriptors2;

		// Create filenames
		std::string detector;
		if(detector_iterator == 0)
		{
			detector = "MSER";
		}
		if(detector_iterator == 1)
		{
			detector = "Hessian-Affine";
		}
		std::string filename1 = path1.stem().string() + "_" + path2.stem().string() + "_" + detector + "_positive";
		std::string filename2 = path1.stem().string() + "_" + path2.stem().string() + "_" + detector + "_negative";

		// Write files
		std::ofstream file;
		file.open(filename1);
		for(int i = 0; i < desc_pos.size(); ++i)
		{
			for(int j = 0; j < 127; ++j)
			{
				file << desc_pos[i].data[j] << " ";
			}
			file << desc_pos[i].data[127] << std::endl;
		}
		file.close();
		file.open(filename2);
		for(int i = 0; i < desc_neg.size(); ++i)
		{
			for(int j = 0; j < 127; ++j)
			{
				file << desc_neg[i].data[j] << " ";
			}
			file << desc_neg[i].data[127] << std::endl;
		}
		file.close();

		std::cout << "Saved " << desc_pos.size() << " positive descriptors" << std::endl;
		std::cout << "Saved " << desc_neg.size() << " negative descriptors" << std::endl;

		// Display matches
		drawMatches("Display matches", displayDelay, image1_color, image2_color, desc_pos_1, desc_pos_2, desc_neg_1, desc_neg_2);
		*/
	}
}