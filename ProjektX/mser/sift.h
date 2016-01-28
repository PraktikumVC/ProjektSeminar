//--------------------------------------------------------------------------------------------------
// Linear time Maximally Stable Extremal Regions implementation as described in D. Nistér and H.
// Stewénius. Linear Time Maximally Stable Extremal Regions. Proceedings of the European Conference
// on Computer Vision (ECCV), 2008.
//
// Scale-Invariant Feature Transform implemetation as described in D. Lowe. Distinctive image
// features from scale-invariant keypoints. International journal of computer vision (IJCV), 2004.
//
// Copyright (c) 2011 Idiap Research Institute, http://www.idiap.ch/.
// Written by Charles Dubout <charles.dubout@idiap.ch>.
//
// MSER is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// MSER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with MSER. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#ifndef SIFT_H
#define SIFT_H

#include "mser.h"

/// The SIFT class computes the SIFT (Scale-Invariant Feature Transform) descriptor of the regions
/// extracted from a grayscale (8 bit) image.
class SIFT
{
public:
    /// A SIFT descriptor.
    struct Descriptor
    {
        double x;                   ///< Center.
        double y;                   ///< Center.
        double a;                   ///< Covariance matrix [a b; b c].
        double b;                   ///< Covariance matrix [a b; b c].
        double c;                   ///< Covariance matrix [a b; b c].
        double angle;               ///< Angle of the x-axis relative to the principal axis.
        //std::vector<uint8_t> data;  ///< Descriptor data.
        std::vector<double> data;  ///< Descriptor data.
    };

    /// Constructor.
    /// @param[in]  resolution  Resolution (in pixels) at which the regions shoud be extracted.
    ///                         Must be a multiple of 4 in the range [4, 256].
    /// @param[in]  radius      Number of standard deviations of the ellipses fit to the regions.
    SIFT(int resolution = 64,
         double radius = 3.0);

    /// Computes the SIFT descriptor of the regions extracted from a grayscale (8 bit) image.
    /// @param[in]  bits                  Pointer to the first scanline of the image.
    /// @param[in]  width                 Width of the image.
    /// @param[in]  height                Height of the image.
    /// @param[in]  regions               Detected MSERs.
    /// @param[in]  orientationInvariant  Whether to compute a descriptor for each dominant
    ///                                   orientation or only for the original one.
    /// @return  The descriptor associated to each region.
    std::vector<Descriptor> operator()(const uint8_t * bits,
                                       int width,
                                       int height,
                                       const std::vector<MSER::Region> & regions,
                                       bool orientationInvariant = true) const;

    /// Computes the SIFT descriptor of the regions extracted from a grayscale (8 bit) image.
    /// @param[in]  bits                  Pointer to the first scanline of the image.
    /// @param[in]  width                 Width of the image.
    /// @param[in]  height                Height of the image.
    /// @param[in]  regions               Detected MSERs.
    /// @param[in]  orientationInvariant  Whether to compute a descriptor for each dominant
    ///                                   orientation or only for the original one.
    /// @return  The descriptor associated to each region.
    std::vector<Descriptor> describe(uint8_t * bits,
                                       int width,
                                       int height,
                                       std::vector<std::vector<float>> & regions,
                                       bool orientationInvariant = true);

    // Implementation details (could be moved outside this header file)
private:
    // Parameters
    int resolution_;
    double radius_;

    // Lookup tables
    //double sqrtTable_[512][512];
    std::vector<std::vector<double>> sqrtTable_;
    //double atan2Table_[512][512];
    std::vector<std::vector<double>> atan2Table_;
    std::vector<double> siftTables_[4][4];
    std::vector<int> minMaxTables_[4];
};

#endif
