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

#ifndef AFFINE_H
#define AFFINE_H

#include <vector>

#include <stdint.h>

/// The Affine class extracts multiple affine regions from a grayscale (8 bit) image.
/// @note The Affine class is not reentrant, so if you want to extract regions in parallel, each
/// thread needs to have its own Affine class instance.
class Affine
{
public:
    /// An affine region.
    struct Region
    {
        double x;      ///< Center.
        double y;      ///< Center.
        double a;      ///< Covariance matrix [a b; b c].
        double b;      ///< Covariance matrix [a b; b c].
        double c;      ///< Covariance matrix [a b; b c].
        double angle;  ///< Angle of the x-axis relative to the principal axis.
    };

    /// Constructor.
    /// @param[in]  resolution  Resolution (in pixels) at which the regions shoud be extracted.
    ///                         Must be a multiple of 4 in the range [4, 256].
    /// @param[in]  radius      Number of standard deviations of the ellipses fit to the regions.
    Affine(int resolution = 64,
           double radius = 3.0);

    /// @brief Extracts multiple affine regions from a grayscale (8 bit) image.
    /// @param[in]  bits     Pointer to the first scanline of the image.
    /// @param[in]  width    Width of the image.
    /// @param[in]  height   Height of the image.
    /// @param[in]  regions  Affine regions to extract from the image.
    /// @return  Image of the concatenated affine regions of dimensions
    ///          resolution x (regions.size() * resolution).
    std::vector<uint8_t> operator()(const uint8_t * bits,
                                    int width,
                                    int height,
                                    const std::vector<Region> & regions);

    // Implementation details (could be moved outside of this header file)
private:
    // Parameters
    const int resolution_;
    const double radius_;
};

#endif
