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

#ifndef MSER_H
#define MSER_H

#include <deque>
#include <vector>

#include <stdint.h>

/// The MSER class extracts maximally stable extremal regions from a grayscale (8 bit) image.
/// @note The MSER class is not reentrant, so if you want to extract regions in parallel, each
/// thread needs to have its own MSER class instance.
class MSER
{
public:
    /// A Maximally Stable Extremal Region.
    struct Region
    {
        int level;          ///< Level at which the region is processed.
        int pixel;          ///< Index of the initial pixel (y * width + x).
        int area;           ///< Area of the region (moment zero).
        double moments[5];  ///< First and second moments of the region (x, y, x^2, xy, y^2).
        double variation;   ///< MSER variation.

        /// Constructor.
        /// @param[in]  level  Level at which the region is being processed.
        /// @param[in]  pixel  Index of the initial pixel (y * width + x).
        Region(int level = 256,
               int pixel = -1);

        // Implementation details (could be moved outside of this header file)
    private:
        bool stable_;      // Flag indicating if the region is stable
        Region * parent_;  // Pointer to the parent region
        Region * child_;   // Pointer to the first child
        Region * next_;    // Pointer to the next (sister) region

        void accumulate(double x, double y);
        void merge(Region * child);
        void process(int delta, int minArea, int maxArea, double maxVariation, double minDiversity);
        void save(std::vector<Region> & regions) const;

        friend class MSER;
    };

    /// Constructor.
    /// @param[in]  delta         DELTA parameter of the MSER algorithm. Roughly speaking, the
    ///                           stability of a region is the relative variation of the region
    ///                           area when the intensity is changed by delta.
    /// @param[in]  minArea       Minimum area of any stable region in pixels.
    /// @param[in]  maxArea       Maximum area of any stable region in pixels.
    /// @param[in]  maxVariation  Maximum variation (absolute stability score) of the regions.
    /// @param[in]  minDiversity  Minimum diversity of the regions. When the relative area of two
    ///                           nested regions is above this threshold, then only the most stable
    ///                           one is selected.
    /// @param[in]  eight         Use 8-connected pixels instead of 4-connected.
    MSER(int delta = 1,
         int minArea = 100,
         int maxArea = 10000,
         double maxVariation = 0.25,
         double minDiversity = 0.5,
         bool eight = false);

    /// Extracts maximally stable extremal regions from a grayscale (8 bit) image.
    /// @param[in]  bits    Pointer to the first scanline of the image.
    /// @param[in]  width   Width of the image.
    /// @param[in]  height  Height of the image.
    /// @return  Detected MSERs.
    std::vector<Region> operator()(const uint8_t * bits,
                                   int width,
                                   int height) const;

    // Implementation details (could be moved outside of this header file)
private:
    // Helper method
    void processStack(int newPixelGreyLevel, int pixel, std::vector<Region *> & regionStack) const;

    // Parameters
    int delta_;
    int minArea_;
    int maxArea_;
    double maxVariation_;
    double minDiversity_;
    bool eight_;

    // Memory pool of regions for faster allocation
    mutable std::deque<Region> pool_;
    mutable std::size_t poolIndex_;
};

#endif
