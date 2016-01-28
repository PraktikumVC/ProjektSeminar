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

#ifndef MIPMAP_H
#define MIPMAP_H

#include <vector>

#include <stdint.h>

/// The Mipmap class stores a collection of reduced version of the input grayscale (8 bits) image,
/// and can be used to perform nearest neighbor, bilinear and trilinear sampling (with clamp to
/// edge).
class Mipmap
{
public:
    /// Constructor.
    /// @param[in]  bits    Pointer to the first scanline of the image.
    /// @param[in]  width   Width of the image.
    /// @param[in]  height  Height of the image.
    /// @note  The @p bits pointer must remain valid for the entire lifetime of the @c Mipmap.
    Mipmap(const uint8_t * bits,
           int width,
           int height);

    /// Returns the number of levels.
    int numberOfLevels() const;

    /// Returns a pointer to the first scanline of level @p l.
    const uint8_t * level(int l) const;

    /// Nearest neighbor sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    uint8_t operator()(int x,
                       int y,
                       int l = 0) const;

    /// Bilinear sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    double operator()(double x,
                      double y,
                      int l = 0) const;

    /// Trilinear sampling.
    /// @param[in]  x  Abscissa.
    /// @param[in]  y  Ordinate.
    /// @param[in]  l  Level (scale).
    double operator()(double x,
                      double y,
                      double l) const;

private:
    std::vector<const uint8_t *> levels_;
    std::vector<uint8_t> data_;
    int width_;
    int height_;
    int numberOfLevels_;
};

#endif
