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

#include "mipmap.h"

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace std;

Mipmap::Mipmap(const uint8_t * bits, int width, int height)
: width_(width), height_(height), numberOfLevels_(0)
{
    if (!bits || (width <= 0) || (height <= 0))
        return;

    // Compute the number of levels and the memory required to store them
    size_t size = 0;

    while (width && height) {
        ++numberOfLevels_;
        width >>= 1;
        height >>= 1;
        size += width * height;
    }

    // The first level is the original image
    levels_.push_back(bits);

    // The additional storage size required
    data_.resize(size);

    // Fill every level
    width = width_;
    height = height_;
    const uint8_t * src = bits;
    uint8_t * dst = &data_[0];
    vector<uint8_t> tmp((width >> 1) * height);

    for (int l = 1; l < numberOfLevels_; ++l) {
        // Index the level in the data
        levels_.push_back(dst);

        // Blur src's rows into tmp (sigma = 1)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < (width >> 1); ++x) {
                const int x0 = min(max(x * 2 - 2, 0), width - 1);
                const int x1 = min(max(x * 2 - 1, 0), width - 1);
                const int x2 = min(max(x * 2    , 0), width - 1);
                const int x3 = min(max(x * 2 + 1, 0), width - 1);
                const int x4 = min(max(x * 2 + 2, 0), width - 1);
                const int x5 = min(max(x * 2 + 3, 0), width - 1);
                const int a = src[y * width + x0] + src[y * width + x5];
                const int b = src[y * width + x1] + src[y * width + x4];
                const int c = src[y * width + x2] + src[y * width + x3];

                tmp[x * height + y] = (1151 * a + 8503 * b + 23114 * c + 32768) >> 16;
            }
        }

        width >>= 1;

        // Blur tmp's columns into dst (sigma = 1)
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < (height >> 1); ++y) {
                const int y0 = min(max(y * 2 - 2, 0), height - 1);
                const int y1 = min(max(y * 2 - 1, 0), height - 1);
                const int y2 = min(max(y * 2    , 0), height - 1);
                const int y3 = min(max(y * 2 + 1, 0), height - 1);
                const int y4 = min(max(y * 2 + 2, 0), height - 1);
                const int y5 = min(max(y * 2 + 3, 0), height - 1);
                const int a = tmp[x * height + y0] + tmp[x * height + y5];
                const int b = tmp[x * height + y1] + tmp[x * height + y4];
                const int c = tmp[x * height + y2] + tmp[x * height + y3];

                dst[y * width + x] = (1151 * a + 8503 * b + 23114 * c + 32768) >> 16;
            }
        }

        height >>= 1;

        // The source of the next level is the current level
        src = levels_.back();
        dst += width * height;
    }
}

int Mipmap::numberOfLevels() const
{
    return numberOfLevels_;
}

const uint8_t * Mipmap::level(int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    return levels_[l];
}

uint8_t Mipmap::operator()(int x, int y, int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    // Clamp the coordinates
    x = min(max(x, 0), width_ - 1);
    y = min(max(y, 0), height_ - 1);

    // Convert to level l
    return levels_[l][(y >> l) * (width_ >> l) + (x >> l)];
}

double Mipmap::operator()(double x, double y, int l) const
{
    // Clamp the level
    l = min(max(l, 0), numberOfLevels_ - 1);

    x = ldexp(x, -l);
    y = ldexp(y, -l);

    // Clamp the coordinates
    const int width = width_ >> l;
    const int height = height_ >> l;

    x = min(max(x, 0.0), width - 1.0);
    y = min(max(y, 0.0), height - 1.0);

    // Bilinear interpolation
    const int x0 = x;
    const int x1 = min(x0 + 1, width - 1);
    const int y0 = y;
    const int y1 = min(y0 + 1, height - 1);
    const double a = x - x0;
    const double b = 1.0 - a;
    const double c = y - y0;
    const double d = 1.0 - c;

    return (levels_[l][y0 * width + x0] * b + levels_[l][y0 * width + x1] * a) * d +
           (levels_[l][y1 * width + x0] * b + levels_[l][y1 * width + x1] * a) * c;
}

double Mipmap::operator()(double x, double y, double l) const
{
    // Clamp the level
    if (l <= 0.0)
        return operator()(x, y, 0);
    else if (l >= numberOfLevels_ - 1.0)
        return operator()(x, y, numberOfLevels_ - 1);

    // Interpolation of the two closest levels
    const int l0 = l;
    const int l1 = l0 + 1;
    const double a = l - l0;
    const double b = 1.0 - a;

    return operator()(x, y, l0) * b + operator()(x, y, l1) * a;
}
