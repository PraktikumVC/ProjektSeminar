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

#include "affine.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "mipmap.h"

using namespace std;

Affine::Affine(int resolution, double radius)
: resolution_(resolution), radius_(radius)
{
    assert(resolution > 0);
    assert(radius > 0.0);
}

vector<uint8_t> Affine::operator()(const uint8_t * bits, int width, int height,
                                   const vector<Region> & regions)
{
    if (!bits || (width <= 0) || (height <= 0) || regions.empty())
        return vector<uint8_t>();

    const Mipmap mipmap(bits, width, height);

    // Create the output image
    vector<uint8_t> image(regions.size() * resolution_ * resolution_);

    for (size_t i = 0; i < regions.size(); ++i) {
        // Square root of the covariance matrix
        const double tr = regions[i].a + regions[i].c;
        const double sqrtDet = sqrt(regions[i].a * regions[i].c - regions[i].b * regions[i].b);
        const double alpha = 2.0 * radius_ / resolution_ / sqrt(tr + 2.0 * sqrtDet);
        const double c = cos(regions[i].angle) * alpha;
        const double s = sin(regions[i].angle) * alpha;

        double affine[2][2] = {
            { c * (regions[i].a + sqrtDet) + s * regions[i].b,
             -s * (regions[i].a + sqrtDet) + c * regions[i].b},
            { c * regions[i].b + s * (regions[i].c + sqrtDet),
             -s * regions[i].b + c * (regions[i].c + sqrtDet)}
        };

        //const double scale = 0.5 * log2(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]);
        const double scale = 0.5 * (log(affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]) / log(2.0));

        // Recopy the buffer into the output image
        for (int v = 0; v < resolution_; ++v) {
            for (int u = 0; u < resolution_; ++u) {
                const double u2 = u - 0.5 * (resolution_ - 1);
                const double v2 = v - 0.5 * (resolution_ - 1);

                image[(i * resolution_ + v) * resolution_ + u] =
                        mipmap(affine[0][0] * u2 + affine[0][1] * v2 + regions[i].x,
                               affine[1][0] * u2 + affine[1][1] * v2 + regions[i].y, scale);
            }
        }
    }

    return image;
}
