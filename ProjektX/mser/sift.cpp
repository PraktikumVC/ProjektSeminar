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

#define _USE_MATH_DEFINES

#include "sift.h"

#include <algorithm>
#include <cmath>

#include "affine.h"

using namespace std;

SIFT::SIFT(int resolution,
           double radius)
: resolution_(resolution), radius_(radius)
{
    sqrtTable_.resize(512);
    for(int i = 0; i < 512; ++i) {
        sqrtTable_[i].resize(512);
    }
    atan2Table_.resize(512);
    for(int i = 0; i < 512; ++i) {
        atan2Table_[i].resize(512);
    }

    // Fill the sqrt and atan2 tables
    for (int i = -255; i <= 255; ++i) {
        for (int j = -255; j <= 255; ++j) {
            sqrtTable_[i + 255][j + 255] = (double)sqrt((double)(i * i + j * j));

            double a = 0.5 * ((double)atan2((double)i, (double)j) / M_PI + 1.0);

            if (a == 1.0)
                a = 0.0;

            atan2Table_[i + 255][j + 255] = a;
        }
    }

    // Fill the SIFT interpolation tables
    minMaxTables_[0].resize(resolution * resolution, 3);
    minMaxTables_[1].resize(resolution * resolution, 0);
    minMaxTables_[2].resize(resolution * resolution, 3);
    minMaxTables_[3].resize(resolution * resolution, 0);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            siftTables_[i][j].resize(resolution * resolution);

            for (int v = 0; v < resolution; ++v) {
                for (int u = 0; u < resolution; ++u) {
                    const int k = v * resolution + u;

                    siftTables_[i][j][k] =
                            max(1.0 - abs(4.0 * (u + 0.5) / resolution - j - 0.5), 0.0) *
                            max(1.0 - abs(4.0 * (v + 0.5) / resolution - i - 0.5), 0.0);

                    if (siftTables_[i][j][k] > 0.0) {
                        minMaxTables_[0][k] = min(minMaxTables_[0][k], j);
                        minMaxTables_[1][k] = max(minMaxTables_[1][k], j);
                        minMaxTables_[2][k] = min(minMaxTables_[2][k], i);
                        minMaxTables_[3][k] = max(minMaxTables_[3][k], i);
                    }
                }
            }
        }
    }
}

// Normalize a descriptor by dividing it by its norm, clamping it to 0.2, dividing it by its norm
// again, and finally clamping it to 0.3 before converting it to the range [0, 255]
static double normalize(double * x,
                        int n)
{
    double sumSquared = 0.0;

    for (int i = 0; i < n; ++i)
        sumSquared += x[i] * x[i];

    const double norm = sqrt(sumSquared);
    double invNorm = 1.0 / norm;

    sumSquared = 0.0;

    for (int i = 0; i < n; ++i) {
        x[i] = min(x[i] * invNorm, 0.2);
        sumSquared += x[i] * x[i];
    }

    invNorm = 1.0 / sqrt(sumSquared);

    for (int i = 0; i < n; ++i)
        x[i] = min((256.0 / 0.3) * x[i] * invNorm, 255.0);

    return norm;
}

vector<SIFT::Descriptor> SIFT::operator()(const uint8_t * bits,
                                          int width,
                                          int height,
                                          const std::vector<MSER::Region> & mserRegions,
                                          bool orientationInvariant) const
{
    if (!bits || (width <= 0) || (height <= 0) || mserRegions.empty())
        return vector<Descriptor>();

    // Convert the MSER regions to Affine regions
    vector<Affine::Region> regions;

    for (size_t i = 0; i < mserRegions.size(); ++i) {
        const double x = mserRegions[i].moments[0] / mserRegions[i].area;
        const double y = mserRegions[i].moments[1] / mserRegions[i].area;
        const double a = (mserRegions[i].moments[2] - x * mserRegions[i].moments[0]) /
                         (mserRegions[i].area - 1);
        const double b = (mserRegions[i].moments[3] - x * mserRegions[i].moments[1]) /
                         (mserRegions[i].area - 1);
        const double c = (mserRegions[i].moments[4] - y * mserRegions[i].moments[1]) /
                         (mserRegions[i].area - 1);

        //// Skip too elongated regions
        //const double d = sqrt((a - c) * (a - c) + 4.0 * b * b);

        //if ((a + c + d) / (a + c - d) < 25.0) {
            const Affine::Region region = {x, y, a, b, c, 0.0};

            regions.push_back(region);
        //}
    }

    // Extract the regions from the image
    Affine affine(resolution_, radius_);

    vector<uint8_t> image = affine(bits, width, height, regions);

    if (image.empty())
        return vector<Descriptor>();

    // Compute the dominant orientations if needed
    if (orientationInvariant) {
        vector<Affine::Region> newRegions;

        for (size_t i = 0; i < regions.size(); ++i) {
            const uint8_t * pixels = &image[i * resolution_ * resolution_];

            // Compute the gradient of the bounding box
            double hist[36] = {};

            for (int v = 0; v < resolution_; ++v) {
                for (int u = 0; u < resolution_; ++u) {
                    const double r2 =
                            (u - resolution_ / 2.0 + 0.5) * (u - resolution_ / 2.0 + 0.5) +
                            (v - resolution_ / 2.0 + 0.5) * (v - resolution_ / 2.0 + 0.5);

                    if (r2 < resolution_ * resolution_ / 4.0) {
                        const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                                       pixels[v * resolution_ + max(u - 1, 0)];
                        const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                                       pixels[max(v - 1, 0) * resolution_ + u];
                        const double n = sqrtTable_[dv + 255][du + 255];
                        const double a = 36.0 * atan2Table_[dv + 255][du + 255];
                        const int a0 = a;
                        const int a1 = (a0 + 1) % 36;
                        const double z = exp(-r2 / (resolution_ * resolution_ / 4.0));

                        hist[a0] += (1.0 - (a - a0)) * (n * z);
                        hist[a1] +=        (a - a0)  * (n * z);
                    }
                }
            }

            // Blur the histogram
            for (int j = 0; j < 3; ++j) {
                double tmp[36];

                for (int k = 0; k < 36; ++k)
                    tmp[k] = 0.054489 * (hist[(k + 34) % 36] + hist[(k + 2) % 36]) +
                             0.244201 * (hist[(k + 35) % 36] + hist[(k + 1) % 36]) +
                             0.402620 * hist[k];

                for (int k = 0; k < 36; ++k)
                    hist[k] = 0.054489 * (tmp[(k + 34) % 36] + tmp[(k + 2) % 36]) +
                              0.244201 * (tmp[(k + 35) % 36] + tmp[(k + 1) % 36]) +
                              0.402620 * tmp[k];
            }

            // Add a descriptor for each local maximum greater than 80% of the global maximum
            const double maxh = *max_element(hist, hist + 36);

            for (int j = 0; j < 36; ++j) {
                const double h0 = hist[j];
                const double hm = hist[(j + 35) % 36];
                const double hp = hist[(j +  1) % 36];

                if ((h0 > 0.8 * maxh) && (h0 > hm) && (h0 > hp)) {
                    Affine::Region region(regions[i]);

                    region.angle = (j - 0.5 * (hp - hm) / (hp + hm - 2.0 * h0)) * M_PI / 18.0;

                    newRegions.push_back(region);
                }
            }
        }

        regions.swap(newRegions);

        image = affine(bits, width, height, regions);

        if (image.empty())
            return vector<Descriptor>();
    }

    vector<Descriptor> descriptors;

    for (size_t i = 0; i < regions.size(); ++i) {
        const uint8_t * pixels = &image[i * resolution_ * resolution_];

        // Compute the SIFT descriptor
        double desc[128] = {};

        for (int v = 0; v < resolution_; ++v) {
            for (int u = 0; u < resolution_; ++u) {
                const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                               pixels[v * resolution_ + max(u - 1, 0)];
                const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                               pixels[max(v - 1, 0) * resolution_ + u];
                const double n = sqrtTable_[dv + 255][du + 255];
                const double a = 8.0 * atan2Table_[dv + 255][du + 255];
                const int a0 = a;
                const int a1 = (a0 + 1) % 8;
                const int l = v * resolution_ + u;

                for (int j = minMaxTables_[2][l]; j <= minMaxTables_[3][l]; ++j) {
                    for (int k = minMaxTables_[0][l]; k <= minMaxTables_[1][l]; ++k) {
                        desc[(j * 4 + k) * 8 + a0] += siftTables_[j][k][l] * (1.0 - (a - a0)) * n;
                        desc[(j * 4 + k) * 8 + a1] += siftTables_[j][k][l] *        (a - a0)  * n;
                    }
                }
            }
        }

        // Normalize the descriptor and skip regions of too low gradient
        if (normalize(desc, 128) < 0.5 * resolution_ * resolution_)
            continue;

        const Descriptor descriptor = {regions[i].x, regions[i].y, regions[i].a, regions[i].b,
                                       regions[i].c, regions[i].angle,
                                       vector<double>(desc, desc + 128)};

        descriptors.push_back(descriptor);
    }

    return descriptors;
}

vector<SIFT::Descriptor> SIFT::describe(uint8_t * bits,
                                          int width,
                                          int height,
                                          std::vector<std::vector<float>> & mserRegions,
                                          bool orientationInvariant)
{
    if (!bits || (width <= 0) || (height <= 0) || mserRegions.empty())
        return vector<Descriptor>();

    // Convert the MSER regions to Affine regions
    vector<Affine::Region> regions;

    for (size_t i = 0; i < mserRegions.size(); ++i) {
        const double x = mserRegions[i][0];
        const double y = mserRegions[i][1];
        const double a = mserRegions[i][2];
        const double b = mserRegions[i][3];
        const double c = mserRegions[i][4];

        //// Skip too elongated regions
        //const double d = sqrt((a - c) * (a - c) + 4.0 * b * b);

        //if ((a + c + d) / (a + c - d) < 25.0) {
            const Affine::Region region = {x, y, a, b, c, 0.0};

            regions.push_back(region);
        //}
    }

    // Extract the regions from the image
    Affine affine(resolution_, radius_);

    vector<uint8_t> image = affine(bits, width, height, regions);

    if (image.empty())
        return vector<Descriptor>();

    // Compute the dominant orientations if needed
    if (orientationInvariant) {
        vector<Affine::Region> newRegions;

        for (size_t i = 0; i < regions.size(); ++i) {
            const uint8_t * pixels = &image[i * resolution_ * resolution_];

            // Compute the gradient of the bounding box
            double hist[36] = {};

            for (int v = 0; v < resolution_; ++v) {
                for (int u = 0; u < resolution_; ++u) {
                    const double r2 =
                            (u - resolution_ / 2.0 + 0.5) * (u - resolution_ / 2.0 + 0.5) +
                            (v - resolution_ / 2.0 + 0.5) * (v - resolution_ / 2.0 + 0.5);

                    if (r2 < resolution_ * resolution_ / 4.0) {
                        const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                                       pixels[v * resolution_ + max(u - 1, 0)];
                        const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                                       pixels[max(v - 1, 0) * resolution_ + u];
                        const double n = sqrtTable_[dv + 255][du + 255];
                        const double a = 36.0 * atan2Table_[dv + 255][du + 255];
                        const int a0 = a;
                        const int a1 = (a0 + 1) % 36;
                        const double z = exp(-r2 / (resolution_ * resolution_ / 4.0));

                        hist[a0] += (1.0 - (a - a0)) * (n * z);
                        hist[a1] +=        (a - a0)  * (n * z);
                    }
                }
            }

            // Blur the histogram
            for (int j = 0; j < 3; ++j) {
                double tmp[36];

                for (int k = 0; k < 36; ++k)
                    tmp[k] = 0.054489 * (hist[(k + 34) % 36] + hist[(k + 2) % 36]) +
                             0.244201 * (hist[(k + 35) % 36] + hist[(k + 1) % 36]) +
                             0.402620 * hist[k];

                for (int k = 0; k < 36; ++k)
                    hist[k] = 0.054489 * (tmp[(k + 34) % 36] + tmp[(k + 2) % 36]) +
                              0.244201 * (tmp[(k + 35) % 36] + tmp[(k + 1) % 36]) +
                              0.402620 * tmp[k];
            }

            // Add a descriptor for each local maximum greater than 80% of the global maximum
            const double maxh = *max_element(hist, hist + 36);

            for (int j = 0; j < 36; ++j) {
                const double h0 = hist[j];
                const double hm = hist[(j + 35) % 36];
                const double hp = hist[(j +  1) % 36];

                if ((h0 > 0.8 * maxh) && (h0 > hm) && (h0 > hp)) {
                    Affine::Region region(regions[i]);

                    region.angle = (j - 0.5 * (hp - hm) / (hp + hm - 2.0 * h0)) * M_PI / 18.0;

                    newRegions.push_back(region);
                }
            }
        }

        regions.swap(newRegions);

        image = affine(bits, width, height, regions);

        if (image.empty())
            return vector<Descriptor>();
    }

    vector<Descriptor> descriptors;

    for (size_t i = 0; i < regions.size(); ++i) {
        const uint8_t * pixels = &image[i * resolution_ * resolution_];

        // Compute the SIFT descriptor
        double desc[128] = {};

        for (int v = 0; v < resolution_; ++v) {
            for (int u = 0; u < resolution_; ++u) {
                const int du = pixels[v * resolution_ + min(u + 1, resolution_ - 1)] -
                               pixels[v * resolution_ + max(u - 1, 0)];
                const int dv = pixels[min(v + 1, resolution_ - 1) * resolution_ + u] -
                               pixels[max(v - 1, 0) * resolution_ + u];
                const double n = sqrtTable_[dv + 255][du + 255];
                const double a = 8.0 * atan2Table_[dv + 255][du + 255];
                const int a0 = a;
                const int a1 = (a0 + 1) % 8;
                const int l = v * resolution_ + u;

                for (int j = minMaxTables_[2][l]; j <= minMaxTables_[3][l]; ++j) {
                    for (int k = minMaxTables_[0][l]; k <= minMaxTables_[1][l]; ++k) {
                        desc[(j * 4 + k) * 8 + a0] += siftTables_[j][k][l] * (1.0 - (a - a0)) * n;
                        desc[(j * 4 + k) * 8 + a1] += siftTables_[j][k][l] *        (a - a0)  * n;
                    }
                }
            }
        }

        // Normalize the descriptor and skip regions of too low gradient
        if (normalize(desc, 128) < 0.5 * resolution_ * resolution_)
            continue;

        const Descriptor descriptor = {regions[i].x, regions[i].y, regions[i].a, regions[i].b,
                                       regions[i].c, regions[i].angle,
                                       vector<double>(desc, desc + 128)};

        descriptors.push_back(descriptor);
    }

    return descriptors;
}
