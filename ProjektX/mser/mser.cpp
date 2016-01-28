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

#include "mser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

using namespace std;

MSER::Region::Region(int level, int pixel)
: level(level), pixel(pixel), area(0), variation(numeric_limits<double>::infinity()),
  stable_(false), parent_(0), child_(0), next_(0)
{
    fill_n(moments, 5, 0.0);
}

void MSER::Region::accumulate(double x, double y)
{
    ++area;
    moments[0] += x;
    moments[1] += y;
    moments[2] += x * x;
    moments[3] += x * y;
    moments[4] += y * y;
}

void MSER::Region::merge(Region * child)
{
    assert(!child->parent_);
    assert(!child->next_);

    // Add the moments together
    area += child->area;
    moments[0] += child->moments[0];
    moments[1] += child->moments[1];
    moments[2] += child->moments[2];
    moments[3] += child->moments[3];
    moments[4] += child->moments[4];

    child->next_ = child_;
    child_ = child;
    child->parent_ = this;
}

void MSER::Region::process(int delta, int minArea, int maxArea, double maxVariation,
                           double minDiversity)
{
    // Find the last parent with level not higher than level + delta
    Region * parent = this;

    while (parent->parent_ && (parent->parent_->level <= level + delta))
        parent = parent->parent_;

    variation = static_cast<double>(parent->area - area) / area;
    stable_ = (area >= minArea) && (area <= maxArea) && (variation <= maxVariation);

    // Make sure the regions are diverse enough
    for (Region * p = parent_; p && (area > minDiversity * p->area); p = p->parent_) {
        if (p->variation <= variation)
            stable_ = false;

        if (variation < p->variation)
            p->stable_ = false;
    }

    // Process all the children
    for (Region * child = child_; child; child = child->next_)
        child->process(delta, minArea, maxArea, maxVariation, minDiversity);
}

void MSER::Region::save(vector<Region> & regions) const
{
    if (stable_)
        regions.push_back(*this);

    for (const Region * child = child_; child; child = child->next_)
        child->save(regions);
}

MSER::MSER(int delta, int minArea, int maxArea, double maxVariation, double minDiversity,
           bool eight)
: delta_(delta), minArea_(minArea), maxArea_(maxArea), maxVariation_(maxVariation),
  minDiversity_(minDiversity), eight_(eight), pool_(256), poolIndex_(0)
{
    // Parameter check
    assert(delta > 0);
    assert(minArea > 0);
    assert(maxArea > minArea);
    assert(maxVariation >= 0.0);
    assert(minDiversity >= 0.0);
}

vector<MSER::Region> MSER::operator()(const uint8_t * bits, int width, int height) const
{
    if (!bits || (width <= 0) || (height <= 0))
        return vector<Region>();

    // 1. Clear the accessible pixel mask, the heap of boundary pixels and the component stack. Push
    // a dummy-component onto the stack, with grey-level higher than any allowed in the image.
    vector<bool> accessible(width * height);
    vector<int> boundaryPixels[256];
    int priority = 256;
    vector<Region *> regionStack;

    regionStack.push_back(new (&pool_[poolIndex_++]) Region);

    // 2. Make the source pixel (with its first edge) the current pixel, mark it as accessible and
    // store the grey-level of it in the variable current level.
    int curPixel = 0;
    int curEdge = 0;
    int curLevel = bits[0];

    accessible[0] = true;

    // 3. Push an empty component with current level onto the component stack.
step_3:
    regionStack.push_back(new (&pool_[poolIndex_++]) Region(curLevel, curPixel));

    if (poolIndex_ == pool_.size())
        pool_.resize(pool_.size() + 256);

    // 4. Explore the remaining edges to the neighbors of the current pixel, in order, as follows:
    // For each neighbor, check if the neighbor is already accessible. If it is not, mark it as
    // accessible and retrieve its grey-level. If the grey-level is not lower than the current one,
    // push it onto the heap of boundary pixels. If on the other hand the grey-level is lower than
    // the current one, enter the current pixel back into the queue of boundary pixels for later
    // processing (with the next edge number), consider the new pixel and its grey-level and go to
    // 3.
    for (;;) {
        const int x = curPixel % width;
        const int y = curPixel / width;

        const int offsets[8][2] = {
            { 1, 0}, { 0, 1}, {-1, 0}, { 0,-1},
            { 1, 1}, {-1, 1}, {-1,-1}, { 1,-1}
        };

        for (; curEdge < (eight_ ? 8 : 4); ++curEdge) {
            const int nx = x + offsets[curEdge][0];
            const int ny = y + offsets[curEdge][1];

            if ((nx >= 0) && (ny >= 0) && (nx < width) && (ny < height)) {
                const int neighborPixel = ny * width + nx;

                if (!accessible[neighborPixel]) {
                    const int neighborLevel = bits[neighborPixel];

                    accessible[neighborPixel] = true;

                    if (neighborLevel >= curLevel) {
                        boundaryPixels[neighborLevel].push_back(neighborPixel << 4);

                        if (neighborLevel < priority)
                            priority = neighborLevel;
                    }
                    else {
                        boundaryPixels[curLevel].push_back((curPixel << 4) | (curEdge + 1));

                        if (curLevel < priority)
                            priority = curLevel;

                        curPixel = neighborPixel;
                        curEdge = 0;
                        curLevel = neighborLevel;

                        goto step_3;
                    }
                }
            }
        }

        // 5. Accumulate the current pixel to the component at the top of the stack (water
        // saturates the current pixel).
        regionStack.back()->accumulate(x, y);

        // 6. Pop the heap of boundary pixels. If the heap is empty, we are done. If the returned
        // pixel is at the same grey-level as the previous, go to 4.
        if (priority == 256) {
            regionStack.back()->process(delta_, minArea_, maxArea_, maxVariation_, minDiversity_);

            vector<Region> regions;

            regionStack.back()->save(regions);

            poolIndex_ = 0;

            return regions;
        }

        curPixel = boundaryPixels[priority].back() >> 4;
        curEdge = boundaryPixels[priority].back() & 0xf;

        boundaryPixels[priority].pop_back();

        while (boundaryPixels[priority].empty() && (priority < 256))
            ++priority;

        // 7. The returned pixel is at a higher grey-level, so we must now process all components on
        // the component stack until we reach the higher grey-level. This is done with the
        // processStack sub-routine, see below.
        // Then go to 4.
        const int newPixelGreyLevel = bits[curPixel];

        if (newPixelGreyLevel != curLevel) {
            curLevel = newPixelGreyLevel;

            processStack(newPixelGreyLevel, curPixel, regionStack);
        }
    }
}

void MSER::processStack(int newPixelGreyLevel, int pixel, vector<Region *> & regionStack) const
{
    // 1. Process component on the top of the stack. The next grey-level is the minimum of
    // newPixelGreyLevel and the grey-level for the second component on the stack.
    do {
        Region * top = regionStack.back();

        regionStack.pop_back();

        // 2. If newPixelGreyLevel is smaller than the grey-level on the second component on the
        // stack, set the top of stack grey-level to newPixelGreyLevel and return from sub-routine
        // (This occurs when the new pixel is at a grey-level for which there is not yet a component
        // instantiated, so we let the top of stack be that level by just changing its grey-level.
        if (newPixelGreyLevel < regionStack.back()->level) {
            regionStack.push_back(new (&pool_[poolIndex_++]) Region(newPixelGreyLevel, pixel));

            if (poolIndex_ == pool_.size())
                pool_.resize(pool_.size() + 256);

            regionStack.back()->merge(top);

            return;
        }

        // 3. Remove the top of stack and merge it into the second component on stack as follows:
        // Add the first and second moment accumulators together and/or join the pixel lists.
        // Either merge the histories of the components, or take the history from the winner. Note
        // here that the top of stack should be considered one ’time-step’ back, so its current
        // size is part of the history. Therefore the top of stack would be the winner if its
        // current size is larger than the previous size of second on stack.
        regionStack.back()->merge(top);
    }
    // 4. If(newPixelGreyLevel>top of stack grey-level) go to 1.
    while (newPixelGreyLevel > regionStack.back()->level);
}
