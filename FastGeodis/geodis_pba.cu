// BSD 3-Clause License

// Copyright (c) 2021, Muhammad Asad (masadcv@gmail.com)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// PBA+ (Parallel Banding Algorithm Plus) for Exact Euclidean Distance Transform
// Based on: Cao et al. (2010) "Parallel Banding Algorithm to compute exact distance transform with the GPU"
//
// Portions of this implementation are adapted from the cuCIM project
// (MIT License): https://github.com/rapidsai/cucim
//
// The adapted portions are used under the following MIT license:
//
//   Copyright (c) The cuCIM contributors
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy
//   of this software and associated documentation files (the "Software"), to deal
//   in the Software without restriction, including without limitation the rights
//   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//   THE SOFTWARE.
//
// This file also draws on the original PBA+ implementation:
//   https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus
//
// The parallel banding algorithm was originally described in:
// Thanh-Tung Cao, Ke Tang, Anis Mohamed, and Tiow-Seng Tan. 2010.
// Parallel Banding Algorithm to compute exact distance transform with the GPU.
// In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive 3D Graphics and Games.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#include <cfloat>
#include <cmath>

// ============================================================================
// PBA+ Constants and Macros
// ============================================================================
#define MARKER      -32768
#define BLOCKSIZE   32

#define TOID(x, y, size)  ((y) * (size) + (x))

// Use short2 for 2D coordinate encoding (supports images up to 32767x32767)
typedef short2 pixel_int2_t;
#define make_pixel(x, y)  make_short2(x, y)

// ============================================================================
// Domination test for Voronoi diagram construction
// Returns true if site2 dominates site3 at column x0 (site3 should be removed)
// ============================================================================
#define LL long long
__device__ __forceinline__ bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0)
{
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2 > \
           (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}
#undef LL

// Spacing-aware domination test
__device__ __forceinline__ bool dominate_sp(int _x1, int _y1, int _x2, int _y2, int _x3, int _y3, int _x0, float sx, float sy)
{
    float x1 = static_cast<float>(_x1) * sx;
    float x2 = static_cast<float>(_x2) * sx;
    float x3 = static_cast<float>(_x3) * sx;
    float y1 = static_cast<float>(_y1) * sy;
    float y2 = static_cast<float>(_y2) * sy;
    float y3 = static_cast<float>(_y3) * sy;
    float x0_2 = static_cast<float>(_x0 << 1) * sx;
    float k1 = (y2 - y1);
    float k2 = (y3 - y2);
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - x0_2)) * k2 > \
           (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - x0_2)) * k1;
}

// ============================================================================
// Phase 1 Kernels: Vertical (Y-axis) Processing
// ============================================================================

// Flood downward along columns
__global__ void kernelFloodDown(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);

    pixel_int2_t pixel1, pixel2;

    pixel1 = make_pixel(MARKER, MARKER);

    for (int i = 0; i < bandSize; i++, id += size) {
        pixel2 = input[id];

        if (pixel2.x != MARKER)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

// Flood upward along columns
__global__ void kernelFloodUp(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + 1) * bandSize - 1;
    int id = TOID(tx, ty, size);

    pixel_int2_t pixel1, pixel2;
    int dist1, dist2;

    pixel1 = make_pixel(MARKER, MARKER);

    for (int i = 0; i < bandSize; i++, id -= size) {
        dist1 = abs(pixel1.y - ty + i);

        pixel2 = input[id];
        dist2 = abs(pixel2.y - ty + i);

        if (dist2 < dist1)
            pixel1 = pixel2;

        output[id] = pixel1;
    }
}

// Propagate information between bands
__global__ void kernelPropagateInterband(pixel_int2_t *input, pixel_int2_t *margin_out, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc = bandSize * size;
    int ny, nid, nDist;
    pixel_int2_t pixel;

    int ty = blockIdx.y * bandSize;
    int topId = TOID(tx, ty, size);
    int bottomId = TOID(tx, ty + bandSize - 1, size);
    int tid = blockIdx.y * size + tx;
    int bid = tid + (size * size / bandSize);

    pixel = input[topId];
    int myDist = abs(pixel.y - ty);
    margin_out[tid] = pixel;

    for (nid = bottomId - inc; nid >= 0; nid -= inc) {
        pixel = input[nid];

        if (pixel.x != MARKER) {
            nDist = abs(pixel.y - ty);

            if (nDist < myDist)
                margin_out[tid] = pixel;

            break;
        }
    }

    ty = ty + bandSize - 1;
    pixel = input[bottomId];
    myDist = abs(pixel.y - ty);
    margin_out[bid] = pixel;

    for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
        pixel = input[nid];

        if (pixel.x != MARKER) {
            nDist = abs(pixel.y - ty);

            if (nDist < myDist)
                margin_out[bid] = pixel;

            break;
        }
    }
}

// Update vertical distances and transpose
__global__ void kernelUpdateVertical(pixel_int2_t *color, pixel_int2_t *margin, pixel_int2_t *output, int size, int bandSize)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;

    pixel_int2_t top = margin[blockIdx.y * size + tx];
    pixel_int2_t bottom = margin[(blockIdx.y + size / bandSize) * size + tx];
    pixel_int2_t pixel;

    int dist, myDist;

    int id = TOID(tx, ty, size);

    int n_step = bandSize / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        int y_start = blockIdx.y * bandSize + step * blockDim.x;
        int y_end = y_start + blockDim.x;

        for (ty = y_start; ty < y_end; ++ty, id += size) {
            pixel = color[id];
            myDist = abs(pixel.y - ty);

            dist = abs(top.y - ty);
            if (dist < myDist) { myDist = dist; pixel = top; }

            dist = abs(bottom.y - ty);
            if (dist < myDist) pixel = bottom;

            block[threadIdx.x][ty - y_start] = make_pixel(pixel.y, pixel.x);
        }

        __syncthreads();

        int tid = TOID(blockIdx.y * bandSize + step * blockDim.x + threadIdx.x, \
                       blockIdx.x * blockDim.x, size);

        for(int i = 0; i < blockDim.x; ++i, tid += size) {
            output[tid] = block[i][threadIdx.x];
        }

        __syncthreads();
    }
}

// ============================================================================
// Phase 2 Kernels: Horizontal (X-axis) Processing with Stack
// ============================================================================

// Build stack of proximate points using domination test (isotropic)
__global__ void kernelProximatePoints(pixel_int2_t *input, pixel_int2_t *stack, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);
    int lasty = -1;
    pixel_int2_t last1, last2, current;

    last1.y = -1; last2.y = -1;

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate(last1.x, last2.y, last2.x, lasty, current.x, current.y, tx))
                    break;

                lasty = last2.y; last2 = last1;

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)];
            }

            last1 = last2; last2 = make_pixel(current.x, lasty); lasty = current.y;

            stack[id] = last2;
        }
    }

    if (lasty != ty + bandSize - 1)
        stack[TOID(tx, ty + bandSize - 1, size)] = make_pixel(MARKER, lasty);
}

// Build stack with spacing support
__global__ void kernelProximatePointsWithSpacing(pixel_int2_t *input, pixel_int2_t *stack, int size, int bandSize, float sx, float sy)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * bandSize;
    int id = TOID(tx, ty, size);
    int lasty = -1;
    pixel_int2_t last1, last2, current;

    last1.y = -1; last2.y = -1;

    for (int i = 0; i < bandSize; i++, id += size) {
        current = input[id];

        if (current.x != MARKER) {
            while (last2.y >= 0) {
                if (!dominate_sp(last1.x, last2.y, last2.x, lasty, current.x, current.y, tx, sx, sy))
                    break;

                lasty = last2.y; last2 = last1;

                if (last1.y >= 0)
                    last1 = stack[TOID(tx, last1.y, size)];
            }

            last1 = last2; last2 = make_pixel(current.x, lasty); lasty = current.y;

            stack[id] = last2;
        }
    }

    if (lasty != ty + bandSize - 1)
        stack[TOID(tx, ty + bandSize - 1, size)] = make_pixel(MARKER, lasty);
}

// Create forward pointers from backward-linked structure
__global__ void kernelCreateForwardPointers(pixel_int2_t *input, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + 1) * bandSize - 1;
    int id = TOID(tx, ty, size);
    int lasty = -1, nexty;
    pixel_int2_t current;

    current = input[id];

    if (current.x == MARKER)
        nexty = current.y;
    else
        nexty = ty;

    for (int i = 0; i < bandSize; i++, id -= size)
        if (ty - i == nexty) {
            current = make_pixel(lasty, input[id].y);
            output[id] = current;

            lasty = nexty;
            nexty = current.y;
        }

    if (lasty != ty - bandSize + 1)
        output[id + size] = make_pixel(lasty, MARKER);
}

// Merge adjacent bands (isotropic)
__global__ void kernelMergeBands(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size, int bandSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty, lasty;
    pixel_int2_t last1, last2, current;

    lasty = band2 * bandSize - 1;
    last2 = make_pixel(color[TOID(tx, lasty, size)].x, link[TOID(tx, lasty, size)].y);

    if (last2.x == MARKER) {
        lasty = last2.y;

        if (lasty >= 0)
            last2 = make_pixel(color[TOID(tx, lasty, size)].x, link[TOID(tx, lasty, size)].y);
        else
            last2 = make_pixel(MARKER, MARKER);
    }

    if (last2.y >= 0) {
        last1 = make_pixel(color[TOID(tx, last2.y, size)].x, link[TOID(tx, last2.y, size)].y);
    }

    firsty = band2 * bandSize;
    current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);

    if (current.y == MARKER) {
        firsty = current.x;

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    int top = 0;

    while (top < 2 && current.y >= 0) {
        while (last2.y >= 0) {
            if (!dominate(last1.x, last2.y, last2.x, lasty, current.y, firsty, tx))
                break;

            lasty = last2.y; last2 = last1;
            top--;

            if (last1.y >= 0)
                last1 = make_pixel(color[TOID(tx, last1.y, size)].x, link[TOID(tx, last1.y, size)].y);
        }

        output[TOID(tx, firsty, size)] = make_pixel(current.x, lasty);

        if (lasty >= 0)
            output[TOID(tx, lasty, size)] = make_pixel(firsty, last2.y);

        last1 = last2; last2 = make_pixel(current.y, lasty); lasty = firsty;
        firsty = current.x;

        top = max(1, top + 1);

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    firsty = band1 * bandSize;
    lasty = band2 * bandSize;
    current = link[TOID(tx, firsty, size)];

    if (current.y == MARKER && current.x < 0) {
        last1 = link[TOID(tx, lasty, size)];

        if (last1.y == MARKER)
            current.x = last1.x;
        else
            current.x = lasty;

        output[TOID(tx, firsty, size)] = current;
    }

    firsty = band1 * bandSize + bandSize - 1;
    lasty = band2 * bandSize + bandSize - 1;
    current = link[TOID(tx, lasty, size)];

    if (current.x == MARKER && current.y < 0) {
        last1 = link[TOID(tx, firsty, size)];

        if (last1.x == MARKER)
            current.y = last1.y;
        else
            current.y = firsty;

        output[TOID(tx, lasty, size)] = current;
    }
}

// Merge bands with spacing support
__global__ void kernelMergeBandsWithSpacing(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size, int bandSize, float sx, float sy)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int band1 = blockIdx.y * 2;
    int band2 = band1 + 1;
    int firsty, lasty;
    pixel_int2_t last1, last2, current;

    lasty = band2 * bandSize - 1;
    last2 = make_pixel(color[TOID(tx, lasty, size)].x, link[TOID(tx, lasty, size)].y);

    if (last2.x == MARKER) {
        lasty = last2.y;

        if (lasty >= 0)
            last2 = make_pixel(color[TOID(tx, lasty, size)].x, link[TOID(tx, lasty, size)].y);
        else
            last2 = make_pixel(MARKER, MARKER);
    }

    if (last2.y >= 0) {
        last1 = make_pixel(color[TOID(tx, last2.y, size)].x, link[TOID(tx, last2.y, size)].y);
    }

    firsty = band2 * bandSize;
    current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);

    if (current.y == MARKER) {
        firsty = current.x;

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    int top = 0;

    while (top < 2 && current.y >= 0) {
        while (last2.y >= 0) {
            if (!dominate_sp(last1.x, last2.y, last2.x, lasty, current.y, firsty, tx, sx, sy))
                break;

            lasty = last2.y; last2 = last1;
            top--;

            if (last1.y >= 0)
                last1 = make_pixel(color[TOID(tx, last1.y, size)].x, link[TOID(tx, last1.y, size)].y);
        }

        output[TOID(tx, firsty, size)] = make_pixel(current.x, lasty);

        if (lasty >= 0)
            output[TOID(tx, lasty, size)] = make_pixel(firsty, last2.y);

        last1 = last2; last2 = make_pixel(current.y, lasty); lasty = firsty;
        firsty = current.x;

        top = max(1, top + 1);

        if (firsty >= 0)
            current = make_pixel(link[TOID(tx, firsty, size)].x, color[TOID(tx, firsty, size)].x);
        else
            current = make_pixel(MARKER, MARKER);
    }

    firsty = band1 * bandSize;
    lasty = band2 * bandSize;
    current = link[TOID(tx, firsty, size)];

    if (current.y == MARKER && current.x < 0) {
        last1 = link[TOID(tx, lasty, size)];

        if (last1.y == MARKER)
            current.x = last1.x;
        else
            current.x = lasty;

        output[TOID(tx, firsty, size)] = current;
    }

    firsty = band1 * bandSize + bandSize - 1;
    lasty = band2 * bandSize + bandSize - 1;
    current = link[TOID(tx, lasty, size)];

    if (current.x == MARKER && current.y < 0) {
        last1 = link[TOID(tx, firsty, size)];

        if (last1.x == MARKER)
            current.y = last1.y;
        else
            current.y = firsty;

        output[TOID(tx, lasty, size)] = current;
    }
}

// Convert double-linked list to single list
__global__ void kernelDoubleToSingleList(pixel_int2_t *color, pixel_int2_t *link, pixel_int2_t *output, int size)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;
    int id = TOID(tx, ty, size);

    output[id] = make_pixel(color[id].x, link[id].y);
}

// ============================================================================
// Phase 3 Kernels: Final Distance Computation
// ============================================================================

// Compute final Voronoi coloring (isotropic)
__global__ void kernelColor(pixel_int2_t *input, pixel_int2_t *output, int size)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int dx, dy, lasty;
    unsigned int best, dist;
    pixel_int2_t last1, last2;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = max(last2.y, 0);
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0)
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = last2.x - tx; dy = lasty - ty;
            best = dist = dx * dx + dy * dy;

            while (last2.y >= 0) {
                dx = last1.x - tx; dy = last2.y - ty;
                dist = dx * dx + dy * dy;

                if (dist > best)
                    break;

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0)
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_pixel(lasty, last2.x);
        }

        __syncthreads();

        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for(int i = 0; i < blockDim.x; ++i, id += size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}

// Compute final Voronoi coloring with spacing
__global__ void kernelColorWithSpacing(pixel_int2_t *input, pixel_int2_t *output, int size, float sx, float sy)
{
    __shared__ pixel_int2_t block[BLOCKSIZE][BLOCKSIZE];

    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int lasty;
    float dx, dy, best, dist;
    pixel_int2_t last1, last2;

    lasty = size - 1;

    last2 = input[TOID(tx, lasty, size)];

    if (last2.x == MARKER) {
        lasty = max(last2.y, 0);
        last2 = input[TOID(tx, lasty, size)];
    }

    if (last2.y >= 0)
        last1 = input[TOID(tx, last2.y, size)];

    int y_start, y_end, n_step = size / blockDim.x;
    for(int step = 0; step < n_step; ++step) {
        y_start = size - step * blockDim.x - 1;
        y_end = size - (step + 1) * blockDim.x;

        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            dx = static_cast<float>(last2.x - tx) * sx;
            dy = static_cast<float>(lasty - ty) * sy;
            best = dist = dx * dx + dy * dy;

            while (last2.y >= 0) {
                dx = static_cast<float>(last1.x - tx) * sx;
                dy = static_cast<float>(last2.y - ty) * sy;
                dist = dx * dx + dy * dy;

                if (dist > best)
                    break;

                best = dist; lasty = last2.y; last2 = last1;

                if (last2.y >= 0)
                    last1 = input[TOID(tx, last2.y, size)];
            }

            block[threadIdx.x][ty - y_end] = make_pixel(lasty, last2.x);
        }

        __syncthreads();

        if(!threadIdx.y) {
            int id = TOID(y_end + threadIdx.x, blockIdx.x * blockDim.x, size);
            for(int i = 0; i < blockDim.x; ++i, id += size) {
                output[id] = block[i][threadIdx.x];
            }
        }

        __syncthreads();
    }
}

// ============================================================================
// Kernel to initialize buffer to MARKER (for proper padding initialization)
// Note: cudaMemset cannot be used because MARKER (-32768 = 0x8000) cannot be
// represented as a single byte pattern. This kernel ensures the padded region
// is properly initialized to MARKER values.
// ============================================================================
__global__ void kernelInitToMarker(pixel_int2_t* __restrict__ buffer, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;

    int id = y * size + x;
    buffer[id] = make_pixel(MARKER, MARKER);
}

// ============================================================================
// Kernel to initialize sites from mask
// ============================================================================
__global__ void kernelInitSites(const float* __restrict__ mask, pixel_int2_t* __restrict__ sites,
                                 int width, int height, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // mask has stride 'width', sites buffer has stride 'size' (padded)
    int mask_id = y * width + x;
    int site_id = y * size + x;

    // mask == 0 means seed point (foreground in EDT terms)
    if (mask[mask_id] < 0.5f) {
        sites[site_id] = make_pixel(x, y);
    } else {
        sites[site_id] = make_pixel(MARKER, MARKER);
    }
}

// ============================================================================
// Kernel to compute final distance from Voronoi sites
// ============================================================================
__global__ void kernelComputeDistance(const pixel_int2_t* __restrict__ sites, float* __restrict__ distance,
                                       int width, int height, int size, float sx, float sy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // sites buffer has stride 'size' (padded), output has stride 'width'
    int site_id = y * size + x;
    int out_id = y * width + x;
    pixel_int2_t site = sites[site_id];

    if (site.x == MARKER) {
        distance[out_id] = FLT_MAX;
    } else {
        float dx = static_cast<float>(x - site.x) * sx;
        float dy = static_cast<float>(y - site.y) * sy;
        distance[out_id] = sqrtf(dx * dx + dy * dy);
    }
}

// ============================================================================
// Helper: Compute band sizes for PBA+
// ============================================================================
inline void computeBandSizes(int size, int& m1, int& m2)
{
    // Use heuristics from PBA+ paper
    // m1: band size for phase 1 (should divide size)
    // m2: band size for phase 2 (should be power of 2 and divide size)

    // Default values that work well
    m1 = BLOCKSIZE;
    m2 = BLOCKSIZE;

    // Adjust if size is not divisible
    while (size % m1 != 0 && m1 > 1) m1 /= 2;
    while (size % m2 != 0 && m2 > 1) m2 /= 2;

    // Ensure minimum band size
    if (m1 < 1) m1 = 1;
    if (m2 < 1) m2 = 1;
}

// ============================================================================
// Main 2D PBA+ EDT function
// ============================================================================
torch::Tensor exact_euclidean2d_cuda(
    const torch::Tensor &mask,
    const std::vector<float> &spacing
) {
    int device = mask.get_device();
    c10::cuda::CUDAGuard device_guard(device);

    // Get dimensions (assume BCHW format)
    const int batch = mask.size(0);
    const int channels = mask.size(1);
    const int height = mask.size(2);
    const int width = mask.size(3);

    // spacing is [height_spacing, width_spacing] to match [H, W] tensor convention
    float sy = spacing.size() > 0 ? spacing[0] : 1.0f;  // height (row) spacing
    float sx = spacing.size() > 1 ? spacing[1] : 1.0f;  // width (column) spacing
    bool use_spacing = (sx != 1.0f || sy != 1.0f);

    // Early return if all pixels are seeds (mask=0 means seed, so sum=0 means all seeds)
    // In this case, all distances should be 0
    if (mask.sum().item<float>() < 0.5f) {
        return torch::zeros({batch, channels, height, width},
                           torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    }

    // PBA+ requires square images with size being power of 2
    // Round up to next power of 2
    int size = max(width, height);

    // Next power of 2 using bit manipulation
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size++;

    // Ensure size is at least 2*BLOCKSIZE for the algorithm to work
    if (size < 2 * BLOCKSIZE) size = 2 * BLOCKSIZE;

    // Compute band sizes
    int bandSize1, bandSize2;
    computeBandSizes(size, bandSize1, bandSize2);

    // Allocate output
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(mask.device());
    torch::Tensor distance = torch::zeros({batch, channels, height, width}, options_float);

    // Allocate intermediate buffers (padded size)
    auto options_short2 = torch::TensorOptions().dtype(torch::kInt32).device(mask.device());
    // We use int32 but interpret as short2 (same size)
    torch::Tensor buffer1 = torch::full({size, size}, 0, options_short2);
    torch::Tensor buffer2 = torch::full({size, size}, 0, options_short2);
    torch::Tensor margin = torch::full({2 * size * size / bandSize1}, 0, options_short2);

    // Process each batch and channel
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            // Get pointers
            const float* mask_ptr = mask.data_ptr<float>() + (b * channels + c) * height * width;
            float* dist_ptr = distance.data_ptr<float>() + (b * channels + c) * height * width;
            pixel_int2_t* buf1_ptr = reinterpret_cast<pixel_int2_t*>(buffer1.data_ptr<int>());
            pixel_int2_t* buf2_ptr = reinterpret_cast<pixel_int2_t*>(buffer2.data_ptr<int>());
            pixel_int2_t* margin_ptr = reinterpret_cast<pixel_int2_t*>(margin.data_ptr<int>());

            // Initialize buffers to MARKER using kernel (cudaMemset cannot set MARKER correctly)
            {
                dim3 block(16, 16);
                dim3 grid((size + 15) / 16, (size + 15) / 16);
                kernelInitToMarker<<<grid, block>>>(buf1_ptr, size);
                kernelInitToMarker<<<grid, block>>>(buf2_ptr, size);
            }

            // Initialize sites from mask (only for valid region)
            {
                dim3 block(16, 16);
                dim3 grid((width + 15) / 16, (height + 15) / 16);
                kernelInitSites<<<grid, block>>>(mask_ptr, buf1_ptr, width, height, size);
            }

            // Phase 1: Vertical processing
            {
                dim3 block(BLOCKSIZE);
                dim3 grid(size / BLOCKSIZE, size / bandSize1);

                kernelFloodDown<<<grid, block>>>(buf1_ptr, buf1_ptr, size, bandSize1);
                kernelFloodUp<<<grid, block>>>(buf1_ptr, buf1_ptr, size, bandSize1);
                kernelPropagateInterband<<<grid, block>>>(buf1_ptr, margin_ptr, size, bandSize1);
                kernelUpdateVertical<<<grid, block>>>(buf1_ptr, margin_ptr, buf2_ptr, size, bandSize1);
            }

            // Phase 2: Horizontal processing with band merging
            {
                dim3 block(BLOCKSIZE);
                dim3 grid(size / BLOCKSIZE, size / bandSize2);

                if (use_spacing) {
                    kernelProximatePointsWithSpacing<<<grid, block>>>(buf2_ptr, buf1_ptr, size, bandSize2, sx, sy);
                } else {
                    kernelProximatePoints<<<grid, block>>>(buf2_ptr, buf1_ptr, size, bandSize2);
                }

                kernelCreateForwardPointers<<<grid, block>>>(buf1_ptr, buf1_ptr, size, bandSize2);

                // Iteratively merge bands
                int noBand = size / bandSize2;
                dim3 grid2(size / BLOCKSIZE, noBand / 2);

                while (noBand > 1) {
                    if (use_spacing) {
                        kernelMergeBandsWithSpacing<<<grid2, block>>>(buf2_ptr, buf1_ptr, buf1_ptr, size, size / noBand, sx, sy);
                    } else {
                        kernelMergeBands<<<grid2, block>>>(buf2_ptr, buf1_ptr, buf1_ptr, size, size / noBand);
                    }
                    noBand /= 2;
                    grid2.y = noBand / 2;
                    if (grid2.y == 0) break;
                }

                dim3 grid3(size / BLOCKSIZE, size);
                kernelDoubleToSingleList<<<grid3, block>>>(buf2_ptr, buf1_ptr, buf1_ptr, size);
            }

            // Phase 3: Final coloring
            {
                dim3 block(BLOCKSIZE, BLOCKSIZE);
                dim3 grid(size / BLOCKSIZE);

                if (use_spacing) {
                    kernelColorWithSpacing<<<grid, block>>>(buf1_ptr, buf2_ptr, size, sx, sy);
                } else {
                    kernelColor<<<grid, block>>>(buf1_ptr, buf2_ptr, size);
                }
            }

            // Compute final distances (only for valid region)
            {
                dim3 block(16, 16);
                dim3 grid((width + 15) / 16, (height + 15) / 16);
                kernelComputeDistance<<<grid, block>>>(buf2_ptr, dist_ptr, width, height, size, sx, sy);
            }
        }
    }

    return distance;
}

// ============================================================================
// 3D PBA+ Implementation
// ============================================================================

// For 3D, we use a similar approach but process along Z, Y, X axes
// Using int for 3D coordinate encoding (10 bits per coordinate, supports up to 1024^3)

#define MARKER3D    -1
#define ENCODE3D(x, y, z)  (((x) << 20) | ((y) << 10) | (z))
#define DECODE3D_X(v)      (((v) >> 20) & 0x3FF)
#define DECODE3D_Y(v)      (((v) >> 10) & 0x3FF)
#define DECODE3D_Z(v)      ((v) & 0x3FF)

// 3D domination test with spacing
__device__ __forceinline__ bool dominate3d_sp(
    int x1, int y1, int z1,
    int x2, int y2, int z2,
    int x3, int y3, int z3,
    int px, int py,
    float sx, float sy, float sz
) {
    // Check if (x2,y2,z2) dominates (x3,y3,z3) at position (px,py,*)
    float fx1 = x1 * sx, fy1 = y1 * sy, fz1 = z1 * sz;
    float fx2 = x2 * sx, fy2 = y2 * sy, fz2 = z2 * sz;
    float fx3 = x3 * sx, fy3 = y3 * sy, fz3 = z3 * sz;
    float fpx = px * sx, fpy = py * sy;

    float d12_xy = (fx1 - fpx) * (fx1 - fpx) + (fy1 - fpy) * (fy1 - fpy) -
                   (fx2 - fpx) * (fx2 - fpx) - (fy2 - fpy) * (fy2 - fpy);
    float d23_xy = (fx2 - fpx) * (fx2 - fpx) + (fy2 - fpy) * (fy2 - fpy) -
                   (fx3 - fpx) * (fx3 - fpx) - (fy3 - fpy) * (fy3 - fpy);

    float dz12 = fz2 - fz1;
    float dz23 = fz3 - fz2;

    if (dz12 == 0 || dz23 == 0) return false;

    float t1 = (d12_xy + fz1 * fz1 - fz2 * fz2) / (2 * dz12);
    float t2 = (d23_xy + fz2 * fz2 - fz3 * fz3) / (2 * dz23);

    return t1 >= t2;
}

// Flood along Z-axis for 3D
__global__ void kernelFloodZ3D(int* input, int* output, int sizeX, int sizeY, int sizeZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= sizeX || y >= sizeY) return;

    int lastSite = MARKER3D;

    // Forward pass
    for (int z = 0; z < sizeZ; z++) {
        int id = z * sizeY * sizeX + y * sizeX + x;
        int site = input[id];

        if (site != MARKER3D)
            lastSite = site;

        output[id] = lastSite;
    }

    lastSite = MARKER3D;

    // Backward pass
    for (int z = sizeZ - 1; z >= 0; z--) {
        int id = z * sizeY * sizeX + y * sizeX + x;
        int site = input[id];

        if (site != MARKER3D)
            lastSite = site;

        int siteForward = output[id];

        if (lastSite == MARKER3D)
            continue;

        if (siteForward == MARKER3D) {
            output[id] = lastSite;
        } else {
            int zForward = DECODE3D_Z(siteForward);
            int zBack = DECODE3D_Z(lastSite);

            if (abs(z - zBack) < abs(z - zForward))
                output[id] = lastSite;
        }
    }
}

// Process Y-axis for 3D (Maurer's algorithm)
__global__ void kernelMaurerY3D(int* input, int* output, int sizeX, int sizeY, int sizeZ,
                                 float sx, float sy, float sz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;

    if (x >= sizeX || z >= sizeZ) return;

    int lastSite = MARKER3D;

    // Forward pass
    for (int y = 0; y < sizeY; y++) {
        int id = z * sizeY * sizeX + y * sizeX + x;
        int site = input[id];

        if (site != MARKER3D)
            lastSite = site;

        output[id] = lastSite;
    }

    lastSite = MARKER3D;

    // Backward pass with distance comparison
    for (int y = sizeY - 1; y >= 0; y--) {
        int id = z * sizeY * sizeX + y * sizeX + x;
        int site = input[id];

        if (site != MARKER3D)
            lastSite = site;

        int siteForward = output[id];

        if (lastSite == MARKER3D)
            continue;

        if (siteForward == MARKER3D) {
            output[id] = lastSite;
        } else {
            // Compare distances
            int xf = DECODE3D_X(siteForward), yf = DECODE3D_Y(siteForward), zf = DECODE3D_Z(siteForward);
            int xb = DECODE3D_X(lastSite), yb = DECODE3D_Y(lastSite), zb = DECODE3D_Z(lastSite);

            float dxf = (x - xf) * sx, dyf = (y - yf) * sy, dzf = (z - zf) * sz;
            float dxb = (x - xb) * sx, dyb = (y - yb) * sy, dzb = (z - zb) * sz;

            float distF = dxf*dxf + dyf*dyf + dzf*dzf;
            float distB = dxb*dxb + dyb*dyb + dzb*dzb;

            if (distB < distF)
                output[id] = lastSite;
        }
    }
}

// Process X-axis for 3D and compute final distance
__global__ void kernelColorX3D(int* input, float* distance, int sizeX, int sizeY, int sizeZ,
                                float sx, float sy, float sz)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

    int id = z * sizeY * sizeX + y * sizeX + x;

    int bestSite = input[id];
    float bestDist = FLT_MAX;

    if (bestSite != MARKER3D) {
        int bx = DECODE3D_X(bestSite), by = DECODE3D_Y(bestSite), bz = DECODE3D_Z(bestSite);
        float dx = (x - bx) * sx, dy = (y - by) * sy, dz = (z - bz) * sz;
        bestDist = dx*dx + dy*dy + dz*dz;
    }

    // Search left
    for (int lx = x - 1; lx >= 0; lx--) {
        float horizDist = (x - lx) * sx;
        if (horizDist * horizDist >= bestDist) break;

        int leftSite = input[z * sizeY * sizeX + y * sizeX + lx];
        if (leftSite != MARKER3D) {
            int lsx = DECODE3D_X(leftSite), lsy = DECODE3D_Y(leftSite), lsz = DECODE3D_Z(leftSite);
            float dx = (x - lsx) * sx, dy = (y - lsy) * sy, dz = (z - lsz) * sz;
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < bestDist) {
                bestDist = dist;
            }
        }
    }

    // Search right
    for (int rx = x + 1; rx < sizeX; rx++) {
        float horizDist = (rx - x) * sx;
        if (horizDist * horizDist >= bestDist) break;

        int rightSite = input[z * sizeY * sizeX + y * sizeX + rx];
        if (rightSite != MARKER3D) {
            int rsx = DECODE3D_X(rightSite), rsy = DECODE3D_Y(rightSite), rsz = DECODE3D_Z(rightSite);
            float dx = (x - rsx) * sx, dy = (y - rsy) * sy, dz = (z - rsz) * sz;
            float dist = dx*dx + dy*dy + dz*dz;
            if (dist < bestDist) {
                bestDist = dist;
            }
        }
    }

    distance[id] = (bestDist < FLT_MAX) ? sqrtf(bestDist) : FLT_MAX;
}

// Initialize 3D sites from mask
__global__ void kernelInitSites3D(const float* mask, int* sites, int sizeX, int sizeY, int sizeZ)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

    int id = z * sizeY * sizeX + y * sizeX + x;

    if (mask[id] < 0.5f) {
        sites[id] = ENCODE3D(x, y, z);
    } else {
        sites[id] = MARKER3D;
    }
}

// ============================================================================
// Main 3D PBA+ EDT function
// ============================================================================
torch::Tensor exact_euclidean3d_cuda(
    const torch::Tensor &mask,
    const std::vector<float> &spacing
) {
    int device = mask.get_device();
    c10::cuda::CUDAGuard device_guard(device);

    // Get dimensions (assume BCDHW format)
    const int batch = mask.size(0);
    const int channels = mask.size(1);
    const int depth = mask.size(2);
    const int height = mask.size(3);
    const int width = mask.size(4);

    // spacing is [depth_spacing, height_spacing, width_spacing] to match [D, H, W] tensor convention
    float sz = spacing.size() > 0 ? spacing[0] : 1.0f;  // depth (z) spacing
    float sy = spacing.size() > 1 ? spacing[1] : 1.0f;  // height (y) spacing
    float sx = spacing.size() > 2 ? spacing[2] : 1.0f;  // width (x) spacing

    // Early return if all pixels are seeds (mask=0 means seed, so sum=0 means all seeds)
    if (mask.sum().item<float>() < 0.5f) {
        return torch::zeros({batch, channels, depth, height, width},
                           torch::TensorOptions().dtype(torch::kFloat32).device(mask.device()));
    }

    // Allocate output
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(mask.device());
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(mask.device());

    torch::Tensor distance = torch::zeros({batch, channels, depth, height, width}, options_float);

    // Allocate intermediate buffers
    torch::Tensor buffer1 = torch::full({depth, height, width}, MARKER3D, options_int);
    torch::Tensor buffer2 = torch::full({depth, height, width}, MARKER3D, options_int);

    // Process each batch and channel
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            const float* mask_ptr = mask.data_ptr<float>() + (b * channels + c) * depth * height * width;
            float* dist_ptr = distance.data_ptr<float>() + (b * channels + c) * depth * height * width;
            int* buf1_ptr = buffer1.data_ptr<int>();
            int* buf2_ptr = buffer2.data_ptr<int>();

            // Initialize sites
            {
                dim3 block(16, 16, 1);
                dim3 grid((width + 15) / 16, (height + 15) / 16, depth);
                kernelInitSites3D<<<grid, block>>>(mask_ptr, buf1_ptr, width, height, depth);
            }

            // Phase 1: Z-axis flooding
            {
                dim3 block(16, 16);
                dim3 grid((width + 15) / 16, (height + 15) / 16);
                kernelFloodZ3D<<<grid, block>>>(buf1_ptr, buf2_ptr, width, height, depth);
            }

            // Phase 2: Y-axis processing
            {
                dim3 block(256);
                dim3 grid((width + 255) / 256, 1, depth);
                kernelMaurerY3D<<<grid, block>>>(buf2_ptr, buf1_ptr, width, height, depth, sx, sy, sz);
            }

            // Phase 3: X-axis processing and distance computation
            {
                dim3 block(16, 16);
                dim3 grid((width + 15) / 16, (height + 15) / 16, depth);
                kernelColorX3D<<<grid, block>>>(buf1_ptr, dist_ptr, width, height, depth, sx, sy, sz);
            }
        }
    }

    return distance;
}

// ============================================================================
// Signed EDT functions
// ============================================================================
torch::Tensor signed_exact_euclidean2d_cuda(
    const torch::Tensor &mask,
    const std::vector<float> &spacing
) {
    // Distance from foreground (mask > 0.5)
    torch::Tensor D_fg = exact_euclidean2d_cuda(mask, spacing);

    // Distance from background - invert mask
    torch::Tensor inv_mask = 1.0f - mask;
    torch::Tensor D_bg = exact_euclidean2d_cuda(inv_mask, spacing);

    // Signed distance: negative inside (mask=1), positive outside (mask=0)
    return D_bg - D_fg;
}

torch::Tensor signed_exact_euclidean3d_cuda(
    const torch::Tensor &mask,
    const std::vector<float> &spacing
) {
    // Distance from foreground
    torch::Tensor D_fg = exact_euclidean3d_cuda(mask, spacing);

    // Distance from background - invert mask
    torch::Tensor inv_mask = 1.0f - mask;
    torch::Tensor D_bg = exact_euclidean3d_cuda(inv_mask, spacing);

    // Signed distance: negative inside (mask=1), positive outside (mask=0)
    return D_bg - D_fg;
}
