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

#include <torch/extension.h>
#include <vector>
// #include <iostream>
#include "fmm/include/thinks/fast_marching_method/fast_marching_method.hpp"

namespace fmm = thinks::fast_marching_method;
namespace F = torch::nn::functional;

template <std::size_t N>
std::vector<float> VaryingSpeedSignedArrivalTime(
    std::array<std::size_t, N> const& grid_size,
    std::vector<std::array<std::int32_t, N>> const& boundary_indices,
    std::vector<float> const& boundary_times, 
    std::array<float, N> const& grid_spacing,
    std::vector<float> speed_buffer) {

//   auto eikonal_solver = fmm::VaryingSpeedEikonalSolver<float, N>
//     (grid_spacing, grid_size, speed_buffer);

  auto eikonal_solver = fmm::HighAccuracyVaryingSpeedEikonalSolver<float, N>(
      grid_spacing, grid_size, speed_buffer);

  std::vector<float> arrival_times = fmm::SignedArrivalTime(
      grid_size, boundary_indices, boundary_times, eikonal_solver);

  return arrival_times;
}

torch::Tensor gradient2d(torch::Tensor image)
{
    const int channel = image.size(1);
    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU, 1)
            .requires_grad(false);
    
    auto gx = torch::zeros({1, 1, 1, 2}, options);
    auto gy = torch::zeros({1, 1, 2, 1}, options);

    auto gx_ptr = gx.accessor<float, 4>();
    auto gy_ptr = gy.accessor<float, 4>();

    gx_ptr[0][0][0][0] = -1.0;
    gx_ptr[0][0][0][1] = 1.0;

    gy_ptr[0][0][0][0] = -1.0;
    gy_ptr[0][0][1][0] = 1.0;

    // tile if channel>1 to have num_filters==channel
    gx = torch::tile(gx, {channel, 1, 1, 1});
    gy = torch::tile(gy, {channel, 1, 1, 1});

    // padding to enable 'same' padding for even kernel size
    // help from: https://github.com/masadcv/PyTorchSamePaddingExplainer
    auto imagepadx = torch::constant_pad_nd(image, {0, 1, 0, 0}, 0);
    auto imagepady = torch::constant_pad_nd(image, {0, 0, 0, 1}, 0);

    // dx and dy gradients
    auto imagegx = F::conv2d(imagepadx, 
                             gx, 
                             F::Conv2dFuncOptions()
                             .stride(1)
                             .groups(channel)
                            );
    auto imagegy = F::conv2d(imagepady, 
                             gy, 
                             F::Conv2dFuncOptions()
                             .stride(1)
                             .groups(channel)
                            );
    
    // combine dx and dy to get gradient
    auto imagegrad = 0.5 * (
        torch::abs(imagegx) + 
        torch::abs(imagegy)
        );
    
    // reduce sum over channel if multiple channels gradient 
    auto imagegradret = torch::sum(imagegrad, 1);

    return imagegradret;
}

// torch::Tensor gradient2dsobel(torch::Tensor image)
// {
//     // sobel filter
//     auto options = torch::TensorOptions()
//             .dtype(torch::kFloat32)
//             .device(torch::kCPU, 1)
//             .requires_grad(false);
    
//     auto gx = torch::zeros({1, 1, 3, 3}, options);
//     auto gy = torch::zeros({1, 1, 3, 3}, options);

//     auto gx_ptr = gx.accessor<float, 4>();
//     auto gy_ptr = gy.accessor<float, 4>();

//     gx_ptr[0][0][0][0] = 1;
//     gx_ptr[0][0][1][0] = 2;
//     gx_ptr[0][0][2][0] = 1;

//     gx_ptr[0][0][0][2] = -1;
//     gx_ptr[0][0][1][2] = -2;
//     gx_ptr[0][0][2][2] = -1;

//     gy_ptr[0][0][0][0] = 1;
//     gy_ptr[0][0][0][1] = 2;
//     gy_ptr[0][0][0][2] = 1;

//     gy_ptr[0][0][2][0] = -1;
//     gy_ptr[0][0][2][1] = -2;
//     gy_ptr[0][0][2][2] = -1;

//     auto imagegx = F::conv2d(image, 
//                              gx, 
//                              F::Conv2dFuncOptions()
//                              .stride(1)
//                              .padding(torch::kSame)
//                             );

//     auto imagegy = F::conv2d(image, 
//                              gy, 
//                              F::Conv2dFuncOptions()
//                              .stride(1)
//                              .padding(torch::kSame)
//                             );
    
//     auto imagegrad = 0.5 * (torch::abs(imagegx) + torch::abs(imagegy));

//     return imagegrad;
// }

void geodesic2d_fastmarch_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, height, width
    // const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    auto distance_ptr = distance.accessor<float, 4>();

    constexpr int nDim = 2;
    std::array<size_t, nDim> grid_size = {size_t(height), size_t(width)};
    std::vector<std::array<std::int32_t, nDim>> boundary_indices;
    std::vector<float> speed_buffer;
    std::vector<float>  boundary_times;

    // get image gradient
    auto imagegrad = torch::squeeze(gradient2d(image));
    auto imagegrad_ptr = imagegrad.accessor<float, 2>();
    
    // being conservative with sqrt(eps) for numerical stability
    float feps = std::sqrt(std::numeric_limits<float>::epsilon());

    // float upper = 2.0;
    // float lower = 0.05;
    // extract the boundary (seed) points
    for (int w = 0; w < width; w++)
    {
        for (int h = 0; h < height; h++)
        {
            if(distance_ptr[0][0][h][w] == 0)
            {   
                std::array<std::int32_t, nDim> c_point = {h, w};
                boundary_indices.push_back(c_point);
                boundary_times.push_back(0.0);
            }
            // vectorize speed
            float c_speed =  (
                l_grad * imagegrad_ptr[h][w] 
                + l_eucl
                + feps
                );
            // optional, clip gradient with lower/upper bounds
            // c_speed = std::max(lower, std::min(c_speed, upper));
            speed_buffer.push_back(1/c_speed);
            
        }
    }
    // std::cout << "Number of seed points: " << boundary_times.size() << std::endl;
    
    // it is a 2d image so spacing is constant
    std::array<float, nDim> spacing = {1.0, 1.0};

    std::vector<float> arrival_times = VaryingSpeedSignedArrivalTime<nDim>(
        grid_size, 
        boundary_indices,
        boundary_times,
        spacing,
        speed_buffer
    );

    // write back the values for arrival_times (distances)
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            distance_ptr[0][0][h][w] = arrival_times[w * height + h];
            // distance_ptr[0][0][h][w] = arrival_times[h * width + w];
        }
    }
}

torch::Tensor geodesic2d_fastmarch_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &l_grad,
    const float &l_eucl)
{
    torch::Tensor distance = mask.clone();

    geodesic2d_fastmarch_cpu(image, distance, l_grad, l_eucl);

    return distance;
}

torch::Tensor gradient3d(torch::Tensor image)
{
    const int channel = image.size(1);
    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU, 1)
            .requires_grad(false);
    
    auto gx = torch::zeros({1, 1, 1, 1, 2}, options);
    auto gy = torch::zeros({1, 1, 1, 2, 1}, options);
    auto gz = torch::zeros({1, 1, 2, 1, 1}, options);

    auto gx_ptr = gx.accessor<float, 5>();
    auto gy_ptr = gy.accessor<float, 5>();
    auto gz_ptr = gy.accessor<float, 5>();

    gx_ptr[0][0][0][0][0] = -1.0;
    gx_ptr[0][0][0][0][1] = 1.0;

    gy_ptr[0][0][0][0][0] = -1.0;
    gy_ptr[0][0][0][1][0] = 1.0;

    gz_ptr[0][0][0][0][0] = -1.0;
    gz_ptr[0][0][1][0][0] = 1.0;

    // tile if channel>1 to have num_filters==channel
    gx = torch::tile(gx, {channel, 1, 1, 1, 1});
    gy = torch::tile(gy, {channel, 1, 1, 1, 1});
    gz = torch::tile(gz, {channel, 1, 1, 1, 1});

    // padding to enable 'same' padding for even kernel size
    // help from: https://github.com/masadcv/PyTorchSamePaddingExplainer
    auto imagepadx = torch::constant_pad_nd(image, {0, 1, 0, 0, 0, 0}, 0);
    auto imagepady = torch::constant_pad_nd(image, {0, 0, 0, 1, 0, 0}, 0);
    auto imagepadz = torch::constant_pad_nd(image, {0, 0, 0, 0, 0, 1}, 0);

    // dx, dy and dz gradients
    auto imagegx = F::conv3d(imagepadx, 
                             gx, 
                             F::Conv3dFuncOptions()
                             .stride(1)
                             .groups(channel)
                            );

    auto imagegy = F::conv3d(imagepady, 
                             gy, 
                             F::Conv3dFuncOptions()
                             .stride(1)
                             .groups(channel)
                            );

    auto imagegz = F::conv3d(imagepadz, 
                             gz, 
                             F::Conv3dFuncOptions()
                             .stride(1)
                             .groups(channel)
                            );
    
    // combine dx, dy and dz to get gradient
    auto imagegrad = (1.0/3.0) * (
        torch::abs(imagegx) + 
        torch::abs(imagegy) + 
        torch::abs(imagegz)
        );

    // reduce sum over channel if multiple channels gradient 
    auto imagegradret = torch::sum(imagegrad, 1);

    return imagegradret;
}

void geodesic3d_fastmarch_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    std::vector<float> spacing,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, depth, height, width
    // const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    auto distance_ptr = distance.accessor<float, 5>();

    constexpr int nDim = 3;
    std::array<size_t, nDim> grid_size = {size_t(depth), size_t(height), size_t(width)};
    std::vector<std::array<std::int32_t, nDim>> boundary_indices;
    std::vector<float> speed_buffer;
    std::vector<float>  boundary_times;

    // get image gradient
    auto imagegrad = torch::squeeze(gradient3d(image));
    auto imagegrad_ptr = imagegrad.accessor<float, 3>();

    // being conservative with sqrt(eps) for numerical stability
    float feps = std::sqrt(std::numeric_limits<float>::epsilon());

    // float upper = 2.0;
    // float lower = 0.05;
    // extract the boundary (seed) points
    for (int w = 0; w < width; w++)
    {
        for(int h = 0; h < height; h++)
        {
            for(int d = 0; d < depth; d++)
            {
                if(distance_ptr[0][0][d][h][w] == 0)
                {   
                    std::array<std::int32_t, nDim> c_point = {d, h, w};
                    boundary_indices.push_back(c_point);
                    boundary_times.push_back(0.0);
                }
                // vectorize speed
                float c_speed =  (
                    l_grad * imagegrad_ptr[d][h][w] 
                    + l_eucl
                    + feps
                    );
                // optional, clip gradient with lower/upper bounds
                // c_speed = std::max(lower, std::min(c_speed, upper));
                speed_buffer.push_back(1/c_speed);
            }
        }
    }
    // std::cout << "Number of seed points: " << boundary_times.size() << std::endl;
    std::array<float, nDim> grid_spacing = {spacing[0], spacing[1], spacing[2]};

    std::vector<float> arrival_times = VaryingSpeedSignedArrivalTime<nDim>(
        grid_size, 
        boundary_indices,
        boundary_times,
        grid_spacing,
        speed_buffer
    );
    
    // write back the values for arrival_times (distances)
    for (int w = 0; w < width; w++)
    {
        for (int h = 0; h < height; h++)
        {
            for(int d = 0; d < depth; d++)
            {
                distance_ptr[0][0][d][h][w] = arrival_times[w * height * depth + h * depth + d];
            }
        }
    }

}

torch::Tensor geodesic3d_fastmarch_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &l_grad,
    const float &l_eucl)
{
    torch::Tensor distance = mask.clone();

    geodesic3d_fastmarch_cpu(image, distance, spacing, l_grad, l_eucl);

    return distance;
}