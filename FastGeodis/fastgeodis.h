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

#pragma once

#include <torch/extension.h>
#include <vector>
#include "common.h"

#ifdef WITH_CUDA
torch::Tensor generalised_geodesic2d_cuda(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic3d_cuda(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    std::vector<float> spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);
#endif

torch::Tensor generalised_geodesic2d_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic3d_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    std::vector<float> spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic2d_toivanen_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic3d_toivanen_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic2d_fastmarch_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl);

torch::Tensor generalised_geodesic3d_fastmarch_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl);


torch::Tensor generalised_geodesic2d(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic3d(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic2d_toivanen(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic3d_toivanen(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor generalised_geodesic2d_fastmarch(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl);

torch::Tensor generalised_geodesic3d_fastmarch(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl);


torch::Tensor signed_generalised_geodesic2d(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor signed_generalised_geodesic3d(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor signed_generalised_geodesic2d_toivanen(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor signed_generalised_geodesic3d_toivanen(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations);

torch::Tensor signed_generalised_geodesic2d_fastmarch(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl);

torch::Tensor signed_generalised_geodesic3d_fastmarch(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl);

torch::Tensor GSF2d(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const float &v, 
    const float &lambda, 
    const int &iterations);

torch::Tensor GSF3d(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &lambda, 
    const int &iterations);

torch::Tensor GSF2d_toivanen(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const float &v, 
    const float &lambda, 
    const int &iterations);

torch::Tensor GSF3d_toivanen(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &lambda, 
    const int &iterations);

torch::Tensor GSF2d_fastmarch(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const float &v, 
    const float &lambda);

torch::Tensor GSF3d_fastmarch(
    const torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &theta, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &lambda);