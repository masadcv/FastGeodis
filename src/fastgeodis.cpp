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
#include <iostream>
#include <vector>
#include "fastgeodis.h"
#include "common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 0

torch::Tensor generalised_geodesic2d(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "Compiled with OpenMP support" << std::endl;
        #else
            std::cout << "Not compiled with OpenMP support" << std::endl;
        #endif
        #ifdef WITH_CUDA
            std::cout << "Compiled with CUDA support" << std::endl;
        #else
            std::cout << "Not compiled with CUDA support" << std::endl;
        #endif
    #endif

    // check input dimensions
    const int num_dims = mask.dim();
    if (num_dims != 4)
    {
        throw std::runtime_error(
            "function only supports 2D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    if (image.is_cuda()) 
    {
    #ifdef WITH_CUDA
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error(
                "cuda.is_available() returned false, please check if the library was compiled successfully with CUDA support");
        }
        CHECK_CUDA(image);
        CHECK_CUDA(mask);

        return generalised_geodesic2d_cuda(image, mask, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with CUDA support.");
    #endif
    }
    return generalised_geodesic2d_cpu(image, mask, v, l_grad, l_eucl, iterations);
}

torch::Tensor generalised_geodesic3d(torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "Compiled with OpenMP support" << std::endl;
        #else
            std::cout << "Not compiled with OpenMP support" << std::endl;
        #endif
        #ifdef WITH_CUDA
            std::cout << "Compiled with CUDA support" << std::endl;
        #else
            std::cout << "Not compiled with CUDA support" << std::endl;
        #endif
    #endif

    // check input dimensions
    const int num_dims = mask.dim();
    if (num_dims != 5)
    {
        throw std::runtime_error(
            "function only supports 3D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    if (spacing.size() != 3)
    {
        throw std::runtime_error(
            "function only supports 3D spacing inputs, received " + std::to_string(spacing.size()));
    }

    // square spacing with transform
    // std::transform(spacing.begin(), spacing.end(), spacing.begin(), spacing.begin(), std::multiplies<float>());

    if (image.is_cuda()) 
    {
    #ifdef WITH_CUDA
        if (!torch::cuda::is_available())
        {   
            throw std::runtime_error(
                "cuda.is_available() returned false, please check if the library was compiled successfully with CUDA support");
        }
        CHECK_CUDA(image);
        CHECK_CUDA(mask);
        
        return generalised_geodesic3d_cuda(image, mask, spacing, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with CUDA support.");
    #endif
    }
    return generalised_geodesic3d_cpu(image, mask, spacing, v, l_grad, l_eucl, iterations);
}

torch::Tensor getDs2d(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor D_M = generalised_geodesic2d(image, mask, v, lambda, 1 - lambda, iterations);
    torch::Tensor D_Mb = generalised_geodesic2d(image, 1 - mask, v, lambda, 1 - lambda, iterations);

    return D_M - D_Mb;
}

torch::Tensor GSF2d(torch::Tensor &image, const torch::Tensor &mask, const float &theta, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = getDs2d(image, mask, v, lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -getDs2d(image, 1 - Md, v, lambda, iterations);
    torch::Tensor De_Me = getDs2d(image, Me, v, lambda, iterations);

    return Dd_Md + De_Me;
}

torch::Tensor getDs3d(torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor D_M = generalised_geodesic3d(image, mask, spacing, v, lambda, 1 - lambda, iterations);
    torch::Tensor D_Mb = generalised_geodesic3d(image, 1 - mask, spacing, v, lambda, 1 - lambda, iterations);

    return D_M - D_Mb;
}

torch::Tensor GSF3d(torch::Tensor &image, const torch::Tensor &mask, const float &theta, const std::vector<float> &spacing, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = getDs3d(image, mask, spacing, v, lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -getDs3d(image, 1 - Md, spacing, v, lambda, iterations);
    torch::Tensor De_Me = getDs3d(image, Me, spacing, v, lambda, iterations);

    return Dd_Md + De_Me;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generalised_geodesic2d", &generalised_geodesic2d, "Generalised Geodesic distance 2d");
    m.def("GSF2d", &GSF2d, "Geodesic Symmetric Filtering 2d");
    m.def("generalised_geodesic3d", &generalised_geodesic3d, "Generalised Geodesic distance 3d");
    m.def("GSF3d", &GSF3d, "Geodesic Symmetric Filtering 3d");
}