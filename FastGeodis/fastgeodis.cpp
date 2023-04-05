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
#include <future>
#include <vector>
#include "fastgeodis.h"
#include "common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 0

torch::Tensor generalised_geodesic2d(const torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
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
    check_input_dimensions(image, mask, 4);
    
    // const int num_dims = mask.dim();
    // if (num_dims != 4)
    // {
    //     throw std::invalid_argument(
    //         "function only supports 2D spatial inputs, received " + std::to_string(num_dims - 2));
    // }

    if (image.is_cuda()) 
    {
    #ifdef WITH_CUDA
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error(
                "cuda.is_available() returned false, please check if the library was compiled successfully with CUDA support");
        }
        check_cuda(mask);

        return generalised_geodesic2d_cuda(image, mask, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with CUDA support.");
    #endif
    }
    else
    {
        check_cpu(mask);
    }
    return generalised_geodesic2d_cpu(image, mask, v, l_grad, l_eucl, iterations);
}

torch::Tensor generalised_geodesic3d(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
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
    check_input_dimensions(image, mask, 5);

    // const int num_dims = mask.dim();
    // if (num_dims != 5)
    // {
    //     throw std::invalid_argument(
    //         "function only supports 3D spatial inputs, received " + std::to_string(num_dims - 2));
    // }
    if (spacing.size() != 3)
    {
        throw std::invalid_argument(
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
        check_cuda(mask);
        
        return generalised_geodesic3d_cuda(image, mask, spacing, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with CUDA support.");
    #endif
    }
    else
    {
        check_cpu(mask);
    }
    return generalised_geodesic3d_cpu(image, mask, spacing, v, l_grad, l_eucl, iterations);
}

torch::Tensor generalised_geodesic2d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{

    // check input dimensions
    check_input_dimensions(image, mask, 4);

    // toivanen method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    return generalised_geodesic2d_toivanen_cpu(image, mask, v, l_grad, l_eucl, iterations);
}

torch::Tensor generalised_geodesic3d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    // check input dimensions
    check_input_dimensions(image, mask, 5);

    // toivanen method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    if (spacing.size() != 3)
    {
        throw std::invalid_argument(
            "function only supports 3D spacing inputs, received " + std::to_string(spacing.size()));
    }

    return generalised_geodesic3d_toivanen_cpu(image, mask, spacing, v, l_grad, l_eucl, iterations);
}

torch::Tensor geodesic2d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const float &l_grad, const float &l_eucl)
{

    // check input dimensions
    check_input_dimensions(image, mask, 4);

    // pixelqueue method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    return geodesic2d_pixelqueue_cpu(image, mask, l_grad, l_eucl);
}

torch::Tensor geodesic3d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    // check input dimensions
    check_input_dimensions(image, mask, 5);

    // pixelqueue method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    if (spacing.size() != 3)
    {
        throw std::invalid_argument(
            "function only supports 3D spacing inputs, received " + std::to_string(spacing.size()));
    }

    return geodesic3d_pixelqueue_cpu(image, mask, spacing, l_grad, l_eucl);
}

torch::Tensor geodesic2d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const float &l_grad, const float &l_eucl)
{

    // check input dimensions
    check_input_dimensions(image, mask, 4);

    // fastmarch method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    return geodesic2d_fastmarch_cpu(image, mask, l_grad, l_eucl);
}

torch::Tensor geodesic3d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    // check input dimensions
    check_input_dimensions(image, mask, 5);

    // fastmarch method is only implementable on cpu
    check_cpu(image);    
    check_cpu(mask);

    if (spacing.size() != 3)
    {
        throw std::invalid_argument(
            "function only supports 3D spacing inputs, received " + std::to_string(spacing.size()));
    }

    return geodesic3d_fastmarch_cpu(image, mask, spacing, l_grad, l_eucl);
}

torch::Tensor signed_generalised_geodesic2d(const torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    auto secondcall = std::async(std::launch::async, generalised_geodesic2d, image, 1 - mask, v, l_grad, l_eucl, iterations);
    torch::Tensor D_M = generalised_geodesic2d(image, mask, v, l_grad, l_eucl, iterations);
    // torch::Tensor D_Mb = generalised_geodesic2d(image, 1 - mask, v, l_grad, l_eucl, iterations);
    torch::Tensor D_Mb = secondcall.get();

    return D_M - D_Mb;
}

torch::Tensor signed_generalised_geodesic3d(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    auto secondcall = std::async(std::launch::async, generalised_geodesic3d, image, 1 - mask, spacing, v, l_grad, l_eucl, iterations);
    torch::Tensor D_M = generalised_geodesic3d(image, mask, spacing, v, l_grad, l_eucl, iterations);
    // torch::Tensor D_Mb = generalised_geodesic3d(image, 1 - mask, spacing, v, l_grad, l_eucl, iterations);
    torch::Tensor D_Mb = secondcall.get();
    return D_M - D_Mb;
}

torch::Tensor signed_generalised_geodesic2d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    auto secondcall = std::async(std::launch::async, generalised_geodesic2d_toivanen, image, 1 - mask, v, l_grad, l_eucl, iterations);
    torch::Tensor D_M = generalised_geodesic2d_toivanen(image, mask, v, l_grad, l_eucl, iterations);
    // torch::Tensor D_Mb = generalised_geodesic2d_toivanen(image, 1 - mask, v, l_grad, l_eucl, iterations);
    torch::Tensor D_Mb = secondcall.get();
    return D_M - D_Mb;
}

torch::Tensor signed_generalised_geodesic3d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    auto secondcall = std::async(std::launch::async, generalised_geodesic3d_toivanen, image, 1 - mask, spacing, v, l_grad, l_eucl, iterations);
    torch::Tensor D_M = generalised_geodesic3d_toivanen(image, mask, spacing, v, l_grad, l_eucl, iterations);
    // torch::Tensor D_Mb = generalised_geodesic3d_toivanen(image, 1 - mask, spacing, v, l_grad, l_eucl, iterations);
    torch::Tensor D_Mb = secondcall.get();
    
    return D_M - D_Mb;
}

torch::Tensor signed_geodesic2d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const float &l_grad, const float &l_eucl)
{
    auto secondcall = std::async(std::launch::async, geodesic2d_pixelqueue, image, 1 - mask, l_grad, l_eucl);
    torch::Tensor D_M = geodesic2d_pixelqueue(image, mask, l_grad, l_eucl);
    // torch::Tensor D_Mb = geodesic2d_pixelqueue(image, 1 - mask, l_grad, l_eucl);
    torch::Tensor D_Mb = secondcall.get();

    return D_M - D_Mb;
}

torch::Tensor signed_geodesic3d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    auto secondcall = std::async(std::launch::async, geodesic3d_pixelqueue, image, 1 - mask, spacing, l_grad, l_eucl);
    torch::Tensor D_M = geodesic3d_pixelqueue(image, mask, spacing, l_grad, l_eucl);
    // torch::Tensor D_Mb = geodesic3d_pixelqueue(image, 1 - mask, spacing, l_grad, l_eucl);
    torch::Tensor D_Mb = secondcall.get();

    return D_M - D_Mb;
}

torch::Tensor signed_geodesic2d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const float &l_grad, const float &l_eucl)
{
    auto secondcall = std::async(std::launch::async, geodesic2d_fastmarch, image, 1 - mask, l_grad, l_eucl);
    torch::Tensor D_M = geodesic2d_fastmarch(image, mask, l_grad, l_eucl);
    // torch::Tensor D_Mb = geodesic2d_fastmarch(image, 1 - mask, l_grad, l_eucl);
    torch::Tensor D_Mb = secondcall.get();

    return D_M - D_Mb;
}

torch::Tensor signed_geodesic3d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    auto secondcall = std::async(std::launch::async, geodesic3d_fastmarch, image, 1 - mask, spacing, l_grad, l_eucl);
    torch::Tensor D_M = geodesic3d_fastmarch(image, mask, spacing, l_grad, l_eucl);
    // torch::Tensor D_Mb = geodesic3d_fastmarch(image, 1 - mask, spacing, l_grad, l_eucl);
    torch::Tensor D_Mb = secondcall.get();

    return D_M - D_Mb;
}

torch::Tensor GSF2d(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = signed_generalised_geodesic2d(image, mask, v, lambda, 1 - lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_generalised_geodesic2d(image, 1 - Md, v, lambda, 1 - lambda, iterations);
    torch::Tensor De_Me = signed_generalised_geodesic2d(image, Me, v, lambda, 1 - lambda, iterations);

    return Dd_Md + De_Me;
}

torch::Tensor GSF3d(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const std::vector<float> &spacing, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = signed_generalised_geodesic3d(image, mask, spacing, v, lambda, 1 - lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_generalised_geodesic3d(image, 1 - Md, spacing, v, lambda, 1 - lambda, iterations);
    torch::Tensor De_Me = signed_generalised_geodesic3d(image, Me, spacing, v, lambda, 1 - lambda, iterations);

    return Dd_Md + De_Me;
}

torch::Tensor GSF2d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = signed_generalised_geodesic2d_toivanen(image, mask, v, lambda, 1 - lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_generalised_geodesic2d_toivanen(image, 1 - Md, v, lambda, 1 - lambda, iterations);
    torch::Tensor De_Me = signed_generalised_geodesic2d_toivanen(image, Me, v, lambda, 1 - lambda, iterations);

    return Dd_Md + De_Me;
}

torch::Tensor GSF3d_toivanen(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const std::vector<float> &spacing, const float &v, const float &lambda, const int &iterations)
{
    torch::Tensor Ds_M = signed_generalised_geodesic3d_toivanen(image, mask, spacing, v, lambda, 1 - lambda, iterations);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_generalised_geodesic3d_toivanen(image, 1 - Md, spacing, v, lambda, 1 - lambda, iterations);
    torch::Tensor De_Me = signed_generalised_geodesic3d_toivanen(image, Me, spacing, v, lambda, 1 - lambda, iterations);

    return Dd_Md + De_Me;
}

torch::Tensor GSF2d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const float &lambda)
{
    torch::Tensor Ds_M = signed_geodesic2d_pixelqueue(image, mask, lambda, 1 - lambda);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_geodesic2d_pixelqueue(image, 1 - Md, lambda, 1 - lambda);
    torch::Tensor De_Me = signed_geodesic2d_pixelqueue(image, Me, lambda, 1 - lambda);

    return Dd_Md + De_Me;
}

torch::Tensor GSF3d_pixelqueue(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const std::vector<float> &spacing, const float &lambda)
{
    torch::Tensor Ds_M = signed_geodesic3d_pixelqueue(image, mask, spacing, lambda, 1 - lambda);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_geodesic3d_pixelqueue(image, 1 - Md, spacing, lambda, 1 - lambda);
    torch::Tensor De_Me = signed_geodesic3d_pixelqueue(image, Me, spacing, lambda, 1 - lambda);

    return Dd_Md + De_Me;
}

torch::Tensor GSF2d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const float &lambda)
{
    torch::Tensor Ds_M = signed_geodesic2d_fastmarch(image, mask, lambda, 1 - lambda);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_geodesic2d_fastmarch(image, 1 - Md, lambda, 1 - lambda);
    torch::Tensor De_Me = signed_geodesic2d_fastmarch(image, Me, lambda, 1 - lambda);

    return Dd_Md + De_Me;
}

torch::Tensor GSF3d_fastmarch(const torch::Tensor &image, const torch::Tensor &mask, const float &theta, const std::vector<float> &spacing, const float &lambda)
{
    torch::Tensor Ds_M = signed_geodesic3d_fastmarch(image, mask, spacing, lambda, 1 - lambda);

    torch::Tensor Md = (Ds_M > theta).type_as(Ds_M);
    torch::Tensor Me = (Ds_M > -theta).type_as(Ds_M);

    torch::Tensor Dd_Md = -signed_geodesic3d_fastmarch(image, 1 - Md, spacing, lambda, 1 - lambda);
    torch::Tensor De_Me = signed_geodesic3d_fastmarch(image, Me, spacing, lambda, 1 - lambda);

    return Dd_Md + De_Me;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generalised_geodesic2d", &generalised_geodesic2d, "Generalised Geodesic distance 2d");
    m.def("generalised_geodesic3d", &generalised_geodesic3d, "Generalised Geodesic distance 3d");
    m.def("generalised_geodesic2d_toivanen", &generalised_geodesic2d_toivanen, "Generalised Geodesic distance 2d using Toivanen's method");
    m.def("generalised_geodesic3d_toivanen", &generalised_geodesic3d_toivanen, "Generalised Geodesic distance 3d using Toivanen's method");
    m.def("geodesic2d_pixelqueue", &geodesic2d_pixelqueue, "Geodesic distance 2d using Pixel Queue method");
    m.def("geodesic3d_pixelqueue", &geodesic3d_pixelqueue, "Geodesic distance 3d using Pixel Queue method");
    m.def("geodesic2d_fastmarch", &geodesic2d_fastmarch, "Geodesic distance 2d using Fast Marching method");
    m.def("geodesic3d_fastmarch", &geodesic3d_fastmarch, "Geodesic distance 3d using Fast Marching method");

    m.def("signed_generalised_geodesic2d", &signed_generalised_geodesic2d, "Signed Generalised Geodesic distance 2d");
    m.def("signed_generalised_geodesic3d", &signed_generalised_geodesic3d, "Signed Generalised Geodesic distance 3d");
    m.def("signed_generalised_geodesic2d_toivanen", &signed_generalised_geodesic2d_toivanen, "Signed Generalised Geodesic distance 2d using Toivanen's method");
    m.def("signed_generalised_geodesic3d_toivanen", &signed_generalised_geodesic3d_toivanen, "Signed Generalised Geodesic distance 3d using Toivanen's method");
    m.def("signed_geodesic2d_pixelqueue", &signed_geodesic2d_pixelqueue, "Signed Geodesic distance 2d using Pixel Queue method");
    m.def("signed_geodesic3d_pixelqueue", &signed_geodesic3d_pixelqueue, "Signed Geodesic distance 3d using Pixel Queue method");
    m.def("signed_geodesic2d_fastmarch", &signed_geodesic2d_fastmarch, "Signed Geodesic distance 2d using Fast Marching method");
    m.def("signed_geodesic3d_fastmarch", &signed_geodesic3d_fastmarch, "Signed Geodesic distance 3d using Fast Marching method");

    m.def("GSF2d", &GSF2d, "Geodesic Symmetric Filtering 2d");
    m.def("GSF3d", &GSF3d, "Geodesic Symmetric Filtering 3d");
    m.def("GSF2d_toivanen", &GSF2d_toivanen, "Geodesic Symmetric Filtering 2d using Toivanen's method");
    m.def("GSF3d_toivanen", &GSF3d_toivanen, "Geodesic Symmetric Filtering 3d using Toivanen's method");
    m.def("GSF2d_pixelqueue", &GSF2d_pixelqueue, "Geodesic Symmetric Filtering 2d using Pixel Queue method");
    m.def("GSF3d_pixelqueue", &GSF3d_pixelqueue, "Geodesic Symmetric Filtering 3d using Pixel Queue method");
    m.def("GSF2d_fastmarch", &GSF2d_fastmarch, "Geodesic Symmetric Filtering 2d using Fast Marching method");
    m.def("GSF3d_fastmarch", &GSF3d_fastmarch, "Geodesic Symmetric Filtering 3d using Fast Marching method");
    
}