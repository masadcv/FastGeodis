#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "fastgeodis.h"
#include "common.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 1

torch::Tensor generalised_geodesic2d(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "OpenMP found, using OpenMP" << std::endl;
        #else
            std::cout << "OpenMP not found" << std::endl;
        #endif
        #ifdef WITH_CUDA
            std::cout << "Compiled with CUDA" << std::endl;
        #else
            std::cout << "Not Compiled with CUDA" << std::endl;
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
        // CHECK_CONTIGUOUS_CUDA(image);
        // CHECK_CONTIGUOUS_CUDA(mask);
        CHECK_CUDA(image);
        CHECK_CUDA(mask);

        return generalised_geodesic2d_cuda(image, mask, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with GPU support.");
    #endif
    }
    return generalised_geodesic2d_cpu(image, mask, v, l_grad, l_eucl, iterations);
}

torch::Tensor generalised_geodesic3d(torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "OpenMP found, using OpenMP" << std::endl;
        #else
            std::cout << "OpenMP not found" << std::endl;
        #endif
        #ifdef WITH_CUDA
            std::cout << "Compiled with CUDA" << std::endl;
        #else
            std::cout << "Not Compiled with CUDA" << std::endl;
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
        // CHECK_CONTIGUOUS_CUDA(image);
        // CHECK_CONTIGUOUS_CUDA(mask);
        CHECK_CUDA(image);
        CHECK_CUDA(mask);
        
        return generalised_geodesic3d_cuda(image, mask, spacing, v, l_grad, l_eucl, iterations);

    #else
        AT_ERROR("Not compiled with GPU support.");
    #endif
    }
    return generalised_geodesic3d_cpu(image, mask, spacing, v, l_grad, l_eucl, iterations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generalised_geodesic2d", &generalised_geodesic2d, "Generalised Geodesic distance 2d");
    m.def("generalised_geodesic3d", &generalised_geodesic3d, "Generalised Geodesic distance 3d");
}