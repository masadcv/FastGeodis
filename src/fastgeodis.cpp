#include <torch/extension.h>
#include <vector>
#include <iostream>
#include "fastgeodis_cpu.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 0

torch::Tensor generalised_geodesic2d(torch::Tensor image, torch::Tensor mask, float v, float l_grad, float l_eucl, int iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "OpenMP found, using OpenMP" << std::endl;
        #else
            std::cout << "OpenMP not found" << std::endl;
        #endif
    #endif

    // initialise distance with soft mask
    torch::Tensor distance = v * mask.clone();

    // check input dimensions
    const int num_dims = distance.dim();
    if (num_dims != 4)
    {
        throw std::runtime_error(
            "function only supports 2D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    generalised_geodesic2d_cpu(image, distance, l_grad, l_eucl, iterations);
    
    return distance;
}

torch::Tensor generalised_geodesic3d(torch::Tensor image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    #if VERBOSE
        #ifdef _OPENMP
            std::cout << "OpenMP found, using OpenMP" << std::endl;
        #else
            std::cout << "OpenMP not found" << std::endl;
        #endif
    #endif

    // initialise distance with soft mask
    torch::Tensor distance = v * mask.clone();

    // check input dimensions
    const int num_dims = distance.dim();
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

    generalised_geodesic3d_cpu(image, distance, spacing, l_grad, l_eucl, iterations);

    return distance;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generalised_geodesic2d", &generalised_geodesic2d, "Generalised Geodesic distance 2d");
    m.def("generalised_geodesic3d", &generalised_geodesic3d, "Generalised Geodesic distance 3d");
}