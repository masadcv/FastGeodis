#pragma once

#include <torch/extension.h>
#include <vector>
#include "common.h"

#ifdef WITH_CUDA
torch::Tensor generalised_geodesic2d_cuda(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations);

torch::Tensor generalised_geodesic3d_cuda(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations);
#endif

torch::Tensor generalised_geodesic2d_cpu(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations);

torch::Tensor generalised_geodesic3d_cpu(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations);

torch::Tensor generalised_geodesic2d(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations
    );

torch::Tensor generalised_geodesic3d(
    torch::Tensor &image, 
    const torch::Tensor &mask, 
    const std::vector<float> &spacing, 
    const float &v, 
    const float &l_grad, 
    const float &l_eucl, 
    const int &iterations
    );


