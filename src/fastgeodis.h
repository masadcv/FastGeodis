#pragma once

#include <torch/extension.h>
#include <vector>
#include "common.h"

#ifdef WITH_CUDA
// std::vector<torch::Tensor> lltm_cuda_forward(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor bias,
//     torch::Tensor old_h,
//     torch::Tensor old_cell);

// std::vector<torch::Tensor> lltm_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights);
#endif

torch::Tensor generalised_geodesic2d_cpu(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations);
torch::Tensor generalised_geodesic3d_cpu(torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations);
