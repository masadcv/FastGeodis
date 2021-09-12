#ifndef FASTGEODIS_CPU_H_
#define FASTGEODIS_CPU_H_

#include <torch/extension.h>
#include <vector>

void generalised_geodesic2d_cpu(torch::Tensor &image, torch::Tensor &distance, const float &l_grad, const float &l_eucl, const int &iterations);

void generalised_geodesic3d_cpu(torch::Tensor &image, torch::Tensor &distance, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl, const int &iterations);

#endif