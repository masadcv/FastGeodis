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

float l1distance_toivanen(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}

void geodesic2d_forback_toivanen_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    auto image_ptr = image.accessor<float, 4>();
    auto distance_ptr = distance.accessor<float, 4>();

    // forward
    const int dh_f[4] = {-1, -1, -1, 0};
    const int dw_f[4] = {-1, 0, 1, -1};

    const float local_dist_f[] = {sqrt(float(2.)), float(1.), sqrt(float(2.)), float(1.)};

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int ind = 0; ind < 4; ind++)
            {
                const int h_ind = h + dh_f[ind];
                const int w_ind = w + dw_f[ind];

                if (w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                    continue;

                l_dist = 0.0;
                if (channel == 1)
                {
                    l_dist = l1distance_toivanen(
                        image_ptr[0][0][h][w],
                        image_ptr[0][0][h_ind][w_ind]);
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        l_dist += l1distance_toivanen(
                            image_ptr[0][c_i][h][w],
                            image_ptr[0][c_i][h_ind][w_ind]);
                    }
                }
                cur_dist = distance_ptr[0][0][h_ind][w_ind] +
                           l_eucl * local_dist_f[ind] +
                           l_grad * l_dist;

                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }

    // backward
    const int dh_b[4] = {0, 1, 1, 1};
    const int dw_b[4] = {1, -1, 0, 1};

    const float local_dist_b[] = {float(1.), sqrt(float(2.)), float(1.), sqrt(float(2.))};

    for (int h = height - 1; h >= 0; h--)
    {
        for (int w = width - 1; w >= 0; w--)
        {
            float l_dist, cur_dist;
            float new_dist = distance_ptr[0][0][h][w];

            for (int ind = 0; ind < 4; ind++)
            {
                const int h_ind = h + dh_b[ind];
                const int w_ind = w + dw_b[ind];

                if (w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                    continue;

                l_dist = 0;
                if (channel == 1)
                {
                    l_dist = l1distance_toivanen(
                        image_ptr[0][0][h][w],
                        image_ptr[0][0][h_ind][w_ind]);
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        l_dist += l1distance_toivanen(
                            image_ptr[0][c_i][h][w],
                            image_ptr[0][c_i][h_ind][w_ind]);
                    }
                }
                cur_dist = distance_ptr[0][0][h_ind][w_ind] +
                           l_eucl * local_dist_b[ind] +
                           l_grad * l_dist;

                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }
}

torch::Tensor generalised_geodesic2d_toivanen_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations)
{
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        geodesic2d_forback_toivanen_cpu(image, distance, l_grad, l_eucl);
    }

    return distance;
}

void geodesic3d_forback_toivanen_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    const std::vector<float> &spacing,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, depth, height, width
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    auto image_ptr = image.accessor<float, 5>();
    auto distance_ptr = distance.accessor<float, 5>();

    // distances for forward
    const int dz_f[13] = {-1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1};
    const int dh_f[13] = {-1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0};
    const int dw_f[13] = {-1, 0, 1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1};

    float local_dist_f[13];
    for (int i = 0; i < 13; i++)
    {
        float ld = 0.0;
        if (dz_f[i] != 0)
        {
            ld += spacing[0] * spacing[0];
        }

        if (dh_f[i] != 0)
        {
            ld += spacing[1] * spacing[1];
        }

        if (dw_f[i] != 0)
        {
            ld += spacing[2] * spacing[2];
        }

        local_dist_f[i] = sqrt(ld);
    }

    // distances for backward
    const int dz_b[13] = {-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    const int dh_b[13] = {0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1};
    const int dw_b[13] = {1, -1, 0, 1, 1, -1, 0, 1, 0, 1, -1, 0, 1};

    float local_dist_b[13];
    for (int i = 0; i < 13; i++)
    {
        float ld = 0.0;
        if (dz_b[i] != 0)
        {
            ld += spacing[0] * spacing[0];
        }

        if (dh_b[i] != 0)
        {
            ld += spacing[1] * spacing[1];
        }

        if (dw_b[i] != 0)
        {
            ld += spacing[2] * spacing[2];
        }

        local_dist_b[i] = sqrt(ld);
    }

    // front-back
    for (int z = 0; z < depth; z++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float l_dist, cur_dist;
                float new_dist = distance_ptr[0][0][z][h][w];

                for (int ind = 0; ind < 13; ind++)
                {
                    const int z_ind = z + dz_f[ind];
                    const int h_ind = h + dh_f[ind];
                    const int w_ind = w + dw_f[ind];

                    if (z_ind < 0 || z_ind >= depth || w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                        continue;

                    l_dist = 0.0;
                    if (channel == 1)
                    {
                        l_dist = l1distance_toivanen(
                            image_ptr[0][0][z][h][w],
                            image_ptr[0][0][z_ind][h_ind][w_ind]);
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            l_dist += l1distance_toivanen(
                                image_ptr[0][c_i][z][h][w],
                                image_ptr[0][c_i][z_ind][h_ind][w_ind]);
                        }
                    }
                    cur_dist = distance_ptr[0][0][z_ind][h_ind][w_ind] +
                               l_eucl * local_dist_f[ind] +
                               l_grad * l_dist;

                    new_dist = std::min(new_dist, cur_dist);
                }
                distance_ptr[0][0][z][h][w] = new_dist;
            }
        }
    }

    // backward
    for (int z = depth - 1; z >= 0; z--)
    {
        for (int h = height - 1; h >= 0; h--)
        {
            for (int w = width - 1; w >= 0; w--)
            {
                float l_dist, cur_dist;
                float new_dist = distance_ptr[0][0][z][h][w];

                for (int ind = 0; ind < 13; ind++)
                {
                    const int z_ind = z + dz_b[ind];
                    const int h_ind = h + dh_b[ind];
                    const int w_ind = w + dw_b[ind];

                    if (z_ind < 0 || z_ind >= depth || w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                        continue;

                    l_dist = 0.0;
                    if (channel == 1)
                    {
                        l_dist = l1distance_toivanen(
                            image_ptr[0][0][z][h][w],
                            image_ptr[0][0][z_ind][h_ind][w_ind]);
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            l_dist += l1distance_toivanen(
                                image_ptr[0][c_i][z][h][w],
                                image_ptr[0][c_i][z_ind][h_ind][w_ind]);
                        }
                    }
                    cur_dist = distance_ptr[0][0][z_ind][h_ind][w_ind] +
                               l_eucl * local_dist_b[ind] +
                               l_grad * l_dist;

                    new_dist = std::min(new_dist, cur_dist);
                }
                distance_ptr[0][0][z][h][w] = new_dist;
            }
        }
    }
}

torch::Tensor generalised_geodesic3d_toivanen_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &v,
    const float &l_grad,
    const float &l_eucl,
    const int &iterations)
{
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        geodesic3d_forback_toivanen_cpu(image, distance, spacing, l_grad, l_eucl);
    }

    return distance;
}