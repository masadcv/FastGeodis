#include <torch/extension.h>
#include <vector>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 0

void print_shape(torch::Tensor data)
{
    auto num_dims = data.dim();
    std::cout << "Shape: (";
    for (int dim = 0; dim < num_dims; dim++)
    {
        std::cout << data.size(dim);
        if (dim != num_dims - 1)
        {
            std::cout << ", ";
        }
        else
        {
            std::cout << ")" << std::endl;
        }
    }
}

float l1distance(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}


float l1distance(const float *in1, const float *in2, int size)
{
    float ret_sum = 0.0;
    for (int c_i = 0; c_i < size; c_i++)
    {
        ret_sum += abs(in1[c_i] - in2[c_i]);
    }
    return ret_sum;
}



// float l2distance(const float *in1, const float *in2, int size)
// {
//     float ret_sum = 0.0;
//     for (int c_i = 0; c_i < size; c_i++)
//     {
//         ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
//     }
//     return std::sqrt(ret_sum);
// }


void geodesic_updown_pass(const torch::Tensor &image, torch::Tensor &distance, const float &l_grad,  const float &l_eucl)
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    auto image_ptr = image.accessor<float, 4>();
    auto distance_ptr = distance.accessor<float, 4>();
    constexpr float local_dist[] = {sqrt(2.), 1., sqrt(2.)};

    // top-down
    for (int h = 1; h < height; h++)
    {
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (int w = 0; w < width; w++)
        {
            float pval;
            float pval_v[channel];
            if (channel == 1)
            {
                pval = image_ptr[0][0][h][w];
            }
            else
            {
                for (int c_i = 0; c_i < channel; c_i++)
                {
                    pval_v[c_i] = image_ptr[0][c_i][h][w];
                }
            }
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;
                if (w_ind < 0 || w_ind >= width)
                    continue;

                float l_dist;
                if (channel == 1)
                {
                    l_dist = l1distance(pval, image_ptr[0][0][h - 1][w_ind]);
                }
                else
                {
                    float qval_v[channel];
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        qval_v[c_i] = image_ptr[0][c_i][h - 1][w_ind];
                    }
                    l_dist = l1distance(pval_v, qval_v, channel);
                }
                const float cur_dist = distance_ptr[0][0][h - 1][w_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }

    // bottom-up
    for (int h = height - 2; h >= 0; h--)
    {
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
            #pragma omp parallel for
        #endif
        for (int w = 0; w < width; w++)
        {
            float pval;
            float pval_v[channel];
            if (channel == 1)
            {
                pval = image_ptr[0][0][h][w];
            }
            else
            {
                for (int c_i = 0; c_i < channel; c_i++)
                {
                    pval_v[c_i] = image_ptr[0][c_i][h][w];
                }
            }
            float new_dist = distance_ptr[0][0][h][w];

            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int w_ind = w + w_i - 1;
                if (w_ind < 0 || w_ind >= width)
                    continue;

                float l_dist;
                if (channel == 1)
                {
                    l_dist = l1distance(pval, image_ptr[0][0][h + 1][w_ind]);
                }
                else
                {
                    float qval_v[channel];
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        qval_v[c_i] = image_ptr[0][c_i][h + 1][w_ind];
                    }
                    l_dist = l1distance(pval_v, qval_v, channel);
                }
                const float cur_dist = distance_ptr[0][0][h + 1][w_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
                new_dist = std::min(new_dist, cur_dist);
            }
            distance_ptr[0][0][h][w] = new_dist;
        }
    }
}

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

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image = image.contiguous();
        distance = distance.contiguous();

        // top-bottom - width*, height
        geodesic_updown_pass(image, distance, l_grad, l_eucl);

        // left-right - height*, width
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_updown_pass(image, distance, l_grad, l_eucl);
        
        // tranpose back to original - width, height
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }
    return distance;
}

void geodesic_frontback_pass(const torch::Tensor &image, torch::Tensor &distance, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    // batch, channel, depth, height, width
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    auto image_ptr = image.accessor<float, 5>();
    auto distance_ptr = distance.accessor<float, 5>();

    float local_dist[3*3];
    for (int h_i = 0; h_i < 3; h_i++)
    {
        for (int w_i = 0; w_i < 3; w_i++)
        {
            float ld = spacing[0];
            ld += float(std::abs(h_i-1)) * spacing[1];
            ld += float(std::abs(w_i-1)) * spacing[2];

            local_dist[h_i * 3 + w_i] = ld;
        }
    }

    // front-back
    for (int z = 1; z < depth; z++)
    {
        // use openmp to parallelise the loops over height and width
        #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
        #endif
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float pval;
                float pval_v[channel];
                if (channel == 1)
                {
                    pval = image_ptr[0][0][z][h][w];
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        pval_v[c_i] = image_ptr[0][c_i][z][h][w];
                    }
                }
                float new_dist = distance_ptr[0][0][z][h][w];

                for (int h_i = 0; h_i < 3; h_i++)
                {
                    for (int w_i = 0; w_i < 3; w_i++)
                    {
                        const int h_ind = h + h_i - 1;
                        const int w_ind = w + w_i - 1;

                        if (w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                            continue;

                        float l_dist;
                        if (channel == 1)
                        {
                            l_dist = std::abs(pval - image_ptr[0][0][z - 1][h_ind][w_ind]);
                        }
                        else
                        {
                            float qval_v[channel];
                            for (int c_i = 0; c_i < channel; c_i++)
                            {
                                qval_v[c_i] = image_ptr[0][c_i][z - 1][h_ind][w_ind];
                            }
                            l_dist = l1distance(pval_v, qval_v, channel);
                        }
                        const float cur_dist = distance_ptr[0][0][z - 1][h_ind][w_ind] + l_eucl * local_dist[h_i * 3 + w_i]  + l_grad * l_dist;
                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
                distance_ptr[0][0][z][h][w] = new_dist;
            }
        }
    }

    // back-front
    for (int z = depth - 2; z >= 0; z--)
    {
        // use openmp to parallelise the loops over height and width
        #ifdef _OPENMP
            #pragma omp parallel for collapse(2)
        #endif
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                float pval;
                float pval_v[channel];
                if (channel == 1)
                {
                    pval = image_ptr[0][0][z][h][w];
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        pval_v[c_i] = image_ptr[0][c_i][z][h][w];
                    }
                }
                float new_dist = distance_ptr[0][0][z][h][w];

                for (int h_i = 0; h_i < 3; h_i++)
                {
                    for (int w_i = 0; w_i < 3; w_i++)
                    {
                        const int h_ind = h + h_i - 1;
                        const int w_ind = w + w_i - 1;

                        if (w_ind < 0 || w_ind >= width || h_ind < 0 || h_ind >= height)
                            continue;

                        float l_dist;
                        if (channel == 1)
                        {
                            l_dist = std::abs(pval - image_ptr[0][0][z + 1][h_ind][w_ind]);
                        }
                        else
                        {
                            float qval_v[channel];
                            for (int c_i = 0; c_i < channel; c_i++)
                            {
                                qval_v[c_i] = image_ptr[0][c_i][z + 1][h_ind][w_ind];
                            }
                            l_dist = l1distance(pval_v, qval_v, channel);
                        }
                        const float cur_dist = distance_ptr[0][0][z + 1][h_ind][w_ind] + l_eucl * local_dist[h_i * 3 + w_i] + l_grad * l_dist;
                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
                distance_ptr[0][0][z][h][w] = new_dist;
            }
        }
    }
}

torch::Tensor generalised_geodesic3d(torch::Tensor image, const torch::Tensor &mask, std::vector<float> spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
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

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image = image.contiguous();
        distance = distance.contiguous();

        // front-back - depth*, height, width
        geodesic_frontback_pass(image, distance, spacing, l_grad, l_eucl);

        // top-bottom - height*, depth, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        
        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass(image, distance, {spacing[1], spacing[0], spacing[2]}, l_grad, l_eucl);
        
        // transpose back to original depth, height, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        
        // left-right - width*, height, depth
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);
        
        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass(image, distance, {spacing[2], spacing[1], spacing[0]}, l_grad, l_eucl);
        
        // transpose back to original depth, height, width
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);

        // * indicates the current direction of pass
    }
    return distance;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("generalised_geodesic2d", &generalised_geodesic2d, "Generalised Geodesic distance 2d");
    m.def("generalised_geodesic3d", &generalised_geodesic3d, "Generalised Geodesic distance 3d");
}