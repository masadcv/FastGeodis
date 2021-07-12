#include <torch/extension.h>
#include <vector>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#define VERBOSE 0

using namespace torch::indexing;

void print_shape(torch::Tensor data)
{
    auto num_dims = data.dim();
    std::cout << "Shape: (";
    for (int dim = 0; dim < num_dims; dim++)
    {
        std::cout << data.size(dim);
        if (dim != num_dims - 1)
            std::cout << ", ";
        else
            std::cout << ")" << std::endl;
    }
}

float l2distance(float in1, float in2)
{
    return std::abs(in1 - in2);
}

float l2distance(std::vector<float> in1, std::vector<float> in2)
{
    float retsum = 0.0;
    for (int ci = 0; ci < int(in1.size()); ci++)
        retsum += (in1[ci] - in2[ci]) * (in1[ci] - in2[ci]);
    return std::sqrt(retsum);
}

void geodesic_updown_pass(torch::Tensor image, torch::Tensor &distance, float lambda)
{
    // batch, channel, height, width
    int channel = image.size(1);
    int height = image.size(2);
    int width = image.size(3);

    auto imageptr = image.accessor<float, 4>();
    auto distanceptr = distance.accessor<float, 4>();
    float local_dist[] = {std::sqrt(2.0), 1.0, std::sqrt(2.0)};

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
            std::vector<float> pval_v;
            if (channel == 1)
                pval = imageptr[0][0][h][w];
            else
                for (int ci = 0; ci < channel; ci++)
                    pval_v.push_back(imageptr[0][ci][h][w]);
            float new_dist = distanceptr[0][0][h][w];

            for (int wi = 0; wi < 3; wi++)
            {
                int wind = w + wi - 1;
                if (wind < 0 || wind >= width)
                    continue;

                float ldist;
                if (channel == 1)
                    ldist = l2distance(pval, imageptr[0][0][h - 1][wind]);
                else
                {
                    std::vector<float> qval_v;
                    for (int ci = 0; ci < channel; ci++)
                        qval_v.push_back(imageptr[0][ci][h - 1][wind]);
                    ldist = l2distance(pval_v, qval_v);
                }
                float cur_dist = distanceptr[0][0][h - 1][wind] + local_dist[wi] / ((1.0 - lambda) + lambda / (ldist + 1e-5));
                new_dist = std::min(new_dist, cur_dist);
            }
            distanceptr[0][0][h][w] = new_dist;
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
            std::vector<float> pval_v;
            if (channel == 1)
                pval = imageptr[0][0][h][w];
            else
                for (int ci = 0; ci < channel; ci++)
                    pval_v.push_back(imageptr[0][ci][h][w]);
            float new_dist = distanceptr[0][0][h][w];

            for (int wi = 0; wi < 3; wi++)
            {
                int wind = w + wi - 1;
                if (wind < 0 || wind >= width)
                    continue;

                float ldist;
                if (channel == 1)
                    ldist = l2distance(pval, imageptr[0][0][h + 1][wind]);
                else
                {
                    std::vector<float> qval_v;
                    for (int ci = 0; ci < channel; ci++)
                        qval_v.push_back(imageptr[0][ci][h + 1][wind]);
                    ldist = l2distance(pval_v, qval_v);
                }
                float cur_dist = distanceptr[0][0][h + 1][wind] + local_dist[wi] / ((1.0 - lambda) + lambda / (ldist + 1e-5));
                new_dist = std::min(new_dist, cur_dist);
            }
            distanceptr[0][0][h][w] = new_dist;
        }
    }
}

torch::Tensor generalised_geodesic2d(torch::Tensor image, torch::Tensor mask, float v, float lambda, int iterations)
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
    int num_dims = distance.dim();
    if (num_dims != 4)
    {
        throw std::runtime_error(
            "function only supports 2D spatial inputs, received " + std::to_string(num_dims - 2));
    }

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        // top-bottom - width*, height
        geodesic_updown_pass(image, distance, lambda);

        // left-right - height*, width
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);
        geodesic_updown_pass(image, distance, lambda);
        
        // tranpose back to original - width, height
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }
    return distance;
}

void geodesic_frontback_pass(torch::Tensor image, torch::Tensor &distance, std::vector<float> spacingsq, float lambda)
{
    // batch, channel, depth, height, width
    int channel = image.size(1);
    int depth = image.size(2);
    int height = image.size(3);
    int width = image.size(4);

    auto imageptr = image.accessor<float, 5>();
    auto distanceptr = distance.accessor<float, 5>();

    std::vector<float> local_dist;
    for (int hi = -1; hi < 2; hi++)
    {
        for (int wi = -1; wi < 2; wi++)
        {
            float ld = spacingsq[0];
            ld += float(std::abs(hi)) * spacingsq[1];
            ld += float(std::abs(wi)) * spacingsq[2];

            local_dist.push_back(std::sqrt(ld));
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
                std::vector<float> pval_v;
                if (channel == 1)
                    pval = imageptr[0][0][z][h][w];
                else
                    for (int ci = 0; ci < channel; ci++)
                        pval_v.push_back(imageptr[0][ci][z][h][w]);
                float new_dist = distanceptr[0][0][z][h][w];

                for (int hi = 0; hi < 3; hi++)
                {
                    for (int wi = 0; wi < 3; wi++)
                    {
                        int hind = h + hi - 1;
                        int wind = w + wi - 1;

                        if (wind < 0 || wind >= width || hind < 0 || hind >= height)
                            continue;

                        float ldist;
                        if (channel == 1)
                            ldist = std::abs(pval - imageptr[0][0][z - 1][hind][wind]);
                        else
                        {
                            std::vector<float> qval_v;
                            for (int ci = 0; ci < channel; ci++)
                                qval_v.push_back(imageptr[0][ci][z - 1][hind][wind]);
                            ldist = l2distance(pval_v, qval_v);
                        }
                        float cur_dist = distanceptr[0][0][z - 1][hind][wind] + local_dist[hi * 3 + wi] / ((1.0 - lambda) + lambda / (ldist + 1e-5));
                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
                distanceptr[0][0][z][h][w] = new_dist;
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
                std::vector<float> pval_v;
                if (channel == 1)
                    pval = imageptr[0][0][z][h][w];
                else
                    for (int ci = 0; ci < channel; ci++)
                        pval_v.push_back(imageptr[0][ci][z][h][w]);
                float new_dist = distanceptr[0][0][z][h][w];

                for (int hi = 0; hi < 3; hi++)
                {
                    for (int wi = 0; wi < 3; wi++)
                    {
                        int hind = h + hi - 1;
                        int wind = w + wi - 1;

                        if (wind < 0 || wind >= width || hind < 0 || hind >= height)
                            continue;

                        float ldist;
                        if (channel == 1)
                            ldist = std::abs(pval - imageptr[0][0][z + 1][hind][wind]);
                        else
                        {
                            std::vector<float> qval_v;
                            for (int ci = 0; ci < channel; ci++)
                                qval_v.push_back(imageptr[0][ci][z + 1][hind][wind]);
                            ldist = l2distance(pval_v, qval_v);
                        }
                        float cur_dist = distanceptr[0][0][z + 1][hind][wind] + local_dist[hi * 3 + wi] / ((1.0 - lambda) + lambda / (ldist + 1e-5));
                        new_dist = std::min(new_dist, cur_dist);
                    }
                }
                distanceptr[0][0][z][h][w] = new_dist;
            }
        }
    }
}

torch::Tensor generalised_geodesic3d(torch::Tensor image, torch::Tensor mask, std::vector<float> spacing, float v, float lambda, int iterations)
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
    int num_dims = distance.dim();
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

    // square spacing
    for (int t = 0; t < int(spacing.size()); t++)
        spacing[t] = spacing[t] * spacing[t];

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        // front-back - depth*, height, width
        geodesic_frontback_pass(image, distance, spacing, lambda);

        // top-bottom - height*, depth, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        geodesic_frontback_pass(image, distance, spacing, lambda);
        // transpose back to original depth, height, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);

        // left-right - width*, height, depth
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);
        geodesic_frontback_pass(image, distance, spacing, lambda);
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