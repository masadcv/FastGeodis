#include <torch/extension.h>
#include <vector>
// #include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#define TILE_DIM     24
#define THREAD_COUNT 512
#define STRIP_HEIGHT 16

// void print_shape(torch::Tensor data)
// {
//     auto num_dims = data.dim();
//     std::cout << "Shape: (";
//     for (int dim = 0; dim < num_dims; dim++)
//     {
//         std::cout << data.size(dim);
//         if (dim != num_dims - 1)
//         {
//             std::cout << ", ";
//         }
//         else
//         {
//             std::cout << ")" << std::endl;
//         }
//     }
// }

__device__ __forceinline__ float l1distance_cuda(const float &in1, const float &in2)
{
    return abs(in1 - in2);
}


float l1distance_cuda(const float *in1, const float *in2, int size)
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

template <typename scalar_t>
__global__ void geodesic_updown_pass_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> image_ptr, 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> distance_ptr,
    const float &l_grad, 
    const float &l_eucl,
    const float *local_dist,
    const int index,
    const int height,
    const int width)
{
    __shared__ float lastRow[THREAD_COUNT+2];
    __shared__ float curRow[THREAD_COUNT+2];

    // overlap
    int blockStartX = (blockIdx.x * THREAD_COUNT) - (blockIdx.x * STRIP_HEIGHT * 2);
    int blockSafeX = blockStartX + THREAD_COUNT - (2 * STRIP_HEIGHT);

    int kernelX = blockStartX + threadIdx.x;

    lastRow[threadIdx.x + 1] = distance_ptr[0][0][index-1][kernelX];
    curRow[threadIdx.x + 1] = distance_ptr[0][0][index][kernelX];
    if (threadIdx.x == 0)
    {
        lastRow[0] = -1;
        curRow[0] = -1;
        if ((kernelX - 1) >= 0)
        { 
            lastRow[0] = distance_ptr[0][0][index-1][kernelX-1];
            curRow[0] = distance_ptr[0][0][index][kernelX-1];
        }
    }
    else if(threadIdx.x == THREAD_COUNT - 1)
    {
        lastRow[threadIdx.x + 1] = -1;
        curRow[threadIdx.x + 1] = -1;
        if ((kernelX + 1) < width)
        {
            lastRow[threadIdx.x + 1] = distance_ptr[0][0][index-1][kernelX+1];
            curRow[threadIdx.x + 1] = distance_ptr[0][0][index][kernelX+1];
        }
    }

    __syncthreads();

    // top-down pass for each row in strip
    for (int i = 0; i < STRIP_HEIGHT; i++)
    {
        int currentHeightIdx = index + i;
        if (currentHeightIdx < height and kernelX < width)
        {
            int currentInnerIdx = (int)threadIdx.x + 1;
            float new_dist = -1;
            if (currentInnerIdx >= 0  && currentInnerIdx < THREAD_COUNT+2)
            {
                new_dist = curRow[currentInnerIdx];
            }
            float pval = image_ptr[0][0][currentHeightIdx][kernelX];

            for(int w_i = 0; w_i < 3; w_i++)
            {
                if (currentInnerIdx >= 0 && currentInnerIdx < THREAD_COUNT+2)
                {
                    const float cur_dist = lastRow[currentInnerIdx];
                    if (cur_dist >= 0)
                    {
                        const int w_ind = kernelX + w_i -1;
                        float qval = image_ptr[0][0][currentHeightIdx-1][w_ind];

                        float l_dist = l1distance_cuda(pval, qval);
                        new_dist = std::min(new_dist, (cur_dist + l_eucl * local_dist[w_i] + l_grad * l_dist));
                    }
                }
            }
            
            __syncthreads();
            if (new_dist >= 0.0 && kernelX < blockSafeX && (curRow[threadIdx.x] < 0.0 || new_dist < curRow[threadIdx.x]))
            {
                distance_ptr[0][0][currentHeightIdx][kernelX];
            }

            lastRow[threadIdx.x] = curRow[threadIdx.x];
            curRow[threadIdx.x] = -1;
            if((currentHeightIdx+1) < height)
            {
                curRow[threadIdx.x] = distance_ptr[0][0][currentHeightIdx + 1][kernelX];
            }
            
            if (threadIdx.x == 0)
            {
                curRow[0] = -1;
                if((kernelX - 1) >= 0 && (currentHeightIdx+1) < height)
                {
                    curRow[0] = distance_ptr[0][0][currentHeightIdx + 1][kernelX - 1];
                }
            }
            else if (threadIdx.x == THREAD_COUNT - 1)
            {
                curRow[THREAD_COUNT + 1] = -1;
                if((kernelX + 1) >= 0 && (currentHeightIdx+1) < height)
                {
                    curRow[THREAD_COUNT + 1] = distance_ptr[0][0][currentHeightIdx + 1][kernelX + 1];
                }
            }

            __syncthreads();
            
            // TODO: bottom up pass
        }
    }
    

}

void geodesic_updown_pass_cuda(const torch::Tensor &image, torch::Tensor &distance, const float &l_grad,  const float &l_eucl)
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    // auto image_ptr = image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    // auto distance_ptr = distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
    
    // constexpr float local_dist[] = {sqrt(2.), 1., sqrt(2.)};
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    int blockSafeZone = (THREAD_COUNT - (STRIP_HEIGHT * 2));
	int blockCountUpDown = width / blockSafeZone;
	int blockCountLeftRight = height / blockSafeZone;

	if (width % blockSafeZone != 0)
		blockCountUpDown++;
	if (height % blockSafeZone != 0)
		blockCountLeftRight++;

    // process each strip
    for (int i = 0; i < height; i += STRIP_HEIGHT)
    {
        // call kernel
        AT_DISPATCH_FLOATING_TYPES(image.scalar_type(), "geodesic_updown_pass_kernel", ([&] {
            geodesic_updown_pass_kernel<scalar_t><<<blockCountUpDown, THREAD_COUNT>>>(
                image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                l_grad, 
                l_eucl, 
                local_dist, 
                i, 
                height, 
                width);
            }));
    }

    // // top-down
    // for (int h = 1; h < height; h++)
    // {
    //     // use openmp to parallelise the loop over width
    //     #ifdef _OPENMP
    //         #pragma omp parallel for
    //     #endif
    //     for (int w = 0; w < width; w++)
    //     {
    //         float pval;
    //         float pval_v[channel];
    //         if (channel == 1)
    //         {
    //             pval = image_ptr[0][0][h][w];
    //         }
    //         else
    //         {
    //             for (int c_i = 0; c_i < channel; c_i++)
    //             {
    //                 pval_v[c_i] = image_ptr[0][c_i][h][w];
    //             }
    //         }
    //         float new_dist = distance_ptr[0][0][h][w];

    //         for (int w_i = 0; w_i < 3; w_i++)
    //         {
    //             const int w_ind = w + w_i - 1;
    //             if (w_ind < 0 || w_ind >= width)
    //                 continue;

    //             float l_dist;
    //             if (channel == 1)
    //             {
    //                 l_dist = l1distance_cuda(pval, image_ptr[0][0][h - 1][w_ind]);
    //             }
    //             else
    //             {
    //                 float qval_v[channel];
    //                 for (int c_i = 0; c_i < channel; c_i++)
    //                 {
    //                     qval_v[c_i] = image_ptr[0][c_i][h - 1][w_ind];
    //                 }
    //                 l_dist = l1distance_cuda(pval_v, qval_v, channel);
    //             }
    //             const float cur_dist = distance_ptr[0][0][h - 1][w_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
    //             new_dist = std::min(new_dist, cur_dist);
    //         }
    //         distance_ptr[0][0][h][w] = new_dist;
    //     }
    // }

    // // bottom-up
    // for (int h = height - 2; h >= 0; h--)
    // {
    //     // use openmp to parallelise the loop over width
    //     #ifdef _OPENMP
    //         #pragma omp parallel for
    //     #endif
    //     for (int w = 0; w < width; w++)
    //     {
    //         float pval;
    //         float pval_v[channel];
    //         if (channel == 1)
    //         {
    //             pval = image_ptr[0][0][h][w];
    //         }
    //         else
    //         {
    //             for (int c_i = 0; c_i < channel; c_i++)
    //             {
    //                 pval_v[c_i] = image_ptr[0][c_i][h][w];
    //             }
    //         }
    //         float new_dist = distance_ptr[0][0][h][w];

    //         for (int w_i = 0; w_i < 3; w_i++)
    //         {
    //             const int w_ind = w + w_i - 1;
    //             if (w_ind < 0 || w_ind >= width)
    //                 continue;

    //             float l_dist;
    //             if (channel == 1)
    //             {
    //                 l_dist = l1distance_cuda(pval, image_ptr[0][0][h + 1][w_ind]);
    //             }
    //             else
    //             {
    //                 float qval_v[channel];
    //                 for (int c_i = 0; c_i < channel; c_i++)
    //                 {
    //                     qval_v[c_i] = image_ptr[0][c_i][h + 1][w_ind];
    //                 }
    //                 l_dist = l1distance_cuda(pval_v, qval_v, channel);
    //             }
    //             const float cur_dist = distance_ptr[0][0][h + 1][w_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
    //             new_dist = std::min(new_dist, cur_dist);
    //         }
    //         distance_ptr[0][0][h][w] = new_dist;
    //     }
    // }
}

torch::Tensor generalised_geodesic2d_cuda(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image = image.contiguous();
        distance = distance.contiguous();

        // top-bottom - width*, height
        geodesic_updown_pass_cuda(image, distance, l_grad, l_eucl);

        // left-right - height*, width
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_updown_pass_cuda(image, distance, l_grad, l_eucl);
        
        // tranpose back to original - width, height
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }

    return distance;
}

void geodesic_frontback_pass_cuda(const torch::Tensor &image, torch::Tensor &distance, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
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
                            l_dist = l1distance_cuda(pval_v, qval_v, channel);
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
                            l_dist = l1distance_cuda(pval_v, qval_v, channel);
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

torch::Tensor generalised_geodesic3d_cuda(torch::Tensor &image, const torch::Tensor &mask, const std::vector<float> &spacing, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    torch::Tensor distance = v * mask.clone();

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image = image.contiguous();
        distance = distance.contiguous();

        // front-back - depth*, height, width
        geodesic_frontback_pass_cuda(image, distance, spacing, l_grad, l_eucl);

        // top-bottom - height*, depth, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        
        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass_cuda(image, distance, {spacing[1], spacing[0], spacing[2]}, l_grad, l_eucl);
        
        // transpose back to original depth, height, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        
        // left-right - width*, height, depth
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);
        
        image = image.contiguous();
        distance = distance.contiguous();
        geodesic_frontback_pass_cuda(image, distance, {spacing[2], spacing[1], spacing[0]}, l_grad, l_eucl);
        
        // transpose back to original depth, height, width
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);

        // * indicates the current direction of pass
    }

    return distance;
}