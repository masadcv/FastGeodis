#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THREAD_COUNT 512
#define STRIP_HEIGHT 32

// whether to use float* or Pytorch accessors in CUDA kernels
#define USE_PTR 0

__device__ float l1distance_cuda(const float &in1, const float &in2)
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


// template <typename scalar_t>
// __global__ void geodesic_updown_pass_kernel(
//     torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> image_ptr, 
//     torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> distance_ptr,
//     const float &l_grad, 
//     const float &l_eucl,
//     const float *local_dist,
//     const int row,
//     const int height,
//     const int width)
// {
//     __shared__ float lastRow[THREAD_COUNT+2];
//     __shared__ float curRow[THREAD_COUNT+2];

//     // overlap
//     int blockStartX = (blockIdx.x * THREAD_COUNT) - (blockIdx.x * STRIP_HEIGHT * 2);
//     int blockSafeX = blockStartX + THREAD_COUNT - (2 * STRIP_HEIGHT);

//     int kernelX = blockStartX + threadIdx.x;

//     // copy last and current row into shared memory using each thread
//     lastRow[threadIdx.x + 1] = -1;// distance_ptr[0][0][row-1][kernelX];
//     curRow[threadIdx.x + 1] = -1;// distance_ptr[0][0][row][kernelX];
//     if(kernelX < width)
//     {
//         lastRow[threadIdx.x + 1] = distance_ptr[0][0][row-1][kernelX];
//         curRow[threadIdx.x + 1] = distance_ptr[0][0][row][kernelX];
//     }
    
//     if (threadIdx.x == 0)
//     {
//         lastRow[0] = -1;
//         curRow[0] = -1;
//         if ((kernelX - 1) >= 0 && (kernelX-1) < width)
//         { 
//             lastRow[0] = distance_ptr[0][0][row-1][kernelX-1];
//             curRow[0] = distance_ptr[0][0][row][kernelX-1];
//         }
//     }
//     else if(threadIdx.x == THREAD_COUNT + 1)
//     {
//         lastRow[threadIdx.x + 1] = -1;
//         curRow[threadIdx.x + 1] = -1;
//         if ((kernelX+1) >= 0 && (kernelX + 1) < width)
//         {
//             lastRow[threadIdx.x + 1] = distance_ptr[0][0][row-1][kernelX+1];
//             curRow[threadIdx.x + 1] = distance_ptr[0][0][row][kernelX+1];
//         }
//     }

//     __syncthreads();

//     if (1==1)
//     {
        
//         // top-down pass for each row in strip
//         for (int i = 0; i < STRIP_HEIGHT; i++)
//         {
//             float solution = -1;
//             int currentHeightIdx = row + i;

//             if (currentHeightIdx < height and kernelX < width)
//             {
//                 int localKernelX = (int)threadIdx.x + 1;
//                 // solution = curRow[localKernelX];

//                 float pval = image_ptr[0][0][currentHeightIdx][kernelX];
                
//                 int w_i, w_ind;
//                 float cur_dist;

//                 // left back
//                 w_i = 0;
//                 w_ind = kernelX + w_i - 1;
//                 cur_dist = lastRow[localKernelX + w_i - 1];
//                 float left_solution=-1;
//                 if (cur_dist >= 0 && (w_ind >= 0 && w_ind < width))
//                 {
//                     float qval = image_ptr[0][0][currentHeightIdx-1][w_ind];
//                     float l_dist = abs(pval - qval);
//                     left_solution = cur_dist + l_eucl * local_dist[w_i] + l_grad * l_dist;
//                 }

//                 // center back
//                 w_i = 1;
//                 w_ind = kernelX + w_i - 1;
//                 cur_dist = lastRow[localKernelX + w_i - 1];
//                 float center_solution=-1;
//                 if (cur_dist >= 0 && (w_ind >= 0 && w_ind < width))
//                 {
//                     float qval = image_ptr[0][0][currentHeightIdx-1][w_ind];
//                     float l_dist = abs(pval - qval);
//                     center_solution = cur_dist + l_eucl * local_dist[w_i] + l_grad * l_dist;
//                 }

//                 // right back
//                 w_i = 2;
//                 w_ind = kernelX + w_i - 1;
//                 cur_dist = lastRow[localKernelX + w_i - 1];
//                 float right_solution=-1;
//                 if (cur_dist >= 0 && (w_ind >= 0 && w_ind < width))
//                 {
//                     float qval = image_ptr[0][0][currentHeightIdx-1][w_ind];
//                     float l_dist = abs(pval - qval);
//                     right_solution = cur_dist + l_eucl * local_dist[w_i]; + l_grad * l_dist;
//                 }

//                 // for(int w_i = 0; w_i < 3; w_i++)
//                 // {
//                 //     const int w_ind = kernelX + w_i - 1;
//                 //     const float cur_dist = lastRow[localKernelX + w_i - 1];
//                 //     if (cur_dist >= 0 && (w_ind >= 0 && w_ind < width))
//                 //     {
//                 //         float qval = image_ptr[0][0][currentHeightIdx-1][w_ind];
//                 //         float l_dist = l1distance_cuda(pval, qval);
//                 //         float cur_solution = (cur_dist + l_eucl * local_dist[w_i] + l_grad * l_dist);
//                 //         if(w_i == 0)
//                 //         {
//                 //             solution = cur_solution;
//                 //         }
//                 //         else
//                 //         {
//                 //             if(cur_solution < solution)
//                 //             {
//                 //                 solution = cur_solution;
//                 //             }
//                 //         }
//                 //    }
//                 // }
//                 solution = left_solution;
//                 if(center_solution < left_solution)
//                 {
//                     solution = center_solution;
//                 }
//                 else if(right_solution < left_solution && right_solution < center_solution)
//                 {
//                     solution = right_solution;
//                 }
//             }
            
//             __syncthreads();

//             // if (solution >= 0.0 && kernelX < blockSafeX && (curRow[threadIdx.x] < 0.0 || solution < curRow[threadIdx.x]))
//             if (solution >= 0.0 && kernelX < blockSafeX)// && solution < curRow[threadIdx.x + 1])
//             {
//                 if(currentHeightIdx >= 0 && currentHeightIdx < height && kernelX >= 0 && kernelX < width)
//                 {
//                     distance_ptr[0][0][currentHeightIdx][kernelX] = solution;
//                     // curRow[threadIdx.x + 1] = solution;
//                 }
//             }

//             lastRow[threadIdx.x] = curRow[threadIdx.x];
//             curRow[threadIdx.x] = -1;
//             if((currentHeightIdx+1) < height)
//             {
//                 curRow[threadIdx.x] = distance_ptr[0][0][currentHeightIdx + 1][kernelX];     
//                 if (threadIdx.x == 0)
//                 {
//                     curRow[0] = -1;
//                     if((kernelX - 1) >= 0)
//                     {
//                         curRow[0] = distance_ptr[0][0][currentHeightIdx + 1][kernelX - 1];
//                     }
//                 }
//                 else if (threadIdx.x == THREAD_COUNT + 1)
//                 {
//                     curRow[THREAD_COUNT + 1] = -1;
//                     if((kernelX + 1) < width)
//                     {
//                         curRow[THREAD_COUNT + 1] = distance_ptr[0][0][currentHeightIdx + 1][kernelX + 1];
//                     }
//                 }
//             }

//             __syncthreads();
                
//                 // TODO: bottom up pass
//         }
//     }
//     else{
//         // top-down pass for each row in strip
//         for (int i = 0; i < STRIP_HEIGHT; i++)
//         {
//             __syncthreads();
//             int currentHeightIdx = row + i;
//             if(currentHeightIdx >= 0 && currentHeightIdx < height && kernelX >= 0 && kernelX < width && kernelX < blockSafeX)
//             {
//                 distance_ptr[0][0][currentHeightIdx][kernelX] = kernelX;
//                 // curRow[threadIdx.x + 1] = solution;

//             }

//             lastRow[threadIdx.x] = curRow[threadIdx.x];
//             curRow[threadIdx.x] = -1;
//             if((currentHeightIdx+1) < height)
//             {
//                 curRow[threadIdx.x] = distance_ptr[0][0][currentHeightIdx + 1][kernelX];     
//                 if (threadIdx.x == 0)
//                 {
//                     curRow[0] = -1;
//                     if((kernelX - 1) >= 0)
//                     {
//                         curRow[0] = distance_ptr[0][0][currentHeightIdx + 1][kernelX - 1];
//                     }
//                 }
//                 else if (threadIdx.x == THREAD_COUNT + 1)
//                 {
//                     curRow[THREAD_COUNT + 1] = -1;
//                     if((kernelX + 1) < width)
//                     {
//                         curRow[THREAD_COUNT + 1] = distance_ptr[0][0][currentHeightIdx + 1][kernelX + 1];
//                     }
//                 }
//             }

//             __syncthreads();
//         }

//     }    

// }


template <typename scalar_t>
__global__ void geodesic_updown_single_row_pass_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> image_ptr, 
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction,
    const int h
    )
{
    const float height = image_ptr.size(2);
    const float width = image_ptr.size(3);

    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    
    // if outside, then skip distance calculation - dont use the thread
    if(kernelW < width)
    {
        float pval = image_ptr[0][0][h][kernelW];
        float new_dist = distance_ptr[0][0][h][kernelW];
        float cur_dist = 0.0;
        for (int w_i = 0; w_i < 3; w_i++)
        {
            const int kernelW_ind = kernelW + w_i - 1;
            if (kernelW_ind >= 0 && kernelW_ind < width)
            {
                float l_dist = l1distance_cuda(pval, image_ptr[0][0][h + direction][kernelW_ind]);
                cur_dist =  distance_ptr[0][0][h + direction][kernelW_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
                new_dist = std::min(new_dist, cur_dist);
            }
        }
        if(new_dist < distance_ptr[0][0][h][kernelW])
        {
            distance_ptr[0][0][h][kernelW] = new_dist;
        } 
    }
}

__global__ void geodesic_updown_single_row_pass_ptr_kernel(
    float* image_ptr, 
    float* distance_ptr,
    float l_grad,
    float l_eucl,
    const int direction,
    const int h,
    const int height,
    const int width
    )
{
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    int kernelW = blockIdx.x * THREAD_COUNT + threadIdx.x;
    
    // if outside, then skip distance calculation - dont use the thread
    if(kernelW < width)
    {
        float pval = image_ptr[h * width + kernelW];
        float new_dist = distance_ptr[h * width + kernelW];
        float cur_dist = 0.0;

        for (int w_i = 0; w_i < 3; w_i++)
        {
            const int kernelW_ind = kernelW + w_i - 1;
            if (kernelW_ind >= 0 && kernelW_ind < width)
            {
                float l_dist = l1distance_cuda(pval, image_ptr[(h + direction) * width + kernelW_ind]);
                cur_dist = distance_ptr[(h + direction) * width + kernelW_ind] + l_eucl * local_dist[w_i] + l_grad * l_dist;
                new_dist = std::min(new_dist, cur_dist);
            }
        }
        if(new_dist < distance_ptr[h * width + kernelW])
        {
            distance_ptr[h * width + kernelW] = new_dist;
        } 
    }
}

void geodesic_updown_pass_cuda(const torch::Tensor image, torch::Tensor distance, const float &l_grad,  const float &l_eucl)
{
    // batch, channel, height, width
    const int height = image.size(2);
    const int width = image.size(3);

    // constexpr float local_dist[] = {sqrt(2.), 1., sqrt(2.)};
    // float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

	int blockCountUpDown = (width + 1)/THREAD_COUNT + 1;

    int direction;

    // top-down
    direction = -1; 
    for (int h = 1; h < height; h++)
    {
        // each distance calculation in down pass require previous row i.e. -1
        // process each row in parallel with CUDA kernel
        if(~USE_PTR)
        {
            AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_updown_single_row_pass_kernel", ([&] {
                geodesic_updown_single_row_pass_kernel<scalar_t><<<blockCountUpDown, THREAD_COUNT>>>(
                    image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                    distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                    l_grad, 
                    l_eucl, 
                    direction,
                    h
                    );
                }));
        }
        else
        {    
            geodesic_updown_single_row_pass_ptr_kernel<<<blockCountUpDown, THREAD_COUNT>>>(
                image.data_ptr<float>(), 
                distance.data_ptr<float>(), 
                l_grad, 
                l_eucl, 
                direction,
                h, 
                height, 
                width
                );
        }    
    }

    direction = +1;
    for (int h = height - 2; h >= 0; h--)
    {
        if(~USE_PTR)
        {
            AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_updown_single_row_pass_kernel", ([&] {
                geodesic_updown_single_row_pass_kernel<scalar_t><<<blockCountUpDown, THREAD_COUNT>>>(
                    image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                    distance.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), 
                    l_grad, 
                    l_eucl, 
                    direction,
                    h
                    );
            }));
        }
        else
        {
            geodesic_updown_single_row_pass_ptr_kernel<<<blockCountUpDown, THREAD_COUNT>>>(
                image.data_ptr<float>(), 
                distance.data_ptr<float>(), 
                l_grad, 
                l_eucl, 
                direction,
                h, 
                height, 
                width
                );
        }
    }
}

torch::Tensor generalised_geodesic2d_cuda(torch::Tensor &image, const torch::Tensor &mask, const float &v, const float &l_grad, const float &l_eucl, const int &iterations)
{
    torch::Tensor distance = v * mask.clone();
    image = image.contiguous();
    distance = distance.contiguous();

    // std::cout << "Reached here2" << std::endl;

    // top-bottom - width*, height
    // geodesic_updown_pass_cuda(image, distance, l_grad, l_eucl);

    // iteratively run the distance transform
    for (int itr = 0; itr < iterations; itr++)
    {
        image = image.contiguous();
        distance = distance.contiguous();

        std::cout << "Reached here2" << std::endl;

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