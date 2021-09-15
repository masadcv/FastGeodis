#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THREAD_COUNT 256

// whether to use float* or Pytorch accessors in CUDA kernels
#define USE_PTR 1

__constant__ float local_dist2d[3];
__constant__ float local_dist3d[3*3];

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

    // const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

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
                cur_dist =  distance_ptr[0][0][h + direction][kernelW_ind] + l_eucl * local_dist2d[w_i] + l_grad * l_dist;
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
    // const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    
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
                cur_dist = distance_ptr[(h + direction) * width + kernelW_ind] + l_eucl * local_dist2d[w_i] + l_grad * l_dist;
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
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);
    
    if (channel != 1)
    {
        throw std::runtime_error(
            "CUDA implementation currently only supports 1 channel, received " + std::to_string(channel) + \
            " channels\nTry passing tensors with tensor.cpu() to run cpu implementation"
            );
    }

    // constexpr float local_dist[] = {sqrt(2.), 1., sqrt(2.)};
    const float local_dist[] = {sqrt(float(2.)), float(1.), sqrt(float(2.))};

    // copy local distances to GPU __constant__ memory
    cudaMemcpyToSymbol(local_dist2d, local_dist, sizeof(float) * 3);

	int blockCountUpDown = (width + 1)/THREAD_COUNT + 1;

    // direction variable used to indicate read from previous (-1) or next (+1) row
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

    // bottom-up
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

template <typename scalar_t>
__global__ void geodesic_frontback_single_plane_pass_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> image_ptr, 
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction,
    const int z
    )
{
    const int height = image_ptr.size(3);
    const int width = image_ptr.size(4);

    // float spacing[] = {1.0, 1.0, 1.0};

    // // make local distances
    // float local_dist[3*3];
    // for (int h_i = 0; h_i < 3; h_i++)
    // {
    //     for (int w_i = 0; w_i < 3; w_i++)
    //     {
    //         float ld = spacing[0];
    //         ld += float(std::abs(h_i-1)) * spacing[1];
    //         ld += float(std::abs(w_i-1)) * spacing[2];

    //         local_dist[h_i * 3 + w_i] = ld;
    //     }
    // }

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelH = blockIdx.y * blockDim.y + threadIdx.y;
    
    // if outside, then skip distance calculation - dont use the thread
    if(kernelH >= 0 && kernelH < height && kernelW >= 0 && kernelW < width)
    {
        float pval = image_ptr[0][0][z][kernelH][kernelW];
        float new_dist = distance_ptr[0][0][z][kernelH][kernelW];

        float cur_dist = 0.0;
        for (int h_i = 0; h_i < 3; h_i++)
        {
            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int kernelH_ind = kernelH + h_i - 1;
                const int kernelW_ind = kernelW + w_i - 1;

                if (kernelH_ind >= 0 && kernelH_ind < height && kernelW_ind >= 0 && kernelW_ind < width)
                {
                    float l_dist = l1distance_cuda(pval, image_ptr[0][0][z + direction][kernelH_ind][kernelW_ind]);
                    cur_dist =  distance_ptr[0][0][z + direction][kernelH_ind][kernelW_ind] + l_eucl * local_dist3d[h_i * 3 + w_i] + l_grad * l_dist;
                    new_dist = std::min(new_dist, cur_dist);
                }
            }
        }
        if(new_dist < distance_ptr[0][0][z][kernelH][kernelW])
        {
            distance_ptr[0][0][z][kernelH][kernelW] = new_dist;
        } 
    }
}

__global__ void geodesic_frontback_single_plane_pass_kernel(
    float *image_ptr, 
    float *distance_ptr,
    const float l_grad,
    const float l_eucl,
    const int direction,
    const int z,
    const int height,
    const int width
    )
{

    // float spacing[] = {1.0, 1.0, 1.0};

    // // make local distances
    // float local_dist[3*3];
    // for (int h_i = 0; h_i < 3; h_i++)
    // {
    //     for (int w_i = 0; w_i < 3; w_i++)
    //     {
    //         float ld = spacing[0];
    //         ld += float(std::abs(h_i-1)) * spacing[1];
    //         ld += float(std::abs(w_i-1)) * spacing[2];

    //         local_dist[h_i * 3 + w_i] = ld;
    //     }
    // }

    int kernelW = blockIdx.x * blockDim.x + threadIdx.x;
    int kernelH = blockIdx.y * blockDim.y + threadIdx.y;
    
    // if outside, then skip distance calculation - dont use the thread
    if(kernelH >= 0 && kernelH < height && kernelW >= 0 && kernelW < width)
    {
        // float pval = image_ptr[0][0][z][kernelH][kernelW];
        float pval = image_ptr[z*height*width + kernelH*width + kernelW];
        // float new_dist = distance_ptr[0][0][z][kernelH][kernelW];
        float new_dist = distance_ptr[z*height*width + kernelH*width + kernelW];

        float cur_dist = 0.0;
        for (int h_i = 0; h_i < 3; h_i++)
        {
            for (int w_i = 0; w_i < 3; w_i++)
            {
                const int kernelH_ind = kernelH + h_i - 1;
                const int kernelW_ind = kernelW + w_i - 1;

                if (kernelH_ind >= 0 && kernelH_ind < height && kernelW_ind >= 0 && kernelW_ind < width)
                {
                    float l_dist = l1distance_cuda(pval, image_ptr[(z + direction)*height*width + kernelH_ind*width + kernelW_ind]);
                    cur_dist =  distance_ptr[(z + direction)*height*width + kernelH_ind*width + kernelW_ind] + l_eucl * local_dist3d[h_i * 3 + w_i] + l_grad * l_dist;
                    new_dist = std::min(new_dist, cur_dist);
                }
            }
        }
        if(new_dist < distance_ptr[z*height*width + kernelH*width + kernelW])
        {
            distance_ptr[z*height*width + kernelH*width + kernelW] = new_dist;
        } 
    }
}


void geodesic_frontback_pass_cuda(const torch::Tensor &image, torch::Tensor &distance, const std::vector<float> &spacing, const float &l_grad, const float &l_eucl)
{
    // batch, channel, depth, height, width
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    if (channel != 1)
    {
        throw std::runtime_error(
            "CUDA implementation currently only supports 1 channel, received " + std::to_string(channel) + \
            " channels\nTry passing tensors with tensor.cpu() to run cpu implementation"
            );
    }

    // convert allowed number of threads into a 2D grid
    // helps if the THREAD_COUNT is N*N already 
    const int THREAD_COUNT_2D = sqrt(THREAD_COUNT);
	int blockCountUpDown = (width + 1)/THREAD_COUNT_2D + 1;
	int blockCountLeftRight = (height + 1)/THREAD_COUNT_2D + 1;

    // pre-calculate local distances based on spacing
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
    // copy local distances to GPU __constant__ memory
    cudaMemcpyToSymbol(local_dist3d, local_dist, sizeof(float) * 3*3);


    dim3 dimGrid(blockCountUpDown, blockCountLeftRight);
    dim3 dimBlock(THREAD_COUNT_2D, THREAD_COUNT_2D);
    // Kernel<<<dimGrid, dimBlock>>>( arg1, arg2, arg2);

    // direction variable used to indicate read from previous (-1) or next (+1) plane
    int direction;

    // front-back
    direction = -1;
    for (int z = 1; z < depth; z++)
    {
        if(~USE_PTR)
        {
            AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_frontback_single_plane_pass_kernel", ([&] {
                geodesic_frontback_single_plane_pass_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                    image.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                    distance.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                    l_grad, 
                    l_eucl, 
                    direction,
                    z
                    );
                }));
        }
        else
        {
            geodesic_frontback_single_plane_pass_kernel<<<dimGrid, dimBlock>>>(
                image.data_ptr<float>(), 
                distance.data_ptr<float>(),
                l_grad,
                l_eucl,
                direction,
                z,
                height,
                width
                );
        }
    }

    direction = +1;
    for (int z = depth - 2; z >= 0; z--)
    {
        if(~USE_PTR)
        {
            AT_DISPATCH_FLOATING_TYPES(image.type(), "geodesic_frontback_single_plane_pass_kernel", ([&] {
                geodesic_frontback_single_plane_pass_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                    image.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                    distance.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(), 
                    l_grad, 
                    l_eucl, 
                    direction,
                    z
                    );
                }));
        }
        else
        {
            geodesic_frontback_single_plane_pass_kernel<<<dimGrid, dimBlock>>>(
                image.data_ptr<float>(), 
                distance.data_ptr<float>(),
                l_grad,
                l_eucl,
                direction,
                z,
                height,
                width
                );
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