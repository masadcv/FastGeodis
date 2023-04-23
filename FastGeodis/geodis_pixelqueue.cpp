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

float l1distance_pixelqueue(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}

struct Point2D
{
    float distance;
    int w;
    int h;
};

void insert_point_to_list(std::vector<Point2D> * list, int start_position,  Point2D p)
{
    int insert_idx = list->size();
    for(int i = start_position; i < int(list->size()); i++)
    {
        if(list->at(i).distance < p.distance)
        {
            insert_idx = i;
            break;
        }
    }
    list->insert(insert_idx + list->begin(), p);
}

void update_point_in_list(std::vector<Point2D> * list, Point2D p)
{
    int remove_idx = -1;
    for(int i = 0; i < int(list->size()); i++)
    {
        if(list->at(i).w == p.w && list->at(i).h == p.h)
        {
            remove_idx = i;
            break;
        }
    }
    list->erase(remove_idx + list->begin());
    insert_point_to_list(list, remove_idx, p);
}

void add_new_accepted_point(
    torch::TensorAccessor<float, 4> image_ptr, 
    torch::TensorAccessor<float, 4> distance_ptr, 
    torch::TensorAccessor<signed char, 2> state_ptr, 
    std::vector<Point2D> * list, 
    const Point2D &p,
    const int *dh_f,
    const int *dw_f,
    const float *local_dist_f,
    const int &channel, 
    const int &height, 
    const int &width, 
    const float &l_grad,
    const float &l_eucl)
{
    int w = p.w;
    int h = p.h;
    int dh, dw, nh, nw, temp_state;
    float l_dist, space_dis, delta_dis, old_dis, new_dis;
    
    for(int ind = 0; ind < 9; ind++)
    {
        dh = dh_f[ind];
        dw = dw_f[ind];

        space_dis = local_dist_f[ind];

        if(dh == 0 && dw == 0) 
        {
            continue;
        }

        nh = dh + h;
        nw = dw + w;
        
        if(nh >=0 && nh < height && nw >=0 && nw < width)
        {
            temp_state = state_ptr[nh][nw];

            if(temp_state == 0)
            {
                continue;
            }
            
            l_dist = 0.0;
            if (channel == 1)
            {
                l_dist = l1distance_pixelqueue(
                    image_ptr[0][0][h][w], 
                    image_ptr[0][0][nh][nw]); 
            }
            else
            {
                for (int c_i=0; c_i < channel; c_i++)
                {
                    l_dist += l1distance_pixelqueue(
                        image_ptr[0][c_i][h][w], 
                        image_ptr[0][c_i][nh][nw]);     
                }       
            }
            delta_dis = l_eucl * space_dis + l_grad * l_dist;
            old_dis   = distance_ptr[0][0][nh][nw];
            new_dis   = distance_ptr[0][0][h][w] + delta_dis;

            if(new_dis < old_dis)
            {
                distance_ptr[0][0][nh][nw] = new_dis;

                Point2D new_point;
                new_point.distance = new_dis;
                new_point.h = nh;
                new_point.w = nw;
                
                if(temp_state == 2)
                {
                    state_ptr[nh][nw] = 1;
                    insert_point_to_list(list, 0, new_point);
                }
                else{
                    update_point_in_list(list, new_point);
                }
            }
        }
    }
}

void geodesic2d_pixelqueue_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, height, width
    const int channel = image.size(1);
    const int height = image.size(2);
    const int width = image.size(3);

    // initialise state
    auto options = torch::TensorOptions()
                    .dtype(torch::kInt8)
                    .device(torch::kCPU, 1)
                    .requires_grad(false);
    auto state = torch::zeros({height, width}, options);

    // state value: 0 == accepted, 1 == temp, 2 == far away
    auto image_ptr = image.accessor<float, 4>();
    auto distance_ptr = distance.accessor<float, 4>();
    auto state_ptr = state.accessor<signed char, 2>();

    const int dh_f[9] = {
        -1, -1, -1,  
         0,  0,  0,   
         1,  1,  1
        };
    const int dw_f[9] = {
        -1,  0,  1, 
        -1,  0,  1,  
        -1,  0,  1
        };

    const float local_dist_f[9] = {
        sqrt(float(2.)), float(1), sqrt(float(2.)), 
        float(1.), float(0.), float(1.), 
        sqrt(float(2.)), float(1.), sqrt(float(2.))};

    int init_state;
    float seed_type, init_dis;
    
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            seed_type = distance_ptr[0][0][h][w];
            if (seed_type > 0)
            {
                init_state = 2;
                init_dis = 1.0e10;
            }
            else
            {
                init_state = 0;
                init_dis = 0.0;
            }
            state[h][w] = init_state;
            distance_ptr[0][0][h][w] = init_dis;
        }
    }

    // get initial temporary set
    std::vector<Point2D> temporary_list;
    temporary_list.reserve(width * height);
    int temp_state;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            temp_state = state_ptr[h][w];
            if (temp_state == 0)
            {
                Point2D accepted_p;
                accepted_p.distance = 0.0;
                accepted_p.h = h;
                accepted_p.w = w;
                add_new_accepted_point(
                    image_ptr, 
                    distance_ptr, 
                    state_ptr, 
                    &temporary_list, 
                    accepted_p,
                    dh_f,
                    dw_f,
                    local_dist_f,
                    channel,
                    height,
                    width,
                    l_grad,
                    l_eucl);
            }
        }
    }

    // update temporary set until it is empty
    while (temporary_list.size() > 0)
    {
        Point2D temp_point = temporary_list[temporary_list.size() -1];
        temporary_list.pop_back();
        state[temp_point.h][temp_point.w] = 0;
        add_new_accepted_point(
            image_ptr, 
            distance_ptr, 
            state_ptr, 
            &temporary_list, 
            temp_point,
            dh_f,
            dw_f,
            local_dist_f,
            channel,
            height,
            width,
            l_grad,
            l_eucl);
    }
}

torch::Tensor geodesic2d_pixelqueue_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const float &l_grad,
    const float &l_eucl)
{
    torch::Tensor distance = mask.clone();

    geodesic2d_pixelqueue_cpu(image, distance, l_grad, l_eucl);

    return distance;
}

struct Point3D
{
    float distance;
    int d;
    int w;
    int h;
};

void insert_point_to_list(std::vector<Point3D> * list, int start_position,  Point3D p)
{
    int insert_idx = list->size();
    for(int i = start_position; i < int(list->size()); i++)
    {
        if(list->at(i).distance < p.distance)
        {
            insert_idx = i;
            break;
        }
    }
    list->insert(insert_idx + list->begin(), p);
}

void update_point_in_list(std::vector<Point3D> * list, Point3D p)
{
    int remove_idx = -1;
    for(int i = 0; i < int(list->size()); i++)
    {
        if(list->at(i).d == p.d && list->at(i).h == p.h && list->at(i).w == p.w)
        {
            remove_idx = i;
            break;
        }
    }
    list->erase(remove_idx + list->begin());
    insert_point_to_list(list, remove_idx, p);
}

void add_new_accepted_point(
    const torch::TensorAccessor<float, 5> &image_ptr, 
    torch::TensorAccessor<float, 5> distance_ptr, 
    torch::TensorAccessor<signed char, 3> state_ptr, 
    std::vector<Point3D> * list,
    const Point3D &p,
    const std::vector<float> &spacing, 
    const int *dd_f,
    const int *dh_f,
    const int *dw_f,
    const float *local_dist_f,
    const int &channel, 
    const int &depth, 
    const int &height, 
    const int &width, 
    const float &l_grad,
    const float &l_eucl)
{
    int d = p.d;
    int h = p.h;
    int w = p.w;
    int dd, dh, dw, nd, nh, nw, temp_state;
    float l_dist, space_dis, delta_dis, old_dis, new_dis;
    
    for (int ind = 0; ind < 27; ind++)   
    {
        dd = dd_f[ind];
        dh = dh_f[ind];
        dw = dw_f[ind];

        space_dis = local_dist_f[ind];

        if(dd == 0 && dh == 0 && dw == 0)
        {
            continue;
        }

        nd = dd + d;
        nh = dh + h;
        nw = dw + w;
        
        if(nd >=0 && nd < depth && nh >=0 && nh < height && nw >=0 && nw < width )
        {
            temp_state = state_ptr[nd][nh][nw];

            if(temp_state == 0)
            {
                continue;
            }

            l_dist = 0.0;
            if (channel == 1)
            {
                l_dist = l1distance_pixelqueue(
                    image_ptr[0][0][d][h][w], 
                    image_ptr[0][0][nd][nh][nw]); 
            }
            else
            {
                for (int c_i=0; c_i < channel; c_i++)
                {
                    l_dist += l1distance_pixelqueue(
                        image_ptr[0][c_i][d][h][w], 
                        image_ptr[0][c_i][nd][nh][nw]);     
                }       
            }                    
            delta_dis = l_eucl * space_dis + l_grad * l_dist;
            old_dis   = distance_ptr[0][0][nd][nh][nw];
            new_dis   = distance_ptr[0][0][d][h][w] + delta_dis;

            if(new_dis < old_dis)
            {
                distance_ptr[0][0][nd][nh][nw] = new_dis;

                Point3D new_point;
                new_point.distance = new_dis;
                new_point.d = nd;
                new_point.h = nh;
                new_point.w = nw;

                if(temp_state == 2)
                {
                    state_ptr[nd][nh][nw] = 1;
                    insert_point_to_list(list, 0, new_point);
                }
                else
                {
                    update_point_in_list(list, new_point);
                }
            }
        }
    }
}


void geodesic3d_pixelqueue_cpu(
    const torch::Tensor &image,
    torch::Tensor &distance,
    std::vector<float> spacing,
    const float &l_grad,
    const float &l_eucl)
{
    // batch, channel, depth, height, width
    const int channel = image.size(1);
    const int depth = image.size(2);
    const int height = image.size(3);
    const int width = image.size(4);

    // initialise state
    auto options = torch::TensorOptions()
                    .dtype(torch::kInt8)
                    .device(torch::kCPU, 1)
                    .requires_grad(false);
    auto state = torch::zeros({depth, height, width}, options);

    // point state: 0--acceptd, 1--temporary, 2--far away
    auto image_ptr = image.accessor<float, 5>();
    auto distance_ptr = distance.accessor<float, 5>();
    auto state_ptr = state.accessor<signed char, 3>();

    const int dd_f[27] = { 
        -1, -1, -1, -1, -1, -1, -1, -1, -1, 
        0,  0,  0,  0,  0,  0,  0,  0,  0, 
        1,  1,  1,  1,  1,  1,  1,  1,  1
        };
    const int dh_f[27] = {
        -1, -1, -1,  0,  0,  0,  1,  1,  1, 
        -1, -1, -1,  0,  0,  0,  1,  1,  1,
        -1, -1, -1,  0,  0,  0,  1,  1,  1
        };
    const int dw_f[27] = { 
        -1,  0,  1, -1,  0,  1, -1,  0,  1, 
        -1,  0,  1, -1,  0,  1, -1,  0,  1, 
        -1,  0,  1, -1,  0,  1, -1,  0,  1
        };
    
    float local_dist_f[27];
    for (int ind = 0; ind < 27; ind++)
    {
        float ld = 0.0;
        if (dd_f[ind] != 0)
        {
            ld += spacing[0] * spacing[0];
        }

        if (dh_f[ind] != 0)
        {
            ld += spacing[1] * spacing[1];
        }

        if (dw_f[ind] != 0)
        {
            ld += spacing[2] * spacing[2];
        }

        local_dist_f[ind] = sqrt(ld);
    }
    
    int init_state;
    float seed_type, init_dis;

    for(int d = 0; d < depth; d++)
    {
        for(int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                seed_type = distance_ptr[0][0][d][h][w];
                if(seed_type > 0)
                {
                    init_state = 2;
                    init_dis = 1.0e10;
                }
                else
                {
                    init_state = 0;
                    init_dis = 0;
                }
                state_ptr[d][h][w] = init_state;
                distance_ptr[0][0][d][h][w] = init_dis;
            }
        }
    }
    
    // get initial temporary set
    std::vector<Point3D> temporary_list;
    temporary_list.reserve(depth * height * width);
    int temp_state;
    for(int d = 0; d < depth; d++)
    {
        for(int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                temp_state = state_ptr[d][h][w];
                if(temp_state == 0)
                {
                    Point3D accepted_p;
                    accepted_p.distance = 0.0;
                    accepted_p.d = d;
                    accepted_p.h = h;
                    accepted_p.w = w;
                    add_new_accepted_point(
                        image_ptr, 
                        distance_ptr, 
                        state_ptr, 
                        &temporary_list, 
                        accepted_p, 
                        spacing,
                        dd_f,
                        dh_f,
                        dw_f,
                        local_dist_f, 
                        channel, 
                        depth, 
                        height, 
                        width, 
                        l_grad,
                        l_eucl);
                 }
            }
        }
    }

    // update temporary set until it is empty
    while(temporary_list.size() > 0)
    {
        Point3D temp_point = temporary_list[temporary_list.size() - 1];
        temporary_list.pop_back();
        state_ptr[temp_point.d][temp_point.h][temp_point.w] = 0;
        add_new_accepted_point(
            image_ptr, 
            distance_ptr, 
            state_ptr, 
            &temporary_list, 
            temp_point,
            spacing, 
            dd_f,
            dh_f,
            dw_f,
            local_dist_f,
            channel, 
            depth, 
            height, 
            width, 
            l_grad,
            l_eucl);
    }
}

torch::Tensor geodesic3d_pixelqueue_cpu(
    const torch::Tensor &image,
    const torch::Tensor &mask,
    const std::vector<float> &spacing,
    const float &l_grad,
    const float &l_eucl)
{
    torch::Tensor distance = mask.clone();

    geodesic3d_pixelqueue_cpu(image, distance, spacing, l_grad, l_eucl);

    return distance;
}