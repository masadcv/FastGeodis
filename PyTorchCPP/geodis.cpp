#include <torch/extension.h>
#include <vector>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace torch::indexing;

void print_shape(torch::Tensor data){
    auto num_dims = data.dim();
    std::cout << "Shape: (";
    for (int dim = 0; dim < num_dims; dim++){
        std::cout << data.size(dim);
        if(dim != num_dims-1)
            std::cout << ", ";
        else
            std::cout << ")" << std::endl;
    }
}

torch::Tensor l1distance(torch::Tensor in1, torch::Tensor in2, int dim){
    return torch::sum(torch::abs(in1-in2), dim, true);
}


float l1distance(float in1, float in2, int dim){
    return std::abs(in1-in2);
}


torch::Tensor l2distance(torch::Tensor in1, torch::Tensor in2, int dim){
    return torch::sqrt(torch::sum((in1-in2) * (in1-in2), dim, true));
}

torch::Tensor l1update(torch::Tensor p, torch::Tensor q, float lambda, float dist){
    // return torch::sqrt(pow(dist, 2.0) + lambda * torch::pow(l1distance(p, q, 1), 2));
    return sqrt(dist)/((1.0 - lambda) + lambda / (l1distance(p, q, 1) + 1e-5));
}

float l1update(float p, float q, float lambda, float dist){
    // return torch::sqrt(pow(dist, 2.0) + lambda * torch::pow(l1distance(p, q, 1), 2));
    return sqrt(dist)/((1.0 - lambda) + lambda / (l1distance(p, q, 1) + 1e-5));
}

torch::Tensor l2update(torch::Tensor p, torch::Tensor q, float lambda, float dist){
    return sqrt(dist)/((1.0 - lambda) + lambda / (l2distance(p, q, 1) + 1e-5));
}

void geodesic_updown_pass(torch::Tensor image, torch::Tensor &distance, float lambda){
    // batch, channel, height, width
    int height = distance.size(2);
    int width = distance.size(3);

    // prepare indices for up-down passes
    std::vector<int> w_index_z0, w_index_m1, w_index_p1;
    for(int wx = 0; wx < width; wx++){
        w_index_z0.push_back(wx);
        int wxm1 = (wx - 1) < 0 ? 0 : (wx - 1);
        int wxp1 = (wx + 1) >= width ? (width - 1) : (wx + 1);
        w_index_m1.push_back(wxm1);
        w_index_p1.push_back(wxp1);
    }

    std::vector<std::vector<int>> w_index;
    w_index.push_back(w_index_m1);
    w_index.push_back(w_index_z0);
    w_index.push_back(w_index_p1);

    float local_dist;
    torch::Tensor p_val_vec, new_dist_vec, q_val_vec, q_dis_vec;

    // top-down
    for(int h = 1; h < height; h++){
        // slice pixel and distance at p
        p_val_vec = image.index({"...", h, Slice(None)});
        new_dist_vec = distance.index({"...", h, Slice(None)});

        for(int wi = 0; wi < int(w_index.size()); wi++){
            local_dist = 1.0 + abs(float(wi)-1.0);
            q_val_vec = image.index({"...", h - 1, torch::tensor(w_index[wi])});
            q_dis_vec = distance.index({"...", h - 1, torch::tensor(w_index[wi])});
            new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
        }
        // compute minimum distance for this row and update
        distance.index_put_({"...", h, Slice(None)}, new_dist_vec);
    }

    // bottom-up
    for(int h = height-2; h >= 0; h--){
        // slice pixel and distance at p
        p_val_vec = image.index({"...", h, Slice(None)});
        new_dist_vec = distance.index({"...", h, Slice(None)});

        for(int wi = 0; wi < int(w_index.size()); wi++){
            local_dist = 1.0 + abs(float(wi)-1.0);
            q_val_vec = image.index({"...", h + 1, torch::tensor(w_index[wi])});
            q_dis_vec = distance.index({"...", h + 1, torch::tensor(w_index[wi])});
            new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
        }
        // compute minimum distance for this row and update
        distance.index_put_({"...", h, Slice(None)}, new_dist_vec);
    }
}

void geodesic_updown_pass_openmp(torch::Tensor image, torch::Tensor &distance, float lambda){
    // batch, channel, height, width
    int height = distance.size(2);
    int width = distance.size(3);

    auto imageptr = image.accessor<float, 4>();
    auto distanceptr = distance.accessor<float, 4>();

    // top-down
    for(int h = 1; h < height; h++){
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
           #pragma omp parallel for
        #endif
        for(int w = 0; w < width; w++)
        {
            float pval = imageptr[0][0][h][w];
            float new_dist = distanceptr[0][0][h][w];

            for(int wi = -1; wi < 2; wi++)
            {
                int wind = w + wi;
                if(wind < 0 || wind >= width) continue;

                float local_dist = 1.0 + float(std::abs(wi));
                
                float qval = imageptr[0][0][h-1][wind];
                float qdist = distanceptr[0][0][h-1][wind];

                new_dist = std::min(new_dist, qdist + l1update(pval, qval, lambda, local_dist));

            }
            distanceptr[0][0][h][w] = new_dist;
        }
    }

    // bottom-up
    for(int h = height-2; h >= 0; h--){
        // use openmp to parallelise the loop over width
        #ifdef _OPENMP
           #pragma omp parallel for
        #endif
        for(int w = 0; w < width; w++)
        {
            float pval = imageptr[0][0][h][w];
            float new_dist = distanceptr[0][0][h][w];

            for(int wi = -1; wi < 2; wi++)
            {
                int wind = w + wi;
                if(wind < 0 || wind >= width) continue;

                float local_dist = 1.0 + float(std::abs(wi));
                
                float qval = imageptr[0][0][h+1][wind];
                float qdist = distanceptr[0][0][h+1][wind];

                new_dist = std::min(new_dist, qdist + l1update(pval, qval, lambda, local_dist));

            }
            distanceptr[0][0][h][w] = new_dist;
        }
    }
}

torch::Tensor generalised_geodesic2d(torch::Tensor image, torch::Tensor mask, float v, float lambda, int iterations) {
    #ifdef _OPENMP
        std::cout << "OpenMP present, using OpenMP" << std::endl;
    #else
        std::cout << "OpenP not present" << std::endl;
    #endif

    // initialise distance with soft mask
    torch::Tensor distance = v * mask.clone();

    // check input dimensions
    int num_dims = distance.dim();
    if(num_dims != 4){
        throw std::runtime_error(
          "function only supports 2D spatial inputs, received " + std::to_string(num_dims - 2));
    }
    
    // iteratively run the distance transform
    for(int itr = 0; itr < iterations; itr++){
        // top-bottom - width*, height
        geodesic_updown_pass_openmp(image, distance, lambda);
        
        // left-right - height*, width
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);
        geodesic_updown_pass_openmp(image, distance, lambda);
        // tranpose back to original - height, width
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);

        // * indicates the current direction of pass
    }    
    return distance;
}

void geodesic_frontback_pass(torch::Tensor image, torch::Tensor &distance, std::vector<float> spacingsq, float lambda){
    // batch, channel, height, width
    int depth = distance.size(2);
    int height = distance.size(3);
    int width = distance.size(4);

    // prepare indices for front-back passes
    std::vector<int> w_index_z0, w_index_m1, w_index_p1;
    for(int wx = 0; wx < width; wx++){
        w_index_z0.push_back(wx);
        int wxm1 = (wx - 1) < 0 ? 0 : (wx - 1);
        int wxp1 = (wx + 1) >= width ? (width - 1) : (wx + 1);
        w_index_m1.push_back(wxm1);
        w_index_p1.push_back(wxp1);
    }
    std::vector<std::vector<int>> w_index;
    w_index.push_back(w_index_m1);
    w_index.push_back(w_index_z0);
    w_index.push_back(w_index_p1);

    std::vector<int> h_index_z0, h_index_m1, h_index_p1;
    for(int hx = 0; hx < height; hx++){
        h_index_z0.push_back(hx);
        int hxm1 = (hx - 1) < 0 ? 0 : (hx - 1);
        int hxp1 = (hx + 1) >= height ? (height - 1) : (hx + 1);
        h_index_m1.push_back(hxm1);
        h_index_p1.push_back(hxp1);
    }
    std::vector<std::vector<int>> h_index;
    h_index.push_back(h_index_m1);
    h_index.push_back(h_index_z0);
    h_index.push_back(h_index_p1);

    float local_dist;
    torch::Tensor p_val_vec, new_dist_vec, q_val_vec, q_dis_vec;

    // front-back
    for(int z = 1; z < depth; z++){
        // slice pixel and distance at p
        p_val_vec = image.index({"...", z, Slice(None), Slice(None)});
        new_dist_vec = distance.index({"...", z, Slice(None), Slice(None)});
        for(int hi = 0; hi < int(h_index.size()); hi++){
            for(int wi = 0; wi < int(w_index.size()); wi++){
                local_dist = spacingsq[0];
                local_dist += pow(float(hi)-1.0, 2.0) * spacingsq[1];
                local_dist += pow(float(wi)-1.0, 2.0) * spacingsq[2];

                q_val_vec = image.index({"...", z - 1, torch::tensor(h_index[hi]), Slice(None)}).index({"...", torch::tensor(w_index[wi])});
                q_dis_vec = distance.index({"...", z - 1, torch::tensor(h_index[hi]), Slice(None)}).index({"...", torch::tensor(w_index[wi])});
                new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
            }
        // compute minimum distance for this plane and update
        distance.index_put_({"...", z, Slice(None), Slice(None)}, new_dist_vec);
        }
    }

    // back-front
    for(int z = depth-2; z >= 0; z--){
        // slice pixel and distance at p
        p_val_vec = image.index({"...", z, Slice(None), Slice(None)});
        new_dist_vec = distance.index({"...", z, Slice(None), Slice(None)});
        for(int hi = 0; hi < int(h_index.size()); hi++){
            for(int wi = 0; wi < int(w_index.size()); wi++){
                local_dist = spacingsq[0];
                local_dist += pow(float(hi)-1.0, 2.0) * spacingsq[1];
                local_dist += pow(float(wi)-1.0, 2.0) * spacingsq[2];

                q_val_vec = image.index({"...", z + 1, torch::tensor(h_index[hi]), Slice(None)}).index({"...", torch::tensor(w_index[wi])});
                q_dis_vec = distance.index({"...", z + 1, torch::tensor(h_index[hi]), Slice(None)}).index({"...", torch::tensor(w_index[wi])});
                new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
            }
        // compute minimum distance for this plane and update
        distance.index_put_({"...", z, Slice(None), Slice(None)}, new_dist_vec);
        }
    }
}

void geodesic_frontback_pass_openmp(torch::Tensor image, torch::Tensor &distance, std::vector<float> spacingsq, float lambda){
    // batch, channel, depth, height, width
    int depth = distance.size(2);
    int height = distance.size(3);
    int width = distance.size(4);

    auto imageptr = image.accessor<float, 5>();
    auto distanceptr = distance.accessor<float, 5>();

    std::vector<float> local_distances;
    for(int hi = -1; hi < 2; hi++)
    {
        for(int wi = -1; wi < 2; wi++)
        {
            float local_dist = spacingsq[0];
            local_dist += float(std::abs(hi)) * spacingsq[1];
            local_dist += float(std::abs(wi)) * spacingsq[2];

            local_distances.push_back(local_dist);
        }
    }

    // front-back
    for(int z = 1; z < depth; z++){
        // use openmp to parallelise the loops over height and width
        #ifdef _OPENMP
           #pragma omp parallel for collapse(2)
        #endif
        for(int h = 0; h < height; h++)
        {
            for(int w = 0; w < width; w++)
            {
                float pval = imageptr[0][0][z][h][w];
                float new_dist = distanceptr[0][0][z][h][w];
                for(int hi = 0; hi < 3; hi++)
                {
                    for(int wi = 0; wi < 3; wi++)
                    {   
                        int hind = h + hi - 1;
                        int wind = w + wi - 1;
                        
                        if(hind < 0 || hind >= height) continue;
                        if(wind < 0 || wind >= width) continue;
                        float local_dist = local_distances[hi * 3 + wi];

                        float qval = imageptr[0][0][z-1][hind][wind];
                        float qdist = distanceptr[0][0][z-1][hind][wind];
                        
                        new_dist = std::min(new_dist, qdist + l1update(pval, qval, lambda, local_dist));
                    }
                }
                distanceptr[0][0][z][h][w] = new_dist;
            }
        }
    }

    // back-front
    for(int z = depth-2; z >= 0; z--){
        // use openmp to parallelise the loops over height and width
        #ifdef _OPENMP
           #pragma omp parallel for collapse(2)
        #endif
        for(int h = 0; h < height; h++)
        {
            for(int w = 0; w < width; w++)
            {
                float pval = imageptr[0][0][z][h][w];
                float new_dist = distanceptr[0][0][z][h][w];
                for(int hi = 0; hi < 3; hi++)
                {
                    for(int wi = 0; wi < 3; wi++)
                    {   
                        int hind = h + hi - 1;
                        int wind = w + wi - 1;

                        if(hind < 0 || hind >= height) continue;
                        if(wind < 0 || wind >= width) continue;

                        float local_dist = local_distances[hi * 3 + wi];

                        float qval = imageptr[0][0][z+1][hind][wind];
                        float qdist = distanceptr[0][0][z+1][hind][wind];
                        
                        new_dist = std::min(new_dist, qdist + l1update(pval, qval, lambda, local_dist));
                    }
                }
               distanceptr[0][0][z][h][w] = new_dist;
            }
        }
    }
}


torch::Tensor generalised_geodesic3d(torch::Tensor image, torch::Tensor mask, std::vector<float> spacing, float v, float lambda, int iterations) {
    #ifdef _OPENMP
        std::cout << "OpenMP present, using OpenMP" << std::endl;
    #else
        std::cout << "OpenP not present" << std::endl;
    #endif

    // initialise distance with soft mask
    torch::Tensor distance = v * mask.clone();

    // check input dimensions
    int num_dims = distance.dim();
    if(num_dims != 5){
        throw std::runtime_error(
          "function only supports 3D spatial inputs, received " + std::to_string(num_dims - 2));
    }
    
    if(spacing.size() != 3){
        throw std::runtime_error(
          "function only supports 3D spacing inputs, received " + std::to_string(spacing.size()));
    }

    // square spacing
    for(int t = 0; t < int(spacing.size()); t++)
        spacing[t] = spacing[t] * spacing[t];
    
    // iteratively run the distance transform
    for(int itr = 0; itr < iterations; itr++){
        // front-back - depth*, height, width
        geodesic_frontback_pass_openmp(image, distance, spacing, lambda);

        // top-bottom - height*, depth, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        geodesic_frontback_pass_openmp(image, distance, spacing, lambda);
        // transpose back to original depth, height, width
        image = torch::transpose(image, 3, 2);
        distance = torch::transpose(distance, 3, 2);
        
        // left-right - width*, height, depth
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);
        geodesic_frontback_pass_openmp(image, distance, spacing, lambda);
        // transpose back to original depth, height, width
        image = torch::transpose(image, 4, 2);
        distance = torch::transpose(distance, 4, 2);

        // * indicates the current direction of pass
    }    
    return distance;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generalised_geodesic2d", &generalised_geodesic2d, "Geodesic distance 2d");
  m.def("generalised_geodesic3d", &generalised_geodesic3d, "Geodesic distance 3d");
}