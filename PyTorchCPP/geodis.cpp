#include <torch/extension.h>
#include <vector>
#include <iostream>

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

torch::Tensor l2distance(torch::Tensor in1, torch::Tensor in2, int dim){
    return torch::sqrt(torch::sum((in1-in2) * (in1-in2), dim, true));
}

torch::Tensor l1update(torch::Tensor p, torch::Tensor q, float lambda, float dist){
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

    // top-down
    for(int h = 1; h < height; h++){
        // slice pixel and distance at p
        torch::Tensor p_val_vec = image.index({"...", h, Slice(None)});
        torch::Tensor new_dist_vec = distance.index({"...", h, Slice(None)});

        for(int wi = 0; wi < int(w_index.size()); wi++){
            float local_dist = 1.0 + abs(float(wi)-1.0);
            torch::Tensor q_val_vec = image.index({"...", h - 1, torch::tensor(w_index[wi])});
            torch::Tensor q_dis_vec = distance.index({"...", h - 1, torch::tensor(w_index[wi])});
            new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
        }

        // compute minimum distance for this row and update
        distance.index_put_({"...", h, Slice(None)}, new_dist_vec);
    }

    // bottom-up
    for(int h = height-2; h >= 0; h--){
        // slice pixel and distance at p
        torch::Tensor p_val_vec = image.index({"...", h, Slice(None)});
        torch::Tensor new_dist_vec = distance.index({"...", h, Slice(None)});

        for(int wi = 0; wi < int(w_index.size()); wi++){
            float local_dist = 1.0 + abs(float(wi)-1.0);
            torch::Tensor q_val_vec = image.index({"...", h + 1, torch::tensor(w_index[wi])});
            torch::Tensor q_dis_vec = distance.index({"...", h + 1, torch::tensor(w_index[wi])});
            new_dist_vec = torch::minimum(new_dist_vec, q_dis_vec + l1update(p_val_vec, q_val_vec, lambda, local_dist));
        }

        // compute minimum distance for this row and update
        distance.index_put_({"...", h, Slice(None)}, new_dist_vec);
    }
}


torch::Tensor generalised_geodesic2d(torch::Tensor image, torch::Tensor mask, float v, float lambda, int iterations) {

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
        // top-bottom pass
        geodesic_updown_pass(image, distance, lambda);
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);
        // left-right pass
        geodesic_updown_pass(image, distance, lambda);
        image = image.transpose(2, 3);
        distance = distance.transpose(2, 3);
    }    

    return distance;
       
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generalised_geodesic2d", &generalised_geodesic2d, "Geodesic distance 2d");
}