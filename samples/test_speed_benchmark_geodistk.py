import json
import GeodisTK
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
import SimpleITK as sitk
import FastGeodis
import time
import os

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap

@timing
def generalised_geodesic_distance_2d(I, S, v, lamb, iter):
    return GeodisTK.geodesic2d_raster_scan(I, 1-S.astype(np.uint8), lamb, iter)

@timing
def generalised_geodesic2d_raster_cpu(I, S, v, lamb, iter):
    return FastGeodis.generalised_geodesic2d(I, S, v, lamb, iter)

@timing
def generalised_geodesic2d_raster_gpu(I, S, v, lamb, iter):
    return FastGeodis.generalised_geodesic2d(I, S, v, lamb, iter)

@timing
def generalised_geodesic_distance_3d(I, S, spacing, v, lamb, iter):
    return GeodisTK.geodesic3d_raster_scan(I, 1-S.astype(np.uint8), spacing, lamb, iter)

@timing
def generalised_geodesic3d_raster_cpu(I, S, spacing, v, lamb, iter):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamb, iter)

@timing
def generalised_geodesic3d_raster_gpu(I, S, spacing, v, lamb, iter):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamb, iter)
    
func_to_test_2d = [generalised_geodesic_distance_2d, generalised_geodesic2d_raster_cpu, generalised_geodesic2d_raster_gpu]
func_to_test_3d = [generalised_geodesic_distance_3d, generalised_geodesic3d_raster_cpu, generalised_geodesic3d_raster_gpu]

def test2d():
    num_runs = 10

    sizes_to_test = [64, 64*2, 64*(2**2), 64*(2**3), 64*(2**4), 64*(2**5)]#, 64*(2**6)]
    print(sizes_to_test)
    time_taken_dict = dict()
    for func in func_to_test_2d:
        time_taken_dict[func.__name__] = []
        for size in sizes_to_test:
            image = torch.rand((1, 1, size, size))
            seed = torch.ones((1, 1, size, size))
            seed[:, :, size//2, size//2] = 0.0

            if 'gpu' in func.__name__:
                image = image.to('cuda').contiguous()
                seed = seed.to('cuda').contiguous()
            
            tic = time.time()
            for i in range(num_runs):
                if 'cpu' in func.__name__:
                    func(image, seed, 10000, 1.0, 2)
                elif 'gpu' in func.__name__ and torch.cuda.is_available():
                    func(image, seed, 10000, 1.0, 2)
                else:
                    func(np.squeeze(image.cpu().numpy()), np.squeeze(seed.cpu().numpy()), 10000, 1.0, 2)

            time_taken_dict[func.__name__].append((time.time() - tic)/num_runs)
        print()

    return sizes_to_test, time_taken_dict

def test3d():
    num_runs = 10
    spacing = [1.0, 1.0, 1.0]

    sizes_to_test = [64*(2**0), 64*(2**1), 64*(2**2), 64*(2**3)]#, 64*(2**4)]#, 64*(2**5), 64*(2**6)]
    print(sizes_to_test)
    time_taken_dict = dict()
    for func in func_to_test_3d:
        time_taken_dict[func.__name__] = []
        for size in sizes_to_test:
            image = torch.rand((1, 1, size, size, size))
            seed = torch.ones((1, 1, size, size, size))
            seed[:, :, size//2, size//2, size//2] = 0.0
            
            if 'gpu' in func.__name__:
                image = image.to('cuda').contiguous()
                seed = seed.to('cuda').contiguous()

            tic = time.time()
            for i in range(num_runs):
                if 'cpu' in func.__name__:
                    func(image, seed, spacing, 10000, 1.0, 2)
                elif 'gpu' in func.__name__ and torch.cuda.is_available():
                    func(image, seed, spacing, 10000, 1.0, 2)
                else:
                    func(np.squeeze(image.cpu().numpy()), np.squeeze(seed.cpu().numpy()), spacing, 10000, 1.0, 2)

            time_taken_dict[func.__name__].append((time.time() - tic)/num_runs)
        print()

    return sizes_to_test, time_taken_dict

def save_plot(sizes, time_taken_dict, figname):
    plt.figure()
    plt.grid()
    for key in time_taken_dict.keys():
        if 'cpu' in key:
            plt.plot(sizes, time_taken_dict[key], 'm-o', label='FastGeodis (cpu)')
        elif 'gpu' in key:
            plt.plot(sizes, time_taken_dict[key], 'g-o', label='FastGeodis (gpu)')
        else:
            plt.plot(sizes, time_taken_dict[key], 'r-o', label='GeodisTK')
    plt.legend()
    plt.xticks(sizes, [str(s) for s in sizes], rotation=45)
    plt.title(figname)
    plt.xlabel('Spatial size')
    plt.ylabel('Execution time (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', figname + '.png'))
    time_taken_dict['spatial_dim'] = sizes
    with open(os.path.join('figures', figname + '.json'), 'w') as fp:
        json.dump(time_taken_dict, fp, indent=4)



if __name__ == "__main__":
    sizes, ttdict = test2d()
    save_plot(sizes, ttdict, 'experiment_2d')

    sizes, ttdict = test3d()
    save_plot(sizes, ttdict, 'experiment_3d')