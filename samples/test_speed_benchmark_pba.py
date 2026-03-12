"""
Benchmark script for PBA+ (Parallel Banding Algorithm Plus) Exact EDT.

Compares:
- scipy.ndimage.distance_transform_edt (CPU reference)
- FastGeodis raster scanning GPU (approximate)
- FastGeodis PBA+ GPU (exact)

Also computes accuracy errors vs scipy reference.
"""

import json
import os
import time
from functools import wraps

import FastGeodis
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, some benchmarks will be skipped")


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
def scipy_edt_2d(mask_np):
    """Scipy reference EDT (CPU)"""
    return ndimage.distance_transform_edt(mask_np)


@timing
def scipy_edt_3d(mask_np):
    """Scipy reference EDT (CPU)"""
    return ndimage.distance_transform_edt(mask_np)


@timing
def generalised_geodesic2d_raster_gpu(I, S, v, lamb, iter):
    """FastGeodis raster scanning (GPU, approximate)"""
    result = FastGeodis.generalised_geodesic2d(I, S, v, lamb, iter)
    torch.cuda.synchronize()
    return result


@timing
def generalised_geodesic3d_raster_gpu(I, S, spacing, v, lamb, iter):
    """FastGeodis raster scanning 3D (GPU, approximate)"""
    result = FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamb, iter)
    torch.cuda.synchronize()
    return result


@timing
def exact_euclidean2d_pba_gpu(mask, spacing):
    """FastGeodis PBA+ exact EDT (GPU)"""
    result = FastGeodis.exact_euclidean2d(mask, spacing)
    torch.cuda.synchronize()
    return result


@timing
def exact_euclidean3d_pba_gpu(mask, spacing):
    """FastGeodis PBA+ exact EDT 3D (GPU)"""
    result = FastGeodis.exact_euclidean3d(mask, spacing)
    torch.cuda.synchronize()
    return result


def test2d():
    """Benchmark 2D EDT methods"""
    num_runs = 5

    sizes_to_test = [64, 128, 256, 512, 1024, 2048]
    print(sizes_to_test)

    time_taken_dict = {}
    error_dict = {'raster_error': [], 'pba_error': []}

    # Test scipy
    if SCIPY_AVAILABLE:
        time_taken_dict['scipy_edt_2d'] = []
        for size in sizes_to_test:
            mask_np = np.ones((size, size), dtype=np.float32)
            mask_np[size // 2, size // 2] = 0  # Single seed point

            tic = time.time()
            for _ in range(num_runs):
                scipy_edt_2d(mask_np)
            time_taken_dict['scipy_edt_2d'].append((time.time() - tic) / num_runs)
        print()

    # Test raster GPU
    time_taken_dict['generalised_geodesic2d_raster_gpu'] = []
    for size in sizes_to_test:
        image = torch.ones((1, 1, size, size), device='cuda')
        seed = torch.ones((1, 1, size, size), device='cuda')
        seed[:, :, size // 2, size // 2] = 0.0

        tic = time.time()
        for _ in range(num_runs):
            generalised_geodesic2d_raster_gpu(image, seed, 1e10, 0.0, 4)
        time_taken_dict['generalised_geodesic2d_raster_gpu'].append((time.time() - tic) / num_runs)
    print()

    # Test PBA+ GPU
    time_taken_dict['exact_euclidean2d_pba_gpu'] = []
    for size in sizes_to_test:
        mask = torch.ones((1, 1, size, size), device='cuda')
        mask[:, :, size // 2, size // 2] = 0.0

        tic = time.time()
        for _ in range(num_runs):
            exact_euclidean2d_pba_gpu(mask, [1.0, 1.0])
        time_taken_dict['exact_euclidean2d_pba_gpu'].append((time.time() - tic) / num_runs)
    print()

    # Compute errors vs scipy
    if SCIPY_AVAILABLE:
        print("Computing errors vs scipy reference...")
        for size in sizes_to_test:
            mask_np = np.ones((size, size), dtype=np.float32)
            mask_np[size // 2, size // 2] = 0

            # Scipy reference
            scipy_dist = ndimage.distance_transform_edt(mask_np)

            # Raster GPU
            image = torch.ones((1, 1, size, size), device='cuda')
            seed = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to('cuda')
            raster_dist = FastGeodis.generalised_geodesic2d(image, seed, 1e10, 0.0, 4)
            raster_dist_np = raster_dist[0, 0].cpu().numpy()

            # PBA+ GPU
            mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to('cuda')
            pba_dist = FastGeodis.exact_euclidean2d(mask_torch, [1.0, 1.0])
            pba_dist_np = pba_dist[0, 0].cpu().numpy()

            raster_err = np.abs(scipy_dist - raster_dist_np).max()
            pba_err = np.abs(scipy_dist - pba_dist_np).max()

            error_dict['raster_error'].append(float(raster_err))
            error_dict['pba_error'].append(float(pba_err))
            print(f"  Size {size}x{size}: Raster err={raster_err:.2e}, PBA+ err={pba_err:.2e}")

    time_taken_dict['spatial_dim'] = sizes_to_test
    time_taken_dict.update(error_dict)

    return sizes_to_test, time_taken_dict


def test3d():
    """Benchmark 3D EDT methods"""
    num_runs = 3
    spacing = [1.0, 1.0, 1.0]

    sizes_to_test = [32, 64, 128]
    print(sizes_to_test)

    time_taken_dict = {}
    error_dict = {'raster_error': [], 'pba_error': []}

    # Test scipy
    if SCIPY_AVAILABLE:
        time_taken_dict['scipy_edt_3d'] = []
        for size in sizes_to_test:
            mask_np = np.ones((size, size, size), dtype=np.float32)
            mask_np[size // 2, size // 2, size // 2] = 0

            tic = time.time()
            for _ in range(num_runs):
                scipy_edt_3d(mask_np)
            time_taken_dict['scipy_edt_3d'].append((time.time() - tic) / num_runs)
        print()

    # Test raster GPU
    time_taken_dict['generalised_geodesic3d_raster_gpu'] = []
    for size in sizes_to_test:
        image = torch.ones((1, 1, size, size, size), device='cuda')
        seed = torch.ones((1, 1, size, size, size), device='cuda')
        seed[:, :, size // 2, size // 2, size // 2] = 0.0

        tic = time.time()
        for _ in range(num_runs):
            generalised_geodesic3d_raster_gpu(image, seed, spacing, 1e10, 0.0, 4)
        time_taken_dict['generalised_geodesic3d_raster_gpu'].append((time.time() - tic) / num_runs)
    print()

    # Test PBA+ GPU
    time_taken_dict['exact_euclidean3d_pba_gpu'] = []
    for size in sizes_to_test:
        mask = torch.ones((1, 1, size, size, size), device='cuda')
        mask[:, :, size // 2, size // 2, size // 2] = 0.0

        tic = time.time()
        for _ in range(num_runs):
            exact_euclidean3d_pba_gpu(mask, spacing)
        time_taken_dict['exact_euclidean3d_pba_gpu'].append((time.time() - tic) / num_runs)
    print()

    # Compute errors vs scipy
    if SCIPY_AVAILABLE:
        print("Computing errors vs scipy reference...")
        for size in sizes_to_test:
            mask_np = np.ones((size, size, size), dtype=np.float32)
            mask_np[size // 2, size // 2, size // 2] = 0

            # Scipy reference
            scipy_dist = ndimage.distance_transform_edt(mask_np)

            # Raster GPU
            image = torch.ones((1, 1, size, size, size), device='cuda')
            seed = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to('cuda')
            raster_dist = FastGeodis.generalised_geodesic3d(image, seed, spacing, 1e10, 0.0, 4)
            raster_dist_np = raster_dist[0, 0].cpu().numpy()

            # PBA+ GPU
            mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to('cuda')
            pba_dist = FastGeodis.exact_euclidean3d(mask_torch, spacing)
            pba_dist_np = pba_dist[0, 0].cpu().numpy()

            raster_err = np.abs(scipy_dist - raster_dist_np).max()
            pba_err = np.abs(scipy_dist - pba_dist_np).max()

            error_dict['raster_error'].append(float(raster_err))
            error_dict['pba_error'].append(float(pba_err))
            print(f"  Size {size}x{size}x{size}: Raster err={raster_err:.2e}, PBA+ err={pba_err:.2e}")

    time_taken_dict['spatial_dim'] = sizes_to_test
    time_taken_dict.update(error_dict)

    return sizes_to_test, time_taken_dict


def save_timing_plot(sizes, time_taken_dict, figname, dim='2D'):
    """Save timing comparison plot"""
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    for key in time_taken_dict.keys():
        if key in ['spatial_dim', 'raster_error', 'pba_error']:
            continue
        if 'scipy' in key:
            plt.plot(sizes, time_taken_dict[key], 'b-s',
                    label="scipy EDT (CPU)", linewidth=2, markersize=8)
        elif 'raster' in key:
            plt.plot(sizes, time_taken_dict[key], 'r-^',
                    label="FastGeodis Raster (GPU)", linewidth=2, markersize=8)
        elif 'pba' in key or 'exact' in key:
            plt.plot(sizes, time_taken_dict[key], 'g-o',
                    label="FastGeodis PBA+ (GPU)", linewidth=2, markersize=8)

    plt.legend(fontsize=12)
    plt.xticks(sizes, [str(s) for s in sizes], rotation=45)
    plt.title(f"{dim} EDT Performance Comparison", fontsize=14)
    plt.xlabel("Spatial size", fontsize=12)
    plt.ylabel("Execution time (seconds)", fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join("figures", figname + ".png"), dpi=150)
    plt.close()

    # Save JSON
    with open(os.path.join('figures', figname + '.json'), 'w') as fp:
        json.dump(time_taken_dict, fp, indent=4)


def save_error_plot(sizes, time_taken_dict, figname, dim='2D'):
    """Save accuracy comparison plot"""
    if 'raster_error' not in time_taken_dict or 'pba_error' not in time_taken_dict:
        return

    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    plt.plot(sizes, time_taken_dict['raster_error'], 'r-^',
            label='FastGeodis Raster (GPU)', linewidth=2, markersize=8)
    plt.plot(sizes, time_taken_dict['pba_error'], 'g-o',
            label='FastGeodis PBA+ (GPU)', linewidth=2, markersize=8)

    plt.legend(fontsize=12)
    plt.xticks(sizes, [str(s) for s in sizes], rotation=45)
    plt.title(f"{dim} EDT Max Error vs scipy Reference", fontsize=14)
    plt.xlabel("Spatial size", fontsize=12)
    plt.ylabel("Max absolute error", fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join("figures", figname + "_error.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)

    # 2D benchmarks
    sizes, ttdict = test2d()
    save_timing_plot(sizes, ttdict, "experiment_2d_pba", "2D")
    save_error_plot(sizes, ttdict, "experiment_2d_pba", "2D")

    # 3D benchmarks
    sizes, ttdict = test3d()
    save_timing_plot(sizes, ttdict, "experiment_3d_pba", "3D")
    save_error_plot(sizes, ttdict, "experiment_3d_pba", "3D")
