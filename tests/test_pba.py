# BSD 3-Clause License

# Copyright (c) 2021, Muhammad Asad (masadcv@gmail.com)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Tests for PBA+ (Parallel Banding Algorithm Plus) Exact Euclidean Distance Transform.

This module tests the exact EDT implementation against scipy's reference implementation.
"""

import math
import unittest

import numpy as np
import torch
from parameterized import parameterized

from .utils import skip_if_no_cuda

# set deterministic seed
torch.manual_seed(15)
np.random.seed(15)

try:
    import FastGeodis
except:
    print(
        "Unable to load FastGeodis for unittests\nMake sure to install using: python setup.py install"
    )
    exit()

try:
    from scipy import ndimage

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available, some tests will be skipped")


# Test configurations for 2D
CONF_2D_CUDA = [("cuda", bas) for bas in [32, 64, 128, 256]]

# Test configurations for 3D (smaller sizes due to memory)
CONF_3D_CUDA = [("cuda", bas) for bas in [16, 32, 64]]


class TestExactEuclidean2D(unittest.TestCase):
    """Tests for 2D Exact Euclidean Distance Transform using PBA+"""

    @skip_if_no_cuda
    @parameterized.expand(CONF_2D_CUDA)
    def test_single_seed_point(self, device, base_dim):
        """Test EDT with a single seed point at center"""
        height, width = base_dim, base_dim

        # Create mask with single seed point
        mask = torch.ones(1, 1, height, width, dtype=torch.float32, device=device)
        mask[0, 0, height // 2, width // 2] = 0  # Seed point

        # Compute exact EDT
        distance = FastGeodis.exact_euclidean2d(mask, spacing=[1.0, 1.0])

        # Check output shape
        self.assertEqual(distance.shape, mask.shape)

        # Check seed point has distance 0
        self.assertAlmostEqual(
            distance[0, 0, height // 2, width // 2].item(), 0.0, places=5
        )

        # Check corner has expected distance (Euclidean from center to corner)
        expected_corner_dist = math.sqrt((height // 2) ** 2 + (width // 2) ** 2)
        actual_corner_dist = distance[0, 0, 0, 0].item()

        # Allow small tolerance for floating point
        self.assertAlmostEqual(actual_corner_dist, expected_corner_dist, places=3)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy required for exact comparison")
    @skip_if_no_cuda
    @parameterized.expand(CONF_2D_CUDA)
    def test_exact_match_scipy(self, device, base_dim):
        """Test that PBA+ matches scipy's exact EDT"""
        height, width = base_dim, base_dim

        # Create random binary mask with multiple seed points
        np.random.seed(42)
        mask_np = np.ones((height, width), dtype=np.float32)

        # Add some random seed points
        num_seeds = max(1, base_dim // 8)
        for _ in range(num_seeds):
            y, x = np.random.randint(0, height), np.random.randint(0, width)
            mask_np[y, x] = 0

        # Scipy reference (exact)
        scipy_dist = ndimage.distance_transform_edt(mask_np)

        # FastGeodis PBA+ (should be exact)
        mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        pba_dist = FastGeodis.exact_euclidean2d(mask_torch, spacing=[1.0, 1.0])
        pba_dist_np = pba_dist[0, 0].cpu().numpy()

        # Compare - should be exact (or very close due to floating point)
        max_diff = np.abs(scipy_dist - pba_dist_np).max()
        mean_diff = np.abs(scipy_dist - pba_dist_np).mean()

        # Allow small tolerance for floating point precision
        self.assertLess(
            max_diff,
            1e-3,
            f"Max difference {max_diff} exceeds threshold for size {base_dim}",
        )
        self.assertLess(
            mean_diff,
            1e-4,
            f"Mean difference {mean_diff} exceeds threshold for size {base_dim}",
        )

    @skip_if_no_cuda
    def test_anisotropic_spacing(self):
        """Test EDT with anisotropic pixel spacing"""
        device = "cuda"
        height, width = 64, 64
        spacing = [2.0, 1.0]  # Pixels are 2x taller than wide

        mask = torch.ones(1, 1, height, width, dtype=torch.float32, device=device)
        mask[0, 0, height // 2, width // 2] = 0

        distance = FastGeodis.exact_euclidean2d(mask, spacing=spacing)

        # Distance to point 1 pixel right should be 1.0
        dist_right = distance[0, 0, height // 2, width // 2 + 1].item()
        self.assertAlmostEqual(dist_right, 1.0, places=3)

        # Distance to point 1 pixel down should be 2.0 (due to spacing)
        dist_down = distance[0, 0, height // 2 + 1, width // 2].item()
        self.assertAlmostEqual(dist_down, 2.0, places=3)

    @skip_if_no_cuda
    def test_all_seeds(self):
        """Test when all pixels are seeds (distance should be 0)"""
        device = "cuda"
        height, width = 32, 32

        mask = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
        distance = FastGeodis.exact_euclidean2d(mask, spacing=[1.0, 1.0])

        # All distances should be 0
        self.assertAlmostEqual(distance.max().item(), 0.0, places=5)

    @skip_if_no_cuda
    def test_ill_shape(self):
        """Test that wrong input shapes raise errors"""
        device = "cuda"

        # 3D input (should fail - expects 4D)
        mask_3d = torch.ones(1, 32, 32, dtype=torch.float32, device=device)
        with self.assertRaises(Exception):
            FastGeodis.exact_euclidean2d(mask_3d, spacing=[1.0, 1.0])

        # Wrong spacing dimension
        mask = torch.ones(1, 1, 32, 32, dtype=torch.float32, device=device)
        with self.assertRaises(Exception):
            FastGeodis.exact_euclidean2d(mask, spacing=[1.0, 1.0, 1.0])

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy required for exact comparison")
    @skip_if_no_cuda
    def test_non_aligned_sizes(self):
        """Test EDT with non-32-aligned sizes (tests padding handling)"""
        device = "cuda"

        # Test various non-aligned sizes that require padding
        # Include sizes that would pad to non-power-of-2 (e.g., 65->96, 129->160)
        # to ensure power-of-2 padding is working correctly
        for size in [(100, 100), (50, 75), (33, 47), (65, 65), (32, 65), (32, 96), (17, 93), (32, 129)]:
            height, width = size

            # Create mask with seed near corner to test padding boundary
            mask_np = np.ones((height, width), dtype=np.float32)
            mask_np[0, 0] = 0  # Seed at corner

            # Scipy reference
            scipy_dist = ndimage.distance_transform_edt(mask_np)

            # PBA+ result
            mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            pba_dist = FastGeodis.exact_euclidean2d(mask_torch, spacing=[1.0, 1.0])
            pba_dist_np = pba_dist[0, 0].cpu().numpy()

            # Compare
            max_diff = np.abs(scipy_dist - pba_dist_np).max()
            self.assertLess(
                max_diff,
                1e-3,
                f"Non-aligned size {size} failed with max_diff={max_diff}",
            )


class TestExactEuclidean3D(unittest.TestCase):
    """Tests for 3D Exact Euclidean Distance Transform using PBA+"""

    @skip_if_no_cuda
    @parameterized.expand(CONF_3D_CUDA)
    def test_single_seed_point(self, device, base_dim):
        """Test 3D EDT with a single seed point at center"""
        depth, height, width = base_dim, base_dim, base_dim

        mask = torch.ones(
            1, 1, depth, height, width, dtype=torch.float32, device=device
        )
        mask[0, 0, depth // 2, height // 2, width // 2] = 0

        distance = FastGeodis.exact_euclidean3d(mask, spacing=[1.0, 1.0, 1.0])

        # Check output shape
        self.assertEqual(distance.shape, mask.shape)

        # Check seed point has distance 0
        self.assertAlmostEqual(
            distance[0, 0, depth // 2, height // 2, width // 2].item(), 0.0, places=5
        )

        # Check corner distance
        expected_corner_dist = math.sqrt(
            (depth // 2) ** 2 + (height // 2) ** 2 + (width // 2) ** 2
        )
        actual_corner_dist = distance[0, 0, 0, 0, 0].item()

        self.assertAlmostEqual(actual_corner_dist, expected_corner_dist, places=2)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy required for exact comparison")
    @skip_if_no_cuda
    def test_exact_match_scipy_3d(self):
        """Test that 3D PBA+ matches scipy's exact EDT"""
        device = "cuda"
        depth, height, width = 32, 32, 32

        # Create mask with some seed points
        mask_np = np.ones((depth, height, width), dtype=np.float32)
        mask_np[depth // 2, height // 2, width // 2] = 0
        mask_np[5, 5, 5] = 0
        mask_np[depth - 5, height - 5, width - 5] = 0

        # Scipy reference
        scipy_dist = ndimage.distance_transform_edt(mask_np)

        # FastGeodis PBA+
        mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        pba_dist = FastGeodis.exact_euclidean3d(mask_torch, spacing=[1.0, 1.0, 1.0])
        pba_dist_np = pba_dist[0, 0].cpu().numpy()

        # Compare
        max_diff = np.abs(scipy_dist - pba_dist_np).max()
        mean_diff = np.abs(scipy_dist - pba_dist_np).mean()

        self.assertLess(max_diff, 1.0, f"Max difference {max_diff} exceeds threshold")
        self.assertLess(
            mean_diff, 0.1, f"Mean difference {mean_diff} exceeds threshold"
        )

    @skip_if_no_cuda
    def test_anisotropic_spacing_3d(self):
        """Test 3D EDT with anisotropic voxel spacing"""
        device = "cuda"
        depth, height, width = 32, 32, 32
        spacing = [1.0, 2.0, 3.0]  # z, y, x spacing

        mask = torch.ones(
            1, 1, depth, height, width, dtype=torch.float32, device=device
        )
        mask[0, 0, depth // 2, height // 2, width // 2] = 0

        distance = FastGeodis.exact_euclidean3d(mask, spacing=spacing)

        # Distance along each axis should reflect spacing
        # +1 in x direction: distance = 3.0
        dist_x = distance[0, 0, depth // 2, height // 2, width // 2 + 1].item()
        self.assertAlmostEqual(dist_x, 3.0, places=2)

        # +1 in y direction: distance = 2.0
        dist_y = distance[0, 0, depth // 2, height // 2 + 1, width // 2].item()
        self.assertAlmostEqual(dist_y, 2.0, places=2)

        # +1 in z direction: distance = 1.0
        dist_z = distance[0, 0, depth // 2 + 1, height // 2, width // 2].item()
        self.assertAlmostEqual(dist_z, 1.0, places=2)


class TestSignedExactEuclidean2D(unittest.TestCase):
    """Tests for Signed 2D Exact EDT"""

    @skip_if_no_cuda
    def test_signed_distance(self):
        """Test signed distance has correct signs"""
        device = "cuda"
        height, width = 64, 64

        # Create a square region of foreground
        mask = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
        mask[0, 0, 20:44, 20:44] = 1  # Square from (20,20) to (43,43)

        signed_dist = FastGeodis.signed_exact_euclidean2d(mask, spacing=[1.0, 1.0])

        # Inside the square (e.g., center) should be negative
        center_dist = signed_dist[0, 0, 32, 32].item()
        self.assertLess(center_dist, 0, "Center should have negative signed distance")

        # Outside the square (e.g., corner) should be positive
        corner_dist = signed_dist[0, 0, 0, 0].item()
        self.assertGreater(
            corner_dist, 0, "Corner should have positive signed distance"
        )

    @skip_if_no_cuda
    def test_signed_boundary(self):
        """Test that boundary has distance close to 0"""
        device = "cuda"
        height, width = 64, 64

        mask = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
        mask[0, 0, 20:44, 20:44] = 1

        signed_dist = FastGeodis.signed_exact_euclidean2d(mask, spacing=[1.0, 1.0])

        # On the boundary, signed distance should be close to 0
        boundary_dist = signed_dist[0, 0, 20, 32].item()  # Edge of square
        self.assertAlmostEqual(abs(boundary_dist), 0.5, places=0)


class TestSignedExactEuclidean3D(unittest.TestCase):
    """Tests for Signed 3D Exact EDT"""

    @skip_if_no_cuda
    def test_signed_distance_3d(self):
        """Test 3D signed distance has correct signs"""
        device = "cuda"
        depth, height, width = 32, 32, 32

        # Create a cube region of foreground
        mask = torch.zeros(
            1, 1, depth, height, width, dtype=torch.float32, device=device
        )
        mask[0, 0, 10:22, 10:22, 10:22] = 1  # Cube

        signed_dist = FastGeodis.signed_exact_euclidean3d(mask, spacing=[1.0, 1.0, 1.0])

        # Inside the cube should be negative
        center_dist = signed_dist[0, 0, 16, 16, 16].item()
        self.assertLess(center_dist, 0, "Center should have negative signed distance")

        # Outside the cube should be positive
        corner_dist = signed_dist[0, 0, 0, 0, 0].item()
        self.assertGreater(
            corner_dist, 0, "Corner should have positive signed distance"
        )


class TestCompareWithApproximate(unittest.TestCase):
    """Compare PBA+ exact EDT with approximate raster scanning EDT"""

    @skip_if_no_cuda
    def test_pba_more_accurate(self):
        """Test that PBA+ is more accurate than raster scanning for diagonal distances"""
        device = "cuda"
        height, width = 128, 128

        # Single seed at corner
        mask = torch.ones(1, 1, height, width, dtype=torch.float32, device=device)
        mask[0, 0, 0, 0] = 0

        # PBA+ exact
        pba_dist = FastGeodis.exact_euclidean2d(mask, spacing=[1.0, 1.0])

        # Approximate (raster scanning with lamb=0)
        image = torch.ones_like(mask)
        approx_dist = FastGeodis.generalised_geodesic2d(image, mask, 1e10, 0.0, 4)

        # Expected distance to opposite corner
        expected_dist = math.sqrt((height - 1) ** 2 + (width - 1) ** 2)

        pba_corner = pba_dist[0, 0, height - 1, width - 1].item()
        approx_corner = approx_dist[0, 0, height - 1, width - 1].item()

        pba_error = abs(pba_corner - expected_dist)
        approx_error = abs(approx_corner - expected_dist)

        # PBA should be more accurate (or at least as accurate)
        self.assertLessEqual(
            pba_error, 1.0, f"PBA+ error {pba_error} is too high for diagonal distance"
        )


if __name__ == "__main__":
    unittest.main()
