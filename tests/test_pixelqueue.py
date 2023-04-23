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

import math
import unittest

import numpy as np
import torch
from parameterized import parameterized

from .utils import *

# set deterministic seed
torch.manual_seed(15)
np.random.seed(15)


class TestPixelQueue(unittest.TestCase):
    @parameterized.expand(CONF_ALL_CPU)
    @run_cuda_if_available
    def test_ill_shape(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_pixelqueue_func(num_dims=num_dims)

        # batch != 1 - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        with self.assertRaises(ValueError):
            mask_shape_mod[0] = 2
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        with self.assertRaises(ValueError):
            image_shape_mod[0] = 2
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # spatial shape mismatch - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        with self.assertRaises(ValueError):
            image_shape_mod[-1] = 12
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # 3D shape for 2D functions - unsupported
        if num_dims == 2:
            image_shape_mod = image_shape.copy()
            mask_shape_mod = mask_shape.copy()
            with self.assertRaises(ValueError):
                image_shape_mod += [16]
                mask_shape_mod += [16]
                image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
                mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
                geodesic_dist = geodis_func(image, mask, 1.0, 2)

    @parameterized.expand(CONF_ALL_CPU)
    @run_cuda_if_available
    def test_correct_shape(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
        mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

    @parameterized.expand(CONF_ALL_CPU)
    @run_cuda_if_available
    def test_zeros_input(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.zeros(mask_shape, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # output should be zeros as well
        np.testing.assert_allclose(
            np.zeros(mask_shape, dtype=np.float32), geodesic_dist.cpu().numpy()
        )

    @parameterized.expand(CONF_ALL_CPU)
    @run_cuda_if_available
    def test_mask_ones_input(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.ones(mask_shape, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # output should be ones * v
        np.testing.assert_allclose(
            np.ones(mask_shape, dtype=np.float32) * 1e10, geodesic_dist.cpu().numpy()
        )

    @parameterized.expand(CONF_ALL_CPU)
    @run_cuda_if_available
    def test_euclidean_dist_output(self, device, num_dims, base_dim):
        """
        Explanation:

        Taking euclidean distance with x==0 below:
        x-------------
        |            |
        |            |
        |            |
        |            |
        |            |
        |            |
        --------------

        The max distance in euclidean output will be approx equal to
        distance to furthest corner (x->o) along diagonal shown below:
        x-------------
        | \          |
        |   \        |
        |     \      |
        |       \    |
        |         \  |
        |           \|
        -------------o
        """

        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.ones(image_shape, dtype=torch.float32).to(device)
        mask = torch.ones(mask_shape, dtype=torch.float32).to(device)
        mask[0, 0, 0, 0] = 0

        geodesic_dist = geodis_func(image, mask, 0.0, 2)
        pred_max_dist = geodesic_dist.cpu().numpy().max()
        exp_max_dist = math.sqrt(num_dims * (base_dim**2))
        tolerance = 10 if num_dims == 2 else 100  # more tol needed for 3d approx

        check = exp_max_dist - tolerance < pred_max_dist < exp_max_dist + tolerance
        self.assertTrue(check)

    @parameterized.expand(CONF_3D_CPU)
    @run_cuda_if_available
    def test_ill_spacing(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.zeros(mask_shape, dtype=torch.float32).to(device)

        spacing = [1.0, 1.0]
        geodis_func = get_pixelqueue_func(num_dims=num_dims, spacing=spacing)

        with self.assertRaises(ValueError):
            geodesic_dist = geodis_func(image, mask, 1.0, 2)


class TestPixelQueueSigned(unittest.TestCase):
    @parameterized.expand(CONF_ALL_CPU)
    def test_ill_shape(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_signed_pixelqueue_func(num_dims=num_dims)

        # batch != 1 - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        with self.assertRaises(ValueError):
            mask_shape_mod[0] = 2
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        with self.assertRaises(ValueError):
            image_shape_mod[0] = 2
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # spatial shape mismatch - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        with self.assertRaises(ValueError):
            image_shape_mod[-1] = 12
            image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
            mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
            geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # 3D shape for 2D functions - unsupported
        if num_dims == 2:
            image_shape_mod = image_shape.copy()
            mask_shape_mod = mask_shape.copy()
            with self.assertRaises(ValueError):
                image_shape_mod += [16]
                mask_shape_mod += [16]
                image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
                mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)
                geodesic_dist = geodis_func(image, mask, 1.0, 2)

    @parameterized.expand(CONF_ALL_CPU)
    def test_correct_shape(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_signed_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
        mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

    @parameterized.expand(CONF_ALL_CPU)
    def test_zeros_input(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_signed_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.zeros(mask_shape, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # output should be -1 * ones * v
        np.testing.assert_allclose(
            -1 * np.ones(mask_shape, dtype=np.float32) * 1e10,
            geodesic_dist.cpu().numpy(),
        )

    @parameterized.expand(CONF_ALL_CPU)
    def test_mask_ones_input(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_signed_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.ones(mask_shape, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 1.0, 2)

        # output should be ones * v
        np.testing.assert_allclose(
            np.ones(mask_shape, dtype=np.float32) * 1e10, geodesic_dist.cpu().numpy()
        )

    @parameterized.expand(CONF_3D_CPU)
    def test_ill_spacing(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        # device mismatch for input - unsupported
        image = torch.zeros(image_shape, dtype=torch.float32).to(device)
        mask = torch.zeros(mask_shape, dtype=torch.float32).to(device)

        spacing = [1.0, 1.0]
        geodis_func = get_signed_pixelqueue_func(num_dims=num_dims, spacing=spacing)

        with self.assertRaises(ValueError):
            geodesic_dist = geodis_func(image, mask, 1.0, 2)


class TestGSFPixelQueue(unittest.TestCase):
    @parameterized.expand(CONF_ALL_CPU_FM)
    @run_cuda_if_available
    def test_correct_shape(self, device, num_dims, base_dim):
        print(device)
        print(num_dims)

        # start with a good shape for image and mask
        image_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)
        mask_shape = get_simple_shape(base_dim=base_dim, num_dims=num_dims)

        geodis_func = get_GSF_pixelqueue_func(num_dims=num_dims)

        # device mismatch for input - unsupported
        image_shape_mod = image_shape.copy()
        mask_shape_mod = mask_shape.copy()
        image = torch.rand(image_shape_mod, dtype=torch.float32).to(device)
        mask = torch.rand(mask_shape_mod, dtype=torch.float32).to(device)

        # should work without any errors
        geodesic_dist = geodis_func(image, mask, 0.0, 1.0, 2)


if __name__ == "__main__":
    unittest.main()