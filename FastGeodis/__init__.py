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

from typing import List
import torch
from ._version import __version__
import FastGeodisCpp


def generalised_geodesic2d(
    image: torch.Tensor, softmask: torch.Tensor, v: float, lamb: float, iter: int = 2
):
    r"""Computes Generalised Geodesic Distance using FastGeodis raster scanning.
    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic2d(
        image, softmask, v, lamb, 1 - lamb, iter
    )


def generalised_geodesic3d(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float,
    iter: int = 4,
):
    r"""Computes Generalised Geodesic Distance using FastGeodis raster scanning.
    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic3d(
        image, softmask, spacing, v, lamb, 1 - lamb, iter
    )


def signed_generalised_geodesic2d(
    image: torch.Tensor, softmask: torch.Tensor, v: float, lamb: float, iter: int = 2
):
    r"""Computes Signed Generalised Geodesic Distance using FastGeodis raster scanning.
    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_generalised_geodesic2d(
        image, softmask, v, lamb, 1 - lamb, iter
    )


def signed_generalised_geodesic3d(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float,
    iter: int = 4,
):
    r"""Computes Signed Generalised Geodesic Distance using FastGeodis raster scanning.
    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_generalised_geodesic3d(
        image, softmask, spacing, v, lamb, 1 - lamb, iter
    )


def generalised_geodesic2d_toivanen(
    image: torch.Tensor, softmask: torch.Tensor, v: float, lamb: float, iter: int = 2
):
    r"""Computes Generalised Geodesic Distance using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic2d_toivanen(
        image, softmask, v, lamb, 1 - lamb, iter
    )


def generalised_geodesic3d_toivanen(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float,
    iter: int = 4,
):
    r"""Computes Generalised Geodesic Distance using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on generalised geodesic distance, check the following reference:


    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic3d_toivanen(
        image, softmask, spacing, v, lamb, 1 - lamb, iter
    )


def signed_generalised_geodesic2d_toivanen(
    image: torch.Tensor, softmask: torch.Tensor, v: float, lamb: float, iter: int = 2
):
    r"""Computes Signed Generalised Geodesic Distance using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_generalised_geodesic2d_toivanen(
        image, softmask, v, lamb, 1 - lamb, iter
    )


def signed_generalised_geodesic3d_toivanen(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float,
    iter: int = 4,
):
    r"""Computes Signed Generalised Geodesic Distance using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on generalised geodesic distance, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_generalised_geodesic3d_toivanen(
        image, softmask, spacing, v, lamb, 1 - lamb, iter
    )


def geodesic2d_pixelqueue(image: torch.Tensor, seed: torch.Tensor, lamb: float):
    r"""Computes Geodesic Distance using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.geodesic2d_pixelqueue(image, seed, lamb, 1 - lamb)


def geodesic3d_pixelqueue(
    image: torch.Tensor, seed: torch.Tensor, spacing: List, lamb: float
):
    r"""Computes Geodesic Distance using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.geodesic3d_pixelqueue(image, seed, spacing, lamb, 1 - lamb)


def signed_geodesic2d_pixelqueue(image: torch.Tensor, seed: torch.Tensor, lamb: float):
    r"""Computes Signed Generalised Geodesic Distance using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_geodesic2d_pixelqueue(image, seed, lamb, 1 - lamb)


def signed_geodesic3d_pixelqueue(
    image: torch.Tensor, seed: torch.Tensor, spacing: List, lamb: float
):
    r"""Computes Signed Geodesic Distance using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_geodesic3d_pixelqueue(
        image, seed, spacing, lamb, 1 - lamb
    )


def geodesic2d_fastmarch(image: torch.Tensor, seed: torch.Tensor, lamb: float):
    r"""Computes Geodesic Distance using Fast Marching method from:

    Sethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.geodesic2d_fastmarch(image, seed, lamb, 1 - lamb)


def geodesic3d_fastmarch(
    image: torch.Tensor, seed: torch.Tensor, spacing: List, lamb: float
):
    r"""Computes Geodesic Distance using Fast Marching method from:

    TSethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.geodesic3d_fastmarch(image, seed, spacing, lamb, 1 - lamb)


def signed_geodesic2d_fastmarch(image: torch.Tensor, seed: torch.Tensor, lamb: float):
    r"""Computes Signed Geodesic Distance using Fast Marching method from:

    Sethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_geodesic2d_fastmarch(image, seed, lamb, 1 - lamb)


def signed_geodesic3d_fastmarch(
    image: torch.Tensor, seed: torch.Tensor, spacing: List, lamb: float
):
    r"""Computes Signed Geodesic Distance using Fast Marching method from:

    Sethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_geodesic3d_fastmarch(
        image, seed, spacing, lamb, 1 - lamb
    )


def GSF2d(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    v: float,
    lamb: float,
    iter: int,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using FastGeodis raster scanning.
    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF2d(image, softmask, theta, v, lamb, iter)


def GSF3d(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    spacing: List,
    v: float,
    lamb: float,
    iter: int,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using FastGeodis raster scanning.
    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF3d(image, softmask, theta, spacing, v, lamb, iter)


def GSF2d_toivanen(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    v: float,
    lamb: float,
    iter: int,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF2d_toivanen(image, softmask, theta, v, lamb, iter)


def GSF3d_toivanen(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    spacing: List,
    v: float,
    lamb: float,
    iter: int,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Toivanen's raster scanning method from:

    Toivanen, Pekka J.
    "New geodosic distance transforms for gray-scale images."
    Pattern Recognition Letters 17.5 (1996): 437-450.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance
        iter: number of passes of the iterative distance transform method

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF3d_toivanen(image, softmask, theta, spacing, v, lamb, iter)


def GSF2d_pixelqueue(
    image: torch.Tensor, seed: torch.Tensor, theta: float, lamb: float
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF2d_pixelqueue(image, seed, theta, lamb)


def GSF3d_pixelqueue(
    image: torch.Tensor,
    seed: torch.Tensor,
    theta: float,
    spacing: List,
    lamb: float,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Pixel Queue method from:

    Ikonen, L., & Toivanen, P. (2007).
    "Distance and nearest neighbor transforms on gray-level surfaces."
    Pattern Recognition Letters, 28(5), 604-612.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in range {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF3d_pixelqueue(image, seed, theta, spacing, lamb)


def GSF2d_fastmarch(image: torch.Tensor, seed: torch.Tensor, theta: float, lamb: float):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Fast Marching method from:

    Sethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF2d_fastmarch(image, seed, theta, lamb)


def GSF3d_fastmarch(
    image: torch.Tensor,
    seed: torch.Tensor,
    theta: float,
    spacing: List,
    lamb: float,
):
    r"""Computes Geodesic Symmetric Filtering (GSF) using Fast Marching method from:

    Sethian, James A.
    "Fast marching methods."
    SIAM review 41.2 (1999): 199-235.

    For more details on GSF, check the following reference:

    Criminisi, Antonio, Toby Sharp, and Andrew Blake.
    "Geos: Geodesic image segmentation."
    European Conference on Computer Vision, Berlin, Heidelberg, 2008.

    The function expects input as torch.Tensor, which can be run on CPU only using Tensor's device location

    Args:
        image: input image, can be grayscale or multiple channels.
        seed: seed in {0, 1} with seed information.
        spacing: spacing for 3D data
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF3d_fastmarch(image, seed, theta, spacing, lamb)


def exact_euclidean2d(mask: torch.Tensor, spacing: List = [1.0, 1.0]):
    r"""Computes Exact Euclidean Distance Transform using the PBA+ (Parallel Banding Algorithm Plus)
    algorithm from:

    Cao, Thanh-Tung, Ke Tang, Anis Mohamed, and Tiow-Seng Tan.
    "Parallel banding algorithm to compute exact distance transform with the GPU."
    In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive 3D Graphics and Games, pp. 83-90. 2010.

    This function computes the EXACT Euclidean distance transform, unlike the approximate methods
    provided by generalised_geodesic2d with lamb=0.0.

    The function expects input as torch.Tensor on CUDA device.

    Args:
        mask: binary mask where 0 indicates seed points (distance=0) and 1 indicates background.
              Should be a 4D tensor with shape (B, C, H, W). Supports arbitrary batch and channel sizes.
        spacing: pixel spacing [spacing_height, spacing_width] to match [H, W] tensor convention.
                 Default is [1.0, 1.0].

    Returns:
        torch.Tensor with exact Euclidean distance transform

    Note:
        - GPU only: Requires CUDA. No CPU fallback is available.
        - Memory: Uses ~21 bytes per pixel (input + int32 Voronoi buffer + float32 output).

    Example:
        >>> import torch
        >>> import FastGeodis
        >>> mask = torch.ones(1, 1, 512, 512, device='cuda')
        >>> mask[0, 0, 256, 256] = 0  # Single seed point
        >>> distance = FastGeodis.exact_euclidean2d(mask, spacing=[1.0, 1.0])
    """
    return FastGeodisCpp.exact_euclidean2d(mask, spacing)


def exact_euclidean3d(mask: torch.Tensor, spacing: List = [1.0, 1.0, 1.0]):
    r"""Computes Exact Euclidean Distance Transform using the PBA+ (Parallel Banding Algorithm Plus)
    algorithm from:

    Cao, Thanh-Tung, Ke Tang, Anis Mohamed, and Tiow-Seng Tan.
    "Parallel banding algorithm to compute exact distance transform with the GPU."
    In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive 3D Graphics and Games, pp. 83-90. 2010.

    This function computes the EXACT Euclidean distance transform for 3D volumetric data, unlike
    the approximate methods provided by generalised_geodesic3d with lamb=0.0.

    The function expects input as torch.Tensor on CUDA device.

    Args:
        mask: binary mask where 0 indicates seed points (distance=0) and 1 indicates background.
              Should be a 5D tensor with shape (B, C, D, H, W). Supports arbitrary batch and channel sizes.
        spacing: voxel spacing [spacing_depth, spacing_height, spacing_width] to match [D, H, W]
                 tensor convention. Default is [1.0, 1.0, 1.0].

    Returns:
        torch.Tensor with exact Euclidean distance transform

    Note:
        - GPU only: Requires CUDA. No CPU fallback is available.
        - Memory: Uses ~29 bytes per voxel (input + int32 Voronoi buffer + float32 output).

    Example:
        >>> import torch
        >>> import FastGeodis
        >>> mask = torch.ones(1, 1, 128, 128, 128, device='cuda')
        >>> mask[0, 0, 64, 64, 64] = 0  # Single seed point
        >>> distance = FastGeodis.exact_euclidean3d(mask, spacing=[1.0, 1.0, 1.0])
    """
    return FastGeodisCpp.exact_euclidean3d(mask, spacing)


def signed_exact_euclidean2d(mask: torch.Tensor, spacing: List = [1.0, 1.0]):
    r"""Computes Signed Exact Euclidean Distance Transform using the PBA+ algorithm.

    This function computes the signed distance where:
    - Negative values: distance inside the foreground region (mask=1) to nearest boundary
    - Positive values: distance outside the foreground region (mask=0) to nearest boundary

    This follows the convention where distance is negative inside the object, matching
    common signed distance field (SDF) conventions.

    The function expects input as torch.Tensor on CUDA device.

    Args:
        mask: binary mask where 0 indicates background and 1 indicates foreground.
              Should be a 4D tensor with shape (B, C, H, W). Supports arbitrary batch and channel sizes.
        spacing: pixel spacing [spacing_height, spacing_width] to match [H, W] tensor convention.
                 Default is [1.0, 1.0].

    Returns:
        torch.Tensor with signed exact Euclidean distance transform

    Note:
        - GPU only: Requires CUDA. No CPU fallback is available.
        - Memory: Uses ~42 bytes per pixel (2x the unsigned version for inside/outside computation).

    Example:
        >>> import torch
        >>> import FastGeodis
        >>> mask = torch.zeros(1, 1, 512, 512, device='cuda')
        >>> mask[0, 0, 200:300, 200:300] = 1  # Square region
        >>> signed_distance = FastGeodis.signed_exact_euclidean2d(mask, spacing=[1.0, 1.0])
        >>> # signed_distance is negative inside the square, positive outside
    """
    return FastGeodisCpp.signed_exact_euclidean2d(mask, spacing)


def signed_exact_euclidean3d(mask: torch.Tensor, spacing: List = [1.0, 1.0, 1.0]):
    r"""Computes Signed Exact Euclidean Distance Transform for 3D volumetric data using the PBA+ algorithm.

    This function computes the signed distance where:
    - Negative values: distance inside the foreground region (mask=1) to nearest boundary
    - Positive values: distance outside the foreground region (mask=0) to nearest boundary

    This follows the convention where distance is negative inside the object, matching
    common signed distance field (SDF) conventions.

    The function expects input as torch.Tensor on CUDA device.

    Args:
        mask: binary mask where 0 indicates background and 1 indicates foreground.
              Should be a 5D tensor with shape (B, C, D, H, W). Supports arbitrary batch and channel sizes.
        spacing: voxel spacing [spacing_depth, spacing_height, spacing_width] to match [D, H, W]
                 tensor convention. Default is [1.0, 1.0, 1.0].

    Returns:
        torch.Tensor with signed exact Euclidean distance transform

    Note:
        - GPU only: Requires CUDA. No CPU fallback is available.
        - Memory: Uses ~58 bytes per voxel (2x the unsigned version for inside/outside computation).

    Example:
        >>> import torch
        >>> import FastGeodis
        >>> mask = torch.zeros(1, 1, 128, 128, 128, device='cuda')
        >>> mask[0, 0, 40:80, 40:80, 40:80] = 1  # Cube region
        >>> signed_distance = FastGeodis.signed_exact_euclidean3d(mask, spacing=[1.0, 1.0, 1.0])
        >>> # signed_distance is negative inside the cube, positive outside
    """
    return FastGeodisCpp.signed_exact_euclidean3d(mask, spacing)
