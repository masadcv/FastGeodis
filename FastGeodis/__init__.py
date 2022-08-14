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
import FastGeodisCpp


def generalised_geodesic2d(
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float, 
    iter: int = 2
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
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float, 
    iter: int = 2
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
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float, 
    iter: int = 2
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
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float, 
    iter: int = 2
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

def generalised_geodesic2d_fastmarch(
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float
):
    r"""Computes Generalised Geodesic Distance using Fast Marching method from:

    Sethian, James A. 
    "Fast marching methods." 
    SIAM review 41.2 (1999): 199-235.
    
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

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic2d_fastmarch(
        image, softmask, v, lamb, 1 - lamb
    )


def generalised_geodesic3d_fastmarch(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float
):
    r"""Computes Generalised Geodesic Distance using Fast Marching method from:

    TSethian, James A. 
    "Fast marching methods." 
    SIAM review 41.2 (1999): 199-235.
    
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

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.generalised_geodesic3d_fastmarch(
        image, softmask, spacing, v, lamb, 1 - lamb
    )

def signed_generalised_geodesic2d_fastmarch(
    image: torch.Tensor, 
    softmask: torch.Tensor, 
    v: float, 
    lamb: float
):
    r"""Computes Signed Generalised Geodesic Distance using Fast Marching method from:

    Sethian, James A. 
    "Fast marching methods." 
    SIAM review 41.2 (1999): 199-235.
    
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

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.signed_generalised_geodesic2d_fastmarch(
        image, softmask, v, lamb, 1 - lamb
    )


def signed_generalised_geodesic3d_fastmarch(
    image: torch.Tensor,
    softmask: torch.Tensor,
    spacing: List,
    v: float,
    lamb: float
):
    r"""Computes Signed Generalised Geodesic Distance using Fast Marching method from:

    Sethian, James A. 
    "Fast marching methods." 
    SIAM review 41.2 (1999): 199-235.
    
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
    return FastGeodisCpp.signed_generalised_geodesic3d_fastmarch(
        image, softmask, spacing, v, lamb, 1 - lamb
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

def GSF2d_fastmarch(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    v: float,
    lamb: float
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
        softmask: softmask in range [0, 1] with seed information.
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF2d_fastmarch(image, softmask, theta, v, lamb)


def GSF3d_fastmarch(
    image: torch.Tensor,
    softmask: torch.Tensor,
    theta: float,
    spacing: List,
    v: float,
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
        softmask: softmask in range [0, 1] with seed information.
        spacing: spacing for 3D data
        v: weighting factor for establishing relationship between unary and spatial distances.
        lamb: weighting factor between 0.0 and 1.0. 0.0 returns euclidean distance, whereas 1.0 returns geodesic distance

    Returns:
        torch.Tensor with distance transform
    """
    return FastGeodisCpp.GSF3d_fastmarch(image, softmask, theta, spacing, v, lamb)
