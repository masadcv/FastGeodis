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
