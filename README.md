# FastGeodis: Fast Generalised Geodesic Distance Transform
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Downloads](https://static.pepy.tech/personalized-badge/fastgeodis?period=total&units=international_system&left_color=grey&right_color=green&left_text=Total%20Downloads)](https://pepy.tech/project/fastgeodis)
[![status](https://joss.theoj.org/papers/d0b6e3daa4b22fec471691c6f1c60e2a/status.svg)](https://joss.theoj.org/papers/d0b6e3daa4b22fec471691c6f1c60e2a)
[![CI Build](https://github.com/masadcv/FastGeodis/actions/workflows/build.yml/badge.svg)](https://github.com/masadcv/FastGeodis/actions/workflows/build.yml)
[![PyPI version](https://badge.fury.io/py/FastGeodis.svg)](https://badge.fury.io/py/FastGeodis)
<img src="https://img.shields.io/badge/Python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10%20-3776ab.svg"/>
<img src="https://img.shields.io/badge/PyTorch-%3E%3D%201.5.0-brightgreen.svg"/>
<!--<img src="https://img.shields.io/pypi/dm/fastgeodis.svg?label=PyPI%20downloads&logo=python&logoColor=green"/>-->
This repository provides CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both. 

| 2D images, 1 of 4 passes | 3D volumes, 1 of 6 passes  |
|-------------------|-------------------------|
| <img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/FastGeodis2D.png?raw=true" width="300" /> | <img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/FastGeodis3D.png?raw=true" width="300" /> |


The above raster scan method can be parallelised for each row/plane on an available device (CPU or GPU). This leads to significant speed up as compared to existing non-parallelised raster scan implementations (e.g. [https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)). Python interface is provided (using PyTorch) for enabling its use in deep learning and image processing pipelines.

In addition, implementation of generalised version of Geodesic distance transforms along with Geodesic Symmetric Filtering (GSF) is provided for use in interactive segmentation methods, that were originally proposed in [1, 2, 5].

> The raster scan based implementation provides a balance towards speed rather than accuracy of Geodesic distance transform and hence results in efficient hardware utilisation. On the other hand, in case of Euclidean distance transform, exact results can be achieved with other packages (albeit not on necessarilly on GPU) [6, 7, 8]

# Citation
If you use this code in your research, then please consider citing:

[![status](https://joss.theoj.org/papers/d0b6e3daa4b22fec471691c6f1c60e2a/status.svg)](https://joss.theoj.org/papers/d0b6e3daa4b22fec471691c6f1c60e2a)

***Asad, Muhammad, Reuben Dorent, and Tom Vercauteren. "FastGeodis: Fast Generalised Geodesic Distance Transform." Journal of Open Source Software (JOSS), 2022.*** ([paper link](https://doi.org/10.21105/joss.04532))

Bibtex:
```
@article{asad2022fastgeodis, 
  doi = {10.21105/joss.04532}, 
  url = {https://doi.org/10.21105/joss.04532}, 
  year = {2022}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {79}, 
  pages = {4532}, 
  author = {Muhammad Asad and Reuben Dorent and Tom Vercauteren}, 
  title = {FastGeodis: Fast Generalised Geodesic Distance Transform}, 
  journal = {Journal of Open Source Software} 
}
```

# Installation instructions
The provided package can be installed using:

`pip install FastGeodis`

or

`pip install git+https://github.com/masadcv/FastGeodis`

or (on conda environments with existing installation of PyTorch with CUDA)

`pip install FastGeodis --no-build-isolation`

# Included methods
## Optimised Fast Implementations for GPU/CPU based on [1]
| Method | Description | Documentation |
|--------|-------------|---------------|
| Fast Generalised Geodesic Distance 2D   |  Paralellised generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.generalised_geodesic2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic2d)         |
| Fast Generalised Geodesic Distance 3D   |  Paralellised generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.generalised_geodesic3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic3d)         |
| Fast Signed Generalised Geodesic Distance 2D   |  Paralellised signed generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.signed_generalised_geodesic2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic2d)         |
| Fast Signed Generalised Geodesic Distance 3D   |  Paralellised signed generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.signed_generalised_geodesic3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic3d)         |
| Fast Geodesic Symmetric Filtering 2D   |  Paralellised geodesic symmetric filtering for CPU/GPU [2]          |      [FastGeodis.GSF2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF2d)         |
| Fast Geodesic Symmetric Filtering 3D   |  Paralellised geodesic symmetric filtering for CPU/GPU [2]          |      [FastGeodis.GSF3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF3d)         |
----

## Toivanen's Implementations for CPU based on [9]

| Method | Description | Documentation |
|--------|-------------|---------------|
| Toivanen's Generalised Geodesic Distance 2D   |  Toivanen's generalised geodesic distance transform for CPU [9]          |      [FastGeodis.generalised_geodesic2d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic2d_toivanen)         |
| Toivanen's Generalised Geodesic Distance 3D   |  Toivanen's generalised geodesic distance transform for CPU [9]          |      [FastGeodis.generalised_geodesic3d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic3d_toivanen)         |
| Toivanen's Signed Generalised Geodesic Distance 2D   |  Toivanen's signed generalised geodesic distance transform for CPU [9]          |      [FastGeodis.signed_generalised_geodesic2d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic2d_toivanen)         |
| Toivanen's Signed Generalised Geodesic Distance 3D   |  Toivanen's signed generalised geodesic distance transform for CPU [9]          |      [FastGeodis.signed_generalised_geodesic3d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic3d_toivanen)         |
| Toivanen's Geodesic Symmetric Filtering 2D   |  Toivanen's geodesic symmetric filtering for CPU [2, 9]          |      [FastGeodis.GSF2d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF2d_toivanen)         |
| Toivanen's Geodesic Symmetric Filtering 3D   |  Toivanen's geodesic symmetric filtering for CPU [2, 9]          |      [FastGeodis.GSF3d_toivanen](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF3d_toivanen)         |

## Pixel Queue Implementations for CPU based on [11]

| Method | Description | Documentation |
|--------|-------------|---------------|
| Pixel Queue Geodesic Distance 2D   |  Pixel Queue geodesic distance transform for CPU [11]          |      [FastGeodis.geodesic2d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.geodesic2d_pixelqueue)         |
| Pixel Queue Geodesic Distance 3D   |  Pixel Queue geodesic distance transform for CPU [11]          |      [FastGeodis.geodesic3d_pixelqueue](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.geodesic3d_pixelqueue)         |
| Pixel Queue Signed Geodesic Distance 2D   |  Pixel Queue signed geodesic distance transform for CPU [11]          |      [FastGeodis.signed_geodesic2d_pixelqueue](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_geodesic2d_pixelqueue)         |
| Pixel Queue Signed Geodesic Distance 3D   |  Pixel Queue signed geodesic distance transform for CPU [11]          |      [FastGeodis.signed_geodesic3d_pixelqueue](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_geodesic3d_pixelqueue)         |
| Pixel Queue Geodesic Symmetric Filtering 2D   |  Pixel Queue geodesic symmetric filtering for CPU [2, 11]          |      [FastGeodis.GSF2d_pixelqueue](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF2d_pixelqueue)         |
| Pixel Queue Geodesic Symmetric Filtering 3D   |  Pixel Queue geodesic symmetric filtering for CPU [2, 11]          |      [FastGeodis.GSF3d_pixelqueue](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF3d_pixelqueue)         |

## Fast Marching Implementations for CPU based on [4, 10]

| Method | Description | Documentation |
|--------|-------------|---------------|
| Fast Marching Geodesic Distance 2D   |  Fast Marching geodesic distance transform for CPU [9]          |      [FastGeodis.geodesic2d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.geodesic2d_fastmarch)         |
| Fast Marching Geodesic Distance 3D   |  Fast Marching geodesic distance transform for CPU [9]          |      [FastGeodis.geodesic3d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.geodesic3d_fastmarch)         |
| Fast Marching Signed Geodesic Distance 2D   |  Fast Marching signed geodesic distance transform for CPU [9]          |      [FastGeodis.signed_geodesic2d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_geodesic2d_fastmarch)         |
| Fast Marching Signed Geodesic Distance 3D   |  Fast Marching signed geodesic distance transform for CPU [9]          |      [FastGeodis.signed_geodesic3d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_geodesic3d_fastmarch)         |
| Fast Marching Geodesic Symmetric Filtering 2D   |  Fast Marching geodesic symmetric filtering for CPU [2, 9]          |      [FastGeodis.GSF2d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF2d_fastmarch)         |
| Fast Marching Geodesic Symmetric Filtering 3D   |  Fast Marching geodesic symmetric filtering for CPU [2, 9]          |      [FastGeodis.GSF3d_fastmarch](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF3d_fastmarch)         |

# Example usage

### Fast Geodesic Distance Transform
The following demonstrates a simple example showing FastGeodis usage:

To compute Geodesic Distance Transform:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
image = np.asarray(Image.open("data/img2d.png"), np.float32)

image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
image_pt = image_pt.to(device)
mask_pt = torch.ones_like(image_pt)
mask_pt[..., 100, 100] = 0

v = 1e10
# lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)
lamb = 1.0
iterations = 2
geodesic_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, iterations
)
geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())
```

To compute Euclidean Distance Transform:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
image = np.asarray(Image.open("data/img2d.png"), np.float32)

image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
image_pt = image_pt.to(device)
mask_pt = torch.ones_like(image_pt)
mask_pt[..., 100, 100] = 0

v = 1e10
# lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)
lamb = 0.0
iterations = 2
euclidean_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, iterations
)
euclidean_dist = np.squeeze(euclidean_dist.cpu().numpy())
```

For more usage examples see:
| Description  |  Python |  Colab link  |
|--------------|---------|--------------|
| **Simple 2D Geodesic and Euclidean Distance** | [`samples/simpledemo2d.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo2d.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo2d.ipynb)  |
| **Simple Signed 2D Geodesic and Euclidean Distance** | [`samples/simpledemo2d_signed.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo2d_signed.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo2d_signed.ipynb) |
| **Simple 3D Geodesic and Euclidean Distance** | [`samples/simpledemo3d.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo3d.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo3d.ipynb)  |
| **Simple Signed 3D Geodesic and Euclidean Distance** | [`samples/simpledemo3d_signed.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo3d_signed.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo3d_signed.ipynb)  |
| **2D Geodesic Distance** | [`samples/demo2d.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/demo2d.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo2d.ipynb)  |
| **2D Signed Geodesic Distance** | [`samples/demo2d_signed.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/demo2d_signed.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo2d_signed.ipynb)  |
| **3D Geodesic Distance** | [`samples/demo3d.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/demo3d.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo3d.ipynb)  |
| **3D Signed Geodesic Distance** | [`samples/demo3d_signed.py`](https://github.com/masadcv/FastGeodis/blob/master/samples/demo3d_signed.py) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo3d_signed.ipynb)  |
| **2D GSF Segmentation Smoothing** |  [`samples/demoGSF2d_SmoothingSegExample.ipynb`](https://github.com/masadcv/FastGeodis/blob/master/samples/demoGSF2d_SmoothingSegExample.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demoGSF2d_SmoothingSegExample.ipynb) | 

# Unit Tests
A number of unittests are provided, which can be run as:

```
pip install -r requirements-dev.txt
python -m unittest
```

# Documentation
Further details of each function implemented in FastGeodis can be accessed at the documentation hosted at: [https://fastgeodis.readthedocs.io](https://fastgeodis.readthedocs.io).

# Contributing to FastGeodis
Spotted a bug or have a feature request to improve the package? We would love to have your input! See our [guidelines for contributing](https://fastgeodis.readthedocs.io/en/latest/contributing.html).


# Comparison of Execution Time and Accuracy
FastGeodis (CPU/GPU) is compared with existing GeodisTK ([https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)) in terms of execution speed as well as accuracy.


### Execution Time
| 2D images | 3D volumes  |
|-------------------|-------------------------|
|<img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/experiment_2d.png?raw=true" width="400" />|<img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/experiment_3d.png?raw=true" width="400" />|

<br>

### Accuracy
2D case
| Qualitative Comparison | Quantitative (joint histogram)  |
|-------------------|-------------------------|
|<img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/fast_marching_compare_2d.png?raw=true?raw=true" width="800" />  |<img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/fast_marching_compare_2d_jointhist.png?raw=true" width="400" /> |

3D case

| Qualitative Comparison | Quantitative (joint histogram)  |
|-------------------|-------------------------|
| <img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/fast_marching_compare_3d.png?raw=true" width="800" /> | <img src="https://raw.githubusercontent.com/masadcv/FastGeodis/master/figures/fast_marching_compare_3d_jointhist.png?raw=true" width="400" /> |

## References
- [1] Criminisi, Antonio, Toby Sharp, and Khan Siddiqui. "Interactive Geodesic Segmentation of n-Dimensional Medical Images on the Graphics Processor." Radiological Society of North America (RSNA), 2009. [[pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Criminisi_RSNA09_extendedAbstract1.pdf), [ppt](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/RSNA2009_Segmentation_Criminisi.ppt), [url](https://www.microsoft.com/en-us/research/publication/interactive-geodesic-segmentation-of-n-dimensional-medical-images-on-the-graphics-processor/)]

- [2] Criminisi, Antonio, Toby Sharp, and Andrew Blake. "Geos: Geodesic image segmentation." European Conference on Computer Vision. Springer, Berlin, Heidelberg, 2008. [[doi](https://doi.org/10.1007/978-3-540-88682-2_9)]

- [3] Weber, Ofir, et al. "Parallel algorithms for approximation of distance maps on parametric surfaces." ACM Transactions on Graphics (TOG), (2008). [[doi](https://doi.org/10.1145/1409625.1409626)]

- [4] GeodisTK: [https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)

- [5] Criminisi, Antonio, Toby Sharp, Carsten Rother, and Patrick Pérez. "Geodesic image and video editing." ACM Trans. Graph. 29, no. 5 (2010): 134-1. [[doi](https://doi.org/10.1145/1857907.1857910)]

- [6] [https://github.com/seung-lab/euclidean-distance-transform-3d](https://github.com/seung-lab/euclidean-distance-transform-3d)

- [7] [https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)

- [8] [https://www.tensorflow.org/addons/api_docs/python/tfa/image/euclidean_dist_transform](https://www.tensorflow.org/addons/api_docs/python/tfa/image/euclidean_dist_transform)

- [9] Toivanen, Pekka J. "New geodosic distance transforms for gray-scale images." Pattern Recognition Letters 17.5 (1996): 437-450.

- [10] Sethian, James A. "Fast marching methods." SIAM review 41.2 (1999): 199-235.

- [11] Ikonen, L., & Toivanen, P. (2007). Distance and nearest neighbor transforms on gray-level surfaces. Pattern Recognition Letters, 28(5), 604-612. [[doi](https://doi.org/10.1016/j.patrec.2006.10.010)]
