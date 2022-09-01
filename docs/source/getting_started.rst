*****************
Getting Started
*****************

About
########
**FastGeodis** provides efficient CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both.
See :doc:`methodology` section for more details of the implemented algorithm.

1. Criminisi, Antonio, Toby Sharp, and Khan Siddiqui. "Interactive Geodesic Segmentation of n-Dimensional Medical Images on the Graphics Processor." Radiological Society of North America (RSNA), 2009.


Python Version Support
###################################
:code:`Python 3.6`, :code:`Python 3.7`, :code:`Python 3.8`, :code:`Python 3.9`


Installation
###################################

**FastGeodis** can be installed via pip by running the following from a terminal window:
::

    pip install FastGeodis

or 

::

    pip install git+https://github.com/masadcv/FastGeodis

or (on conda environment with existing installation of PyTorch with CUDA)
::
    
    pip install FastGeodis --no-build-isolation


Dependencies
###################################

+------------+------------------+
| Dependency | Minimum Version  |
+============+==================+
|torch       | 1.5.0            |
+------------+------------------+

In addition, for compilation and execution on GPU, the **FastGeodis** package requires a CUDA installation compatible with installed PyTorch version. 

Optional Development Dependencies
###################################
+-------------+------------------+
| Dependency  | Minimum Version  |
+=============+==================+
|numpy        | 1.19.2           |
+-------------+------------------+
|matplotlib   | 3.2.0            |
+-------------+------------------+
|parameterized| 0.7.0            |
+-------------+------------------+
|SimpleITK    | 2.0.0            |
+-------------+------------------+


Example Usage
###################################
The following demonstrates a simple example showing FastGeodis usage:

To compute Geodesic Distance Transform:
::

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

To compute Euclidean Distance Transform:
::

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

For more comprehensive usages examples, see: :doc:`usage_examples`
