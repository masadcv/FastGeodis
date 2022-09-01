.. FastGeodis documentation master file, created by
   sphinx-quickstart on Fri Jun 24 19:31:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FastGeodis's documentation!
======================================

Intro
##############################


**FastGeodis** provides efficient CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both.

.. table:: 
   :align: center

   +--------------------------------------------+--------------------------------------------+
   |   **2D images:** 1 of 4 passes             |   **3D volumes:** 1 of 6 passes            |
   +--------------------------------------------+--------------------------------------------+
   | .. figure:: ../../figures/FastGeodis2D.png | .. figure:: ../../figures/FastGeodis3D.png |
   |   :alt: 2D                                 |   :alt: 3D                                 |
   |                                            |                                            |
   +--------------------------------------------+--------------------------------------------+

The above raster scan method can be parallelised for each row/plane on an available device (CPU or GPU). This leads to significant speed up as compared to existing non-parallelised raster scan implementations (e.g. https://github.com/taigw/GeodisTK). Python interface is provided (using PyTorch) for enabling its use in deep learning and image processing pipelines.

See :doc:`methodology` section for more details of the implemented algorithm.

1. Criminisi, Antonio, Toby Sharp, and Khan Siddiqui. "Interactive Geodesic Segmentation of n-Dimensional Medical Images on the Graphics Processor." Radiological Society of North America (RSNA), 2009.
\


Getting Started
###################################
For information on getting started with using **FastGeodis**, including installation, dependencies, and other similar topics, see :doc:`getting_started` page. 

Table of Contents
#################
Use table of contents below, or on the left panel to explore this documentation.

.. toctree::
   :maxdepth: 2
   
   Introduction <self>

   methodology

   getting_started

   usage_examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference Documentation
   
   api_docs

   license

   acknowledgement
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Involved

   Source Code <https://github.com/masadcv/FastGeodis>
   Contributing <contributing>
   Citing <citing>
