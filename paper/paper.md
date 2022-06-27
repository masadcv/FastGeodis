---
title: 'FastGeodis: Fast Generalised Geodesic Distance Transform'
tags:
  - Python
  - PyTorch
  - Deep Learning
  - Medical Imaging
  - Distance Transform
authors:
  - name: Muhammad Asad
    orcid: 0000-0002-3672-2414
    affiliation: 1
    corresponding: true
  - name: Reuben Dorent
    # orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Tom Vercauteren
    orcid: 0000-0003-1794-0456
    affiliation: 1
affiliations:
 - name: School of Biomedical Engineering & Imaging Sciences, King’s College London, UK
   index: 1
date: 23 June 2022
bibliography: paper.bib
---

# Summary 

  
Geodesic and Euclidean distance transforms have been widely used in a number of applications, where distance from a set of reference points is computed. Methods from recent years have shown effectiveness in applying Geodesic distance transform to interactively segment 3D medical imaging data [@wang2018deepigeos; @criminisi2008geos]. Despite existing methods for efficient computation of Geodesic distance transform on GPU and CPU devices [@criminisiinteractive; @criminisi2008geos; @weber2008parallel; @toivanen1996new], an open-source implementation of such methods do not exist. 
On the contrary, efficient methods for Euclidean distance transform [@felzenszwalb2012distance] have open-source implementations [@tensorflow2015-whitepaper; @eucildeantdimpl]. Existing libraries, e.g. [@geodistk], provide C++ implementations of Geodesic distance transform, however they lack efficient utilization of underlying hardware and hence results in significant computation time especially when applying them on 3D medical imaging volumes.  

The `FastGeodis` package provides an efficient implementation for computing Geodesic and Euclidean distance transforms (or a mixture of both) targeting efficient utilizing of CPU and GPU hardwares. This package is able to handle 2D as well as 3D data where it achieves up to 15x speedup on CPU and up to 60x speedup on GPU as compared to existing open-source libraries [@geodistk]. 

  

# Statement of need 

 
Despite existing open-source implementation of distance transforms [@tensorflow2015-whitepaper; @eucildeantdimpl; @geodistk], open-source implementations of efficient Geodesic distance transform algorithms [@criminisiinteractive; @weber2008parallel] on CPU and GPU do not exist. However, for Euclidean distance efficient CPU [@eucildeantdimpl] and GPU [@tensorflow2015-whitepaper] implementations exist. To the best of our knowledge, `FastGeodis` is the first open-source implementation of efficient Geodesic distance transform [@criminisiinteractive], achieving up to 15x speedup on CPU and up to 60x speedup on GPU as compared to existing open-source libraries [@geodistk]. It also provides efficient implementation of Euclidean distance transform. In addition, it is the first open-source implementation of generalized Geodesic distance transform and Geodesic Symmetric Filtering (GSF) proposed in [@criminisi2008geos]. 

  

The ability to efficiently compute Geodesic and Euclidean distance transforms can significantly enhance distance transform applications especially for training deep learning models that utilize distance transforms [@wang2018deepigeos]. It will improve prototyping, experimentation, and deployment of such methods, where efficient computation of distance transforms has been a limiting factor. In 3D medical imaging problems, efficient computation of distance transforms will lead to significant speed ups, enabling online learning applications for better processing/labelling/inference from volumetric datasets [@asad2022econet].  In addition, `FastGeodis` provides efficient implementation for both CPUs and GPUs hardware and hence will enable efficient use of a wide range of hardware devices. 

  

# Methodology 

  

`FastGeodis` implements an efficient distance transform algorithm from [@criminisiinteractive], which provides parallelizable raster scan kernels to compute distance transform. The implementation consists of data passes parallelized across a line (2D) or plane (3D). \autoref{fig:hwpasses} shows these data passes. For 2D data, four data passes are required, top-bottom, bottom-top, left-right and right-left, whereas for 3D data 6 passes are required, front-back, back-front, top-bottom, bottom-top, left-right and right-left. The algorithm can be applied to efficient compute both Geodesic and Euclidean distance transforms.

  

`FastGeodis` package is implemented using `PyTorch` [@NEURIPS2019_9015] utilizing OpenMP for CPU and CUDA for GPU parallelization of the algorithm. It is accessible as a python package, that can be installed across different operating systems and devices. Several examples are provided for understanding the usage of the library on 2D and 3D data using CPU or GPU.

  

![Raster scan passes in FastGeodis.\label{fig:hwpasses}](FastGeodis.png){ width=100% } 

# Acknowledgements

This research was supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101016131. 

# References