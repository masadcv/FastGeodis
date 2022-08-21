*****************
Usage Examples
*****************

**FastGeodis** package includes a number of examples that demonstrate its usage for 2D and 3D data use-cases. Here we provide details of these examples and how to setup and run them.


Simplest Examples
###################################

For a given 2D image, **FastGeodis** can be used to get Geodesic distance transform as:
::

    device = "cpu"
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

For a given 3D image volume, **FastGeodis** can be used to get Geodesic distance transform as:
::

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_sitk = sitk.ReadImage("data/img3d.nii.gz")
    image = sitk.GetArrayFromImage(image_sitk)
    spacing_raw = image_sitk.GetSpacing()[::-1]
    
    image = np.asarray(image, np.float32)
    mask = np.zeros_like(image, np.float32)
    mask[30, 120, 160] = 1

    image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    mask_pt = torch.from_numpy(1 - mask.astype(np.float32)).unsqueeze_(0).unsqueeze_(0)

    image_pt = image_pt.to(device)
    mask_pt = mask_pt.to(device)

    v = 1e10
    iterations = 2
    # lamb = 0.0 (Euclidean) or 1.0 (Geodesic) or (0.0, 1.0) (mixture)
    lamb = 1.0 
    geodesic_dist = FastGeodis.generalised_geodesic3d(
        image_pt, mask_pt, spacing, v, lamb, iterations
    )

To access complete examples, refer to Simple 2D/3D examples in table below.

Note: the above example execute using CPU with :code:`device = "cpu"`. To change execution device to GPU use :code:`device="cuda"`.

Available Examples
###################################

A number of examples are provided, which can be found in `./samples <https://github.com/masadcv/FastGeodis/tree/master/samples>`_ folder.
The examples are provided as local python (.py) files, local Jupyter (.ipynb) notebooks and cloud Colab (.ipynb) notebooks.

Running Examples Locally
------------------------

To run locally, clone github repository and navigate to `./samples <https://github.com/masadcv/FastGeodis/tree/master/samples>`_ directory as:
::

    git clone https://github.com/masadcv/FastGeodis
    cd FastGeodis/samples

and run an example as:
::

    python demo2d.py


Running Examples in Colab
-------------------------

To setup and run Colab notebooks in the cloud, follow the **Colab** links below.


List of Example Scripts
-----------------------
The table below gives an exhaustive list of all available demo examples. It includes relevant links to Python as well as Colab versions of the same demo. 

+------------------------------------------------+----------------------+----------------------------+
| Description                                    | Python Link          |  Colab Link                |
+================================================+======================+============================+
| Simple 2D Geodesic & Euclidean Distance	 | SimpleDemo2dPy_      | SimpleDemo2dColab_         |
+------------------------------------------------+----------------------+----------------------------+
| Simple Signed 2D Geodesic & Euclidean Distance | SimpleDemo2dSignedPy_| SimpleDemo2dSignedColab_   |
+------------------------------------------------+----------------------+----------------------------+
| Simple 3D Geodesic & Euclidean Distance	 | SimpleDemo3dPy_      | SimpleDemo3dColab_         |
+------------------------------------------------+----------------------+----------------------------+
| Simple Signed 3D Geodesic & Euclidean Distance | SimpleDemo3dSignedPy_| SimpleDemo3dSignedColab_   |
+------------------------------------------------+----------------------+----------------------------+
| 2D Geodesic & Euclidean Distance	         | Demo2dPy_            | Demo2dColab_               |
+------------------------------------------------+----------------------+----------------------------+
| 2D Signed Geodesic & Euclidean Distance        | Demo2dSignedPy_      | Demo2dSignedColab_         |
+------------------------------------------------+----------------------+----------------------------+
| 3D Geodesic & Euclidean Distance	         | Demo3dPy_            | Demo3dColab_               |
+------------------------------------------------+----------------------+----------------------------+
| 3D Signed Geodesic & Euclidean Distance        | Demo3dSignedPy_      | Demo3dSignedColab_         |
+------------------------------------------------+----------------------+----------------------------+
| 2D 2D GSF Segmentation Smoothing	         | DemoGSF2dPy_         | DemoGSF2dColab_            |
+------------------------------------------------+----------------------+----------------------------+

.. _SimpleDemo2dPy: https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo2d.py
.. _SimpleDemo2dColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo2d.ipynb
.. _SimpleDemo2dSignedPy: https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo2d_signed.py
.. _SimpleDemo2dSignedColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo2d_signed.ipynb
.. _SimpleDemo3dPy: https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo3d.py
.. _SimpleDemo3dColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo3d.ipynb
.. _SimpleDemo3dSignedPy: https://github.com/masadcv/FastGeodis/blob/master/samples/simpledemo3d_signed.py
.. _SimpleDemo3dSignedColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/simpledemo3d_signed.ipynb
.. _Demo2dPy: https://github.com/masadcv/FastGeodis/blob/master/samples/demo2d.py
.. _Demo2dSignedPy: https://github.com/masadcv/FastGeodis/blob/master/samples/demo2d_signed.py
.. _Demo2dColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo2d.ipynb
.. _Demo2dSignedColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo2d_signed.ipynb
.. _Demo3dPy: https://github.com/masadcv/FastGeodis/blob/master/samples/demo3d.py
.. _Demo3dSignedPy: https://github.com/masadcv/FastGeodis/blob/master/samples/demo3d_signed.py
.. _Demo3dColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo3d.ipynb
.. _Demo3dSignedColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demo3d_signed.ipynb
.. _DemoGSF2dPy: https://github.com/masadcv/FastGeodis/blob/master/samples/demoGSF2d_SmoothingSegExample.ipynb
.. _DemoGSF2dColab: https://colab.research.google.com/github/masadcv/FastGeodis/blob/master/samples/demoGSF2d_SmoothingSegExample.ipynb
