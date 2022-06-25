import math
import time
from functools import wraps

import torch
import FastGeodis
import GeodisTK
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

def geodistk_geodesic_distance_3d(I, S, spacing, lamb, iter):
    """Compute Geodesic Distance using GeodisTK raster scanning.

    I: input image, can have multiple channels. Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
    spacing: spacing of input 3d volume
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.

    from: https://github.com/taigw/GeodisTK/blob/master/demo2d.py

    """
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)


def demo_geodesic_distance3d():
    SHOW_JOINT_HIST = False
    input_name = "data/img3d.nii.gz"
    img = sitk.ReadImage(input_name)
    I = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233]
    S = np.zeros_like(I, np.uint8)
    S[10][60][70] = 1
    t0 = time.time()
    D1 = GeodisTK.geodesic3d_fast_marching(I, S, spacing)
    t1 = time.time()
    D2 = geodistk_geodesic_distance_3d(I, S, spacing, 1.0, 4)
    dt1 = t1 - t0
    dt2 = time.time() - t1
    It = torch.from_numpy(I).unsqueeze_(0).unsqueeze_(0)
    St = torch.from_numpy(1 - S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0)
    D3 = np.squeeze(
        FastGeodis.generalised_geodesic3d(
            It, St, spacing, 1e10, 1.0, 4
        ).numpy()
    )
    print("runtime(s) fast marching {0:}".format(dt1))
    print("runtime(s) raster scan   {0:}".format(dt2))

    img_d1 = sitk.GetImageFromArray(D1)
    img_d1.SetSpacing(spacing_raw)
    sitk.WriteImage(img_d1, "data/image3d_dis1.nii.gz")

    img_d2 = sitk.GetImageFromArray(D2)
    img_d2.SetSpacing(spacing_raw)
    sitk.WriteImage(img_d2, "data/image3d_dis2.nii.gz")

    img_d3 = sitk.GetImageFromArray(D3)
    img_d3.SetSpacing(spacing_raw)
    sitk.WriteImage(img_d3, "data/image3d_dis3.nii.gz")

    I_sub = sitk.GetImageFromArray(I)
    I_sub.SetSpacing(spacing_raw)
    sitk.WriteImage(I_sub, "data/image3d_sub.nii.gz")

    I = I * 255 / I.max()
    I = np.asarray(I, np.uint8)

    I_slice = I[10]
    D1_slice = D1[10]
    D2_slice = D2[10]
    D3_slice = D3[10]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(I_slice, cmap="gray")
    plt.autoscale(False)
    plt.plot([70], [60], "ro")
    plt.axis("off")
    plt.title("(a) Input image")

    plt.subplot(1, 5, 2)
    plt.imshow(D1_slice)
    plt.axis("off")
    plt.title("(b) Fast Marching")

    plt.subplot(1, 5, 3)
    plt.imshow(D2_slice)
    plt.axis("off")
    plt.title("(c) GeodisTK")

    plt.subplot(1, 5, 4)
    plt.imshow(D3_slice)
    plt.axis("off")
    plt.title("(d) FastGeodis")

    diff = D1 - D3
    diff_slice = diff[10]
    plt.subplot(1, 5, 5)
    plt.imshow(diff_slice)
    plt.axis("off")
    plt.title(
        "(d) Fast Marching \nvs. FastGeodis\nmax diff: {}\nmin diff: {}".format(
            np.max(diff), np.min(diff)
        )
    )
    plt.show()

    if SHOW_JOINT_HIST: 
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist2d(D1.flatten(), D2.flatten(), bins=50)
        plt.xlabel("Fast Marching")
        plt.ylabel("GeodisTK")
        plt.title("Joint histogram\nFast Marching vs. GeodisTK")
        # plt.gca().set_aspect("equal", adjustable="box")

        plt.subplot(1, 2, 2)
        plt.hist2d(D1.flatten(), D3.flatten(), bins=50)
        plt.xlabel("Fast Marching")
        plt.ylabel("FastGeodis")
        plt.title("Joint histogram\nFast Marching vs. FastGeodis")
        # plt.gca().set_aspect("equal", adjustable="box")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    demo_geodesic_distance3d()
