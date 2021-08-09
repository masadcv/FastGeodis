import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import SimpleITK as sitk
import FastGeodis
import dijkstra3d

lam = 1.0
l_eucl = (1-lam)
l_grad = lam
def fastgeodis_generalised_geodesic_distance_3d(I, S, spacing, v, l_grad, l_eucl, iter):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, l_grad, l_eucl, iter)


def demo_geodesic_distance3d():
    input_name = "data/img3d.nii.gz"
    img = sitk.ReadImage(input_name)
    I = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233]
    S = np.zeros_like(I, np.uint8)
    S[10][60][70] = 1


    It = torch.from_numpy(I).unsqueeze_(0).unsqueeze_(0).float()
    St = torch.from_numpy(1 - S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0)
    
    time_raster = time.time()
    D_raster = np.squeeze(
        fastgeodis_generalised_geodesic_distance_3d(
            It, St, spacing, 1e10, l_grad, l_eucl, 4
        ).numpy()
    )
    time_raster = time.time() - time_raster
    
    
    I_np = It.squeeze().numpy()
    time_dijkstra = time.time()
    D_dijkstra3d = dijkstra3d.distance_field(
                    data=I_np,
                    prob=np.zeros_like(I_np),
                    source=[10,60,70], 
                    connectivity=26, 
                    spacing=spacing, 
                    l_grad=l_grad, 
                    l_eucl=l_eucl,
                    l_prob=0.0)
    time_dijkstra = time.time() - time_dijkstra



    I = I * 255 / I.max()
    I = np.asarray(I, np.uint8)

    ## Coronal Figure
    I_slice = I[:,60,:]
    D_raster_slice = D_raster[:,60,:]
    D_dijkstra3d_slice = D_dijkstra3d[:,60,:]
    
    plt.figure(figsize=(15, 3))
    plt.suptitle(f"Comparision with l_eucl={l_eucl}, l_grad={l_grad}")
    plt.subplot(1, 4, 1)
    plt.imshow(I_slice, cmap="gray", aspect=spacing[0]/spacing[2])
    plt.autoscale(False)
    plt.plot([70], [10], "ro")
    plt.axis("off")
    plt.title("(a) Input image")

    plt.subplot(1, 4, 2)
    plt.imshow(D_raster_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(f"(b) FastGeodis ({time_raster*1000:.2f}ms)")

    plt.subplot(1, 4, 3)
    plt.imshow(D_dijkstra3d_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(f"(c) dijkstra3d {time_dijkstra*1000:.2f}ms")    

    diff = abs(D_raster - D_dijkstra3d)/(D_dijkstra3d+1e-7)*100
    diff_slice = diff[:,60,:]
    plt.subplot(1, 4, 4)
    plt.imshow(diff_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(
        "(d) FastGeodis \nvs. dijkstra3d \nmax relative diff (%): {}\nmin relative diff (%)): {}".format(
            np.max(diff), np.min(diff)
        )
    )
    plt.colorbar()
    #plt.show()
    plt.savefig("figures/3d_coronal.png")

    ## Axial Figure
    I_slice = I[10,:,:]
    D_raster_slice = D_raster[10,:,:]
    D_dijkstra3d_slice = D_dijkstra3d[10,:,:]

    plt.figure(figsize=(15, 3))
    plt.suptitle(f"Comparision with l_eucl={l_eucl}, l_grad={l_grad}")
    plt.subplot(1, 4, 1)
    plt.imshow(I_slice, cmap="gray", aspect=spacing[1]/spacing[2])
    plt.autoscale(False)
    plt.plot([70], [60], "ro")
    plt.axis("off")
    plt.title("(a) Input image")

    plt.subplot(1, 4, 2)
    plt.imshow(D_raster_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(f"(b) FastGeodis ({time_raster*1000:.2f}ms)")

    plt.subplot(1, 4, 3)
    plt.imshow(D_dijkstra3d_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(f"(c) dijkstra3d {time_dijkstra*1000:.2f}ms")    

    diff = abs(D_raster - D_dijkstra3d)/(D_dijkstra3d+1e-7)*100
    diff_slice = diff[10,:,:]
    plt.subplot(1, 4, 4)
    plt.imshow(diff_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(
        "(d) FastGeodis \nvs. dijkstra3d \nmax relative diff (%): {}\nmin relative diff (%)): {}".format(
            np.max(diff), np.min(diff)
        )
    )
    plt.colorbar()
    #plt.show()
    plt.savefig("figures/3d_axial.png")

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist2d(D1.flatten(), D2.flatten(), bins=50)
    # plt.xlabel("Fast Marching")
    # plt.ylabel("GeodisTK")
    # plt.title("Joint histogram\nFast Marching vs. GeodisTK")
    # # plt.gca().set_aspect("equal", adjustable="box")

    # plt.subplot(1, 2, 2)
    # plt.hist2d(D1.flatten(), D3.flatten(), bins=50)
    # plt.xlabel("Fast Marching")
    # plt.ylabel("FastGeodis")
    # plt.title("Joint histogram\nFast Marching vs. FastGeodis")
    # # plt.gca().set_aspect("equal", adjustable="box")

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    demo_geodesic_distance3d()
