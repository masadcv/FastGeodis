import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import SimpleITK as sitk
import FastGeodis
import dijkstra3d

def fastgeodis_generalised_geodesic_distance_3d(I, S, spacing, v, lamb, iter):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, lamb, 1-lamb, iter)

def demo_geodesic_distance3d():
    # read input volume and create seed
    input_name = "data/img3d.nii.gz"
    img = sitk.ReadImage(input_name)
    I = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233]
    S = np.zeros_like(I, np.float32)
    S[10][60][70] = 1

    # run and time each method
    iterations = 4
    v = 1e10
    lamb = 1.0

    device = "cpu"
    It = torch.from_numpy(I).unsqueeze_(0).unsqueeze_(0).float().to(device)
    St = torch.from_numpy(1 - S).unsqueeze_(0).unsqueeze_(0).to(device)
    
    tic = time.time()
    fastraster_outuput_cpu = np.squeeze(
        fastgeodis_generalised_geodesic_distance_3d(
            It, St, spacing, v, lamb, iterations
        ).cpu().numpy()
    )
    fastraster_time_cpu = time.time() - tic

    device = "cuda" if It.shape[1] == 1 and torch.cuda.is_available() else None
    if device:
        It = It.to(device)
        St = St.to(device)
        
        tic = time.time()
        fastraster_outuput_gpu = np.squeeze(
            fastgeodis_generalised_geodesic_distance_3d(
                It, St, spacing, v, lamb, iterations
            ).cpu().numpy()
        )
        fastraster_time_gpu = time.time() - tic
    
    I_np = It.squeeze().cpu().numpy()
    tic = time.time()
    dijkstra3d_output = dijkstra3d.distance_field(
                    data=I_np,
                    prob=np.zeros_like(I_np),
                    source=[10,60,70], 
                    connectivity=26, 
                    spacing=spacing, 
                    l_grad=lamb, 
                    l_eucl=1.0-lamb,
                    l_prob=0.0)
    dijkstra3d_time = time.time() - tic

    print('Runtimes:')
    print('dijkstra3d: {:.2f} ms \nFastGeodis CPU raster: {:.2f} ms'.format(\
        dijkstra3d_time*1000, fastraster_time_cpu*1000))

    if device:
        print('FastGeodis GPU raster: {:.2f} ms'.format(fastraster_time_gpu*1000))

    I = I * 255 / I.max()
    I = np.asarray(I, np.uint8)

    ## Coronal Figure
    I_slice = I[:,60,:]
    fastraster_outuput_cpu_slice = fastraster_outuput_cpu[:,60,:]
    fastraster_outuput_gpu_slice = fastraster_outuput_gpu[:,60,:]
    dijkstra3d_output_slice = dijkstra3d_output[:,60,:]
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Comparision with l_eucl={1.0-lamb}, l_grad={lamb}")
    plt.subplot(2, 4, 1)
    plt.imshow(I_slice, cmap="gray", aspect=spacing[0]/spacing[2])
    plt.autoscale(False)
    plt.plot([70], [10], "ro")
    plt.axis("off")
    plt.title("(a) Input image")
    
    plt.subplot(2, 4, 2)
    plt.imshow(dijkstra3d_output_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(f"(b) dijkstra3d ({dijkstra3d_time*1000:.2f} ms)")

    plt.subplot(2, 4, 3)
    plt.imshow(fastraster_outuput_cpu_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(f"(c) FastGeodis ({fastraster_time_cpu*1000:.2f} ms)")

    plt.subplot(2, 4, 4)
    plt.imshow(fastraster_outuput_gpu_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(f"(d) FastGeodis ({fastraster_time_gpu*1000:.2f} ms)")

    diff = abs(fastraster_outuput_cpu - dijkstra3d_output)/(dijkstra3d_output+1e-7)*100
    diff_slice = diff[:,60,:]
    plt.subplot(2, 4, 7)
    plt.imshow(diff_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(
        "(e) dijkstra3d \nvs. FastGeodis (cpu) \ndiff: max: {:.4f}\nmin: {:.4f}".format(
            np.max(diff), np.min(diff)
        )
    )
    diff = abs(fastraster_outuput_gpu - dijkstra3d_output)/(dijkstra3d_output+1e-7)*100
    diff_slice = diff[:,60,:]
    plt.subplot(2, 4, 8)
    plt.imshow(diff_slice, aspect=spacing[0]/spacing[2])
    plt.axis("off")
    plt.title(
        "(f) dijkstra3d \nvs. FastGeodis (gpu) \ndiff: max: {:.4f}\nmin: {:.4f}".format(
            np.max(diff), np.min(diff)
        )
    )
    plt.savefig("figures/3d_coronal.png")
    plt.show()
    # plt.colorbar()

    ## Axial Figure
    I_slice = I[10,:,:]
    fastraster_outuput_cpu_slice = fastraster_outuput_cpu[10,:,:]
    fastraster_outuput_gpu_slice = fastraster_outuput_gpu[10,:,:]
    dijkstra3d_output_slice = dijkstra3d_output[10,:,:]

    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Comparision with l_eucl={1.0-lamb}, l_grad={lamb}")
    plt.subplot(2, 4, 1)
    plt.imshow(I_slice, cmap="gray", aspect=spacing[1]/spacing[2])
    plt.autoscale(False)
    plt.plot([70], [60], "ro")
    plt.axis("off")
    plt.title("(a) Input image")

    plt.subplot(2, 4, 2)
    plt.imshow(dijkstra3d_output_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(f"(b) dijkstra3d {dijkstra3d_time*1000:.2f} ms")  

    plt.subplot(2, 4, 3)
    plt.imshow(fastraster_outuput_cpu_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(f"(c) FastGeodis (cpu) ({fastraster_time_cpu*1000:.2f} ms)")

    plt.subplot(2, 4, 4)
    plt.imshow(fastraster_outuput_gpu_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(f"(d) FastGeodis (gpu) ({fastraster_time_gpu*1000:.2f} ms)")

    diff = abs(fastraster_outuput_cpu - dijkstra3d_output)/(dijkstra3d_output+1e-7)*100
    diff_slice = diff[10,:,:]
    plt.subplot(2, 4, 7)
    plt.imshow(diff_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(
        "(d) dijkstra3d \nvs. FastGeodis (cpu) \ndiff: max: {:.4f}\nmin: {:.4f}".format(
            np.max(diff), np.min(diff)
        )
    )

    diff = abs(fastraster_outuput_gpu - dijkstra3d_output)/(dijkstra3d_output+1e-7)*100
    diff_slice = diff[10,:,:]
    plt.subplot(2, 4, 8)
    plt.imshow(diff_slice, aspect=spacing[1]/spacing[2])
    plt.axis("off")
    plt.title(
        "(d) dijkstra3d \nvs. FastGeodis (gpu) \ndiff: max: {:.4f}\nmin: {:.4f}".format(
            np.max(diff), np.min(diff)
        )
    )
    # plt.colorbar()
    plt.savefig("figures/3d_axial.png")
    plt.show()

    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.hist2d(dijkstra3d_output.flatten(), fastraster_outuput_cpu.flatten(), bins=50)
    plt.xlabel("dijkstra3d")
    plt.ylabel("FastGeodis (cpu)")
    plt.title("Joint histogram\ndijkstra3d vs. FastGeodis (cpu)")
    plt.gca().set_aspect("equal", adjustable="box")

    if device:
        plt.subplot(1, 2, 2)
        plt.hist2d(dijkstra3d_output.flatten(), fastraster_outuput_gpu.flatten(), bins=50)
        plt.xlabel("dijkstra3d")
        plt.ylabel("FastGeodis (gpu)")
        plt.title("Joint histogram\ndijkstra3d vs. FastGeodis (gpu)")
        plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_geodesic_distance3d()
