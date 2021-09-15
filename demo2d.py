import GeodisTK
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import FastGeodis

def geodistk_generalised_geodesic_distance_2d(I, S, v, lamb, iter):
    return GeodisTK.generalised_geodesic2d_raster_scan(I, S, v, lamb, iter)

def fastgeodis_generalised_geodesic_distance_2d(I, S, v, lamb, iter):
    return FastGeodis.generalised_geodesic2d(I, S, v, lamb, 1-lamb, iter)


def demo_geodesic_distance2d(img, seed_pos):
    # get image and create seed image
    I = np.asanyarray(img, np.float32)
    S = np.zeros((I.shape[0], I.shape[1]), np.float32)
    S[seed_pos[0]][seed_pos[1]] = 1

    # run and time each method
    iterations = 2
    v = 1e10
    lamb = 1.0

    tic = time.time()
    fastmarch_output = GeodisTK.generalised_geodesic2d_fast_marching(I, 1-S, v)
    fastmarch_time = time.time()-tic

    tic = time.time()
    geodistkraster_output = geodistk_generalised_geodesic_distance_2d(I, 1-S, v, lamb, iterations)
    geodistkraster_time = time.time() - tic

    if I.ndim == 3:
        I = np.moveaxis(I, -1, 0)
    else:
        I = np.expand_dims(I, 0)
    
    device = "cpu"
    It = torch.from_numpy(I).unsqueeze_(0).to(device)
    St = torch.from_numpy(1-S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to(device)
        
    tic = time.time()
    fastraster_output_cpu = np.squeeze(fastgeodis_generalised_geodesic_distance_2d(It, St, v, lamb, iterations).cpu().numpy())
    fastraster_time_cpu = time.time() - tic

    device = "cuda" if It.shape[1] == 1 and torch.cuda.is_available() else None
    if device:
        It = It.to(device)
        St = St.to(device)
        
        tic = time.time()
        fastraster_output_gpu = np.squeeze(fastgeodis_generalised_geodesic_distance_2d(It, St, v, lamb, iterations).cpu().numpy())
        fastraster_time_gpu = time.time() - tic
        
    print('Runtimes:')
    print('Fast Marching: {:.6f} s \nGeodisTk raster: {:.6f} s \nFastGeodis CPU raster: {:.6f} s'.format(\
        fastmarch_time, geodistkraster_time, fastraster_time_cpu))

    if device:
        print('FastGeodis GPU raster: {:.6f} s'.format(fastraster_time_gpu))

    plt.figure(figsize=(18,6))
    plt.subplot(2,4,1); plt.imshow(img)
    plt.autoscale(False);  plt.plot([seed_pos[0]], [seed_pos[1]], 'ro')
    plt.axis('off'); plt.title('(a) Input image')
    
    plt.subplot(2,4,2); plt.imshow(fastmarch_output)
    plt.axis('off'); plt.title('(b) Fast Marching | time: {:.4f} s'.format(fastmarch_time))

    plt.subplot(2,4,3); plt.imshow(fastraster_output_cpu)
    plt.axis('off'); plt.title('(c) FastGeodis (cpu) | time: {:.4f} s'.format(fastraster_time_cpu))

    plt.subplot(2,4,6); plt.imshow(geodistkraster_output)
    plt.axis('off'); plt.title('(d) GeodisTK | time: {:.4f} s'.format(geodistkraster_time))


    if device:
        plt.subplot(2,4,7); plt.imshow(fastraster_output_gpu)
        plt.axis('off'); plt.title('(e) FastGeodis (gpu) | time: {:.4f} s'.format(fastraster_time_gpu))

    diff = fastmarch_output-fastraster_output_cpu
    plt.subplot(2,4,4); plt.imshow(diff)
    plt.axis('off'); plt.title('(f) Fast Marching vs. FastGeodis (cpu)\ndiff: max: {:.4f} | min: {:.4f}'.format(np.max(diff), np.min(diff)))
    
    if device:
        diff = fastmarch_output-fastraster_output_gpu
        plt.subplot(2,4,8); plt.imshow(diff)
        plt.axis('off'); plt.title('(g) Fast Marching vs. FastGeodis (gpu)\ndiff: max: {:.4f} | min: {:.4f}'.format(np.max(diff), np.min(diff)))
    
    plt.show()

    plt.figure(figsize=(14,4))
    plt.subplot(1, 3, 1)
    plt.hist2d(fastmarch_output.flatten(), geodistkraster_output.flatten(), bins=50)
    plt.xlabel("Fast Marching")
    plt.ylabel("GeodisTK")
    plt.title("Joint histogram\nFast Marching vs. GeodisTK")
    # plt.gca().set_aspect("equal", adjustable="box")

    plt.subplot(1, 3, 2)
    plt.hist2d(fastmarch_output.flatten(), fastraster_output_cpu.flatten(), bins=50)
    plt.xlabel("Fast Marching")
    plt.ylabel("FastGeodis (cpu)")
    plt.title("Joint histogram\nFast Marching vs. FastGeodis (cpu)")
    # plt.gca().set_aspect("equal", adjustable="box")

    if device:
        plt.subplot(1, 3, 3)
        plt.hist2d(fastmarch_output.flatten(), fastraster_output_gpu.flatten(), bins=50)
        plt.xlabel("Fast Marching")
        plt.ylabel("FastGeodis (gpu)")
        plt.title("Joint histogram\nFast Marching vs. FastGeodis (gpu)")
        # plt.gca().set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

def demo_geodesic_distance2d_gray_scale_image():
    img = Image.open('data/img2d.png').convert('L')
    # make image bigger to check how much workload each method can take
    imgsize = img.size
    scale = 6
    imgsize = [x * scale for x in imgsize]
    img = img.resize(imgsize)
    seed_position = [100 * scale, 100 * scale]
    demo_geodesic_distance2d(img, seed_position)

def demo_geodesic_distance2d_RGB_image():
    img = Image.open('data/ISIC_546.jpg')
    # make image bigger to check how much workload each method can take
    imgsize = img.size
    scale = 6
    imgsize = [x * scale for x in imgsize]
    img = img.resize(imgsize)
    seed_position = [128 * scale, 128 * scale]
    demo_geodesic_distance2d(img, seed_position)

if __name__ == '__main__':
    print("example list")
    print(" 0 -- example for gray scale image")
    print(" 1 -- example for RGB image")
    print("please enter the index of an example:")
    # method = input()
    # method = '{0:}'.format(method)
    method = '1'
    if(method == '0'):
        demo_geodesic_distance2d_gray_scale_image()
    elif(method == '1'):
        demo_geodesic_distance2d_RGB_image()
    else:
        print("invalid number : {0:}".format(method))
