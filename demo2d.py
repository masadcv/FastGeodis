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
    return FastGeodis.generalised_geodesic2d(I, S, v, lamb, iter)


def demo_geodesic_distance2d(img, seed_pos):
    I = np.asanyarray(img, np.float32)
    S = np.zeros((I.shape[0], I.shape[1]), np.uint8)
    S[seed_pos[0]][seed_pos[1]] = 1
    t0 = time.time()
    D1 = GeodisTK.generalised_geodesic2d_fast_marching(I, 1-S.astype(np.float32), 1e10)
    t1 = time.time()
    D2 = geodistk_generalised_geodesic_distance_2d(I, 1-S.astype(np.float32), 1e10, 1.0, 2)
    dt1 = t1 - t0
    dt2 = time.time() - t1

    if I.ndim == 3:
        I = np.moveaxis(I, -1, 0)
    else:
        I = np.expand_dims(I, 0)
    It = torch.from_numpy(I).unsqueeze_(0)
    St = torch.from_numpy(1-S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0)

    D3 = np.squeeze(fastgeodis_generalised_geodesic_distance_2d(It, St, 1e10, 1.0, 2).numpy())
    print("runtime(s) of fast marching {0:}".format(dt1))
    print("runtime(s) of raster  scan  {0:}".format(dt2))

    plt.figure(figsize=(15,5))
    plt.subplot(1,5,1); plt.imshow(img)
    plt.autoscale(False);  plt.plot([seed_pos[0]], [seed_pos[1]], 'ro')
    plt.axis('off'); plt.title('(a) Input image')
    
    plt.subplot(1,5,2); plt.imshow(D1)
    plt.axis('off'); plt.title('(b) Fast Marching')
    
    plt.subplot(1,5,3); plt.imshow(D2)
    plt.axis('off'); plt.title('(c) GeodisTK')

    plt.subplot(1,5,4); plt.imshow(D3)
    plt.axis('off'); plt.title('(d) FastGeodis')

    diff = D1-D3
    plt.subplot(1,5,5); plt.imshow(diff)
    plt.axis('off'); plt.title('(e) Fast Marching\nvs. FastGeodis\nmax diff: {}\nmin diff: {}'.format(np.max(diff), np.min(diff)))
    plt.show()

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

def demo_geodesic_distance2d_gray_scale_image():
    img = Image.open('data/img2d.png').convert('L')
    seed_position = [100, 100]
    demo_geodesic_distance2d(img, seed_position)

def demo_geodesic_distance2d_RGB_image():
    img = Image.open('data/ISIC_546.jpg')
    seed_position = [128, 128]
    demo_geodesic_distance2d(img, seed_position)

if __name__ == '__main__':
    print("example list")
    print(" 0 -- example for gray scale image")
    print(" 1 -- example for RB image")
    print("please enter the index of an example:")
    # method = input()
    # method = '{0:}'.format(method)
    method = '0'
    if(method == '0'):
        demo_geodesic_distance2d_gray_scale_image()
    elif(method == '1'):
        demo_geodesic_distance2d_RGB_image()
    else:
        print("invalid number : {0:}".format(method))
