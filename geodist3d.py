import GeodisTK
import math
from PIL import Image
from matplotlib import image
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


@timing
def generalised_geodesic_distance_2d_geodistk(I, S, v, lamb, itr):
    """
    get 2d geodesic disntance by raser scanning.
    I: input image, can have multiple channels. Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    """
    # return GeodisTK.generalised_geodesic2d_raster_scan(I, S, v, lamb, itr)
    return GeodisTK.generalised_geodesic2d_raster_scan_opt(I, S, v, lamb, itr)
    


def get_l2_distance(p_val, q_val, dim=1):
    return torch.sqrt(torch.sum((p_val - q_val) ** 2, dim=dim))


def get_l1_distance(p_val, q_val, dim=1):
    return torch.sum(torch.abs(p_val - q_val), dim=dim)


@timing
def generalised_geodesic3d_raster_2scan(image, mask, spacing, v, lamda, itr):
    batch, channel, depth, height, width = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()
    spacing_squared = [a * a for a in spacing]
    use_speed = True
    for it in range(itr):
        # int dd_f[13] = {-1, -1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1};
        # int dh_f[13] = {-1, -1, -1,  0,  0, -1, -1, -1,  0, -1, -1, -1,  0};
        # int dw_f[13] = {-1,  0,  1, -1,  0, -1,  0,  1, -1, -1,  0,  1, -1};
        
        # d_td = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
        d_td = [(-1, -1, -1), (-1, -1, 0), (-1, -1, 1), (-1, 0, -1), (-1, 0, 0), (0, -1, -1), (0, -1, 0), (0, -1, 1), (0, 0, -1), (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1)]    
        local_dis_td = [math.sqrt(abs(a) * spacing_squared[0] + abs(b) * spacing_squared[1], abs(c) * spacing_squared[2]) for a, b, c in d_td]

        # forward scan
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    p_val = image[..., d, h, w]

                    c_d_td = [(d + dd, h + dh, w + dw) for dd, dh, dw in d_td]
                    for idx, (d_d_td, h_d_td, w_d_td) in enumerate(c_d_td):
                        if d_d_td < 0 or d_d_td >= depth or h_d_td < 0 or h_d_td >= height or w_d_td < 0 or w_d_td >= width:
                            continue
                        q_dis = distance[..., d_d_td, h_d_td, w_d_td]
                        q_val = image[..., d_d_td, h_d_td, w_d_td]
                        if use_speed:
                            l2dis = get_l2_distance(p_val, q_val)
                            speed = (1.0 - lamda) + lamda / (l2dis + 1e-5)
                            delta_d = local_dis_td[idx] / speed
                        else:
                            l1dis = get_l1_distance(p_val, q_val) ** 2
                            delta_d = torch.sqrt(
                                local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis
                            )
                        cur_dis = q_dis + delta_d
                        if cur_dis < distance[..., d, h, w]:
                            distance[..., d, h, w] = cur_dis

        # backward
        # int dd_b[13] = {-1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1,  1};
        # int dh_b[13] = { 0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  1,  1,  1};
        # int dw_b[13] = { 1, -1,  0,  1,  1, -1,  0,  1,  0,  1, -1,  0,  1};
        d_td = [(-1, 0, 1), (-1, 1, -1), (-1, -1, 0), (-1, 1, 1), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)]
        local_dis_td = [math.sqrt(abs(a) * spacing_squared[0] + abs(b) * spacing_squared[1], abs(c) * spacing_squared[2]) for a, b, c in d_td]
        
        for d in reversed(range(depth)):
            for h in reversed(range(height)):
                for w in reversed(range(width)):
                    p_val = image[..., d, h, w]

                    c_d_td = [(d + dd, h + dh, w + dw) for dd, dh, dw in d_td]
                    for idx, (d_d_td, h_d_td, w_d_td) in enumerate(c_d_td):
                        if d_d_td < 0 or d_d_td >= depth or h_d_td < 0 or h_d_td >= height or w_d_td < 0 or w_d_td >= width:
                            continue
                        q_dis = distance[..., d_d_td, h_d_td, w_d_td]
                        q_val = image[..., d_d_td, h_d_td, w_d_td]
                        if use_speed:
                            l2dis = get_l2_distance(p_val, q_val)
                            speed = (1.0 - lamda) + lamda / (l2dis + 1e-5)
                            delta_d = local_dis_td[idx] / speed
                        else:
                            l1dis = get_l1_distance(p_val, q_val) ** 2
                            delta_d = torch.sqrt(
                                local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis
                            )
                        cur_dis = q_dis + delta_d
                        if cur_dis < distance[..., d, h, w]:
                            distance[..., d, h, w] = cur_dis

    return distance


def test_compare_original_imp():
    img = np.asarray(
        Image.open("data/brain.png")
        .convert("L")
        .resize((128, 128), resample=Image.BILINEAR)
    )
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)

    #
    _, _, height, width = img.shape
    msk = np.ones_like(img)
    msk[..., int(height / 2), int(width / 2)] = 0

    img = torch.from_numpy(img.astype(np.float32))
    msk = torch.from_numpy(msk.astype(np.float32))

    img_np = np.squeeze(img.detach().cpu().numpy())
    msk_np = np.squeeze(msk.detach().cpu().numpy())

    dst1 = generalised_geodesic_distance_2d_geodistk(
        img_np, msk_np, v=1e10, lamb=1.0, itr=2
    )

    dst2 = (
        generalised_geodesic2d_raster_2scan(
            image=img.to("cuda"), mask=msk.to("cuda"), v=1e10, lamda=1.0, itr=2
        )
        .squeeze_()
        .detach()
        .cpu()
        .numpy()
    )

    diff = np.sum(np.abs(dst1 - dst2))
    print(diff)
    if diff > 0:
        print("Two implementations are not same, diff {}".format(diff))

    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(dst1))
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(dst2))
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(np.abs(dst1 - dst2)))
    plt.show()

    assert diff < 1




if __name__ == "__main__":

