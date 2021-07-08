import math
from functools import wraps
from time import time

import GeodisTK
import matplotlib.pyplot as plt
import numpy as np
import torch
import geodis
from PIL import Image


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def get_l2_distance(p_val, q_val, dim=1):
    return torch.sqrt(torch.sum((p_val - q_val) ** 2, dim=dim))


def get_l1_distance(p_val, q_val, dim=1):
    return torch.sum(torch.abs(p_val - q_val), dim=dim)


@timing
def generalised_geodesic_distance_2d_geodistk(I, S, v, lamda, iter):
    """
    get 2d geodesic disntance by raser scanning.
    I: input image, can have multiple channels. Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
    lamda: weighting betwween 0.0 and 1.0
          if lamda==0.0, return spatial euclidean distance without considering gradient
          if lamda==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    """
    return GeodisTK.generalised_geodesic2d_raster_scan_opt(I, S, v, lamda, iter)


@timing
def generalised_geodesic2d_raster_2scan(image, mask, v, lamda, iter):
    batch, _, height, width = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()

    use_speed = True
    for it in range(iter):
        d_td = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]

        # forward scan
        for h in range(height):
            for w in range(width):
                p_val = image[..., h, w]

                c_d_td = [(h + dh, w + dw) for dh, dw in d_td]
                for idx, (h_d_td, w_d_td) in enumerate(c_d_td):
                    if h_d_td < 0 or h_d_td >= height or w_d_td < 0 or w_d_td >= width:
                        continue
                    q_dis = distance[..., h_d_td, w_d_td]
                    q_val = image[..., h_d_td, w_d_td]
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
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

        # backward
        d_td = [(0, 1), (1, -1), (1, 0), (1, 1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]
        for h in reversed(range(height)):
            for w in reversed(range(width)):
                p_val = image[..., h, w]

                c_d_td = [(h + dh, w + dw) for dh, dw in d_td]
                for idx, (h_d_td, w_d_td) in enumerate(c_d_td):
                    if h_d_td < 0 or h_d_td >= height or w_d_td < 0 or w_d_td >= width:
                        continue
                    q_dis = distance[..., h_d_td, w_d_td]
                    q_val = image[..., h_d_td, w_d_td]
                    l2dis = get_l2_distance(p_val, q_val)
                    speed = (1.0 - lamda) + lamda / (l2dis + 1e-5)
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
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

    return distance


def geodesic_topdown_pass(image, distance, lamda):
    _, _, height, width = image.shape
    w_index_z0 = [x for x in range(width)]
    w_index_m1 = [(x - 1) if (x - 1) > 0 else 0 for x in w_index_z0]
    w_index_p1 = [(x + 1) if (x + 1) < width else width - 1 for x in w_index_z0]

    use_speed = True
    for h in range(1, height):
        p_val_vec = image[..., h, :]
        for wi, w_index in enumerate([w_index_m1, w_index_z0, w_index_p1]):
            local_dist = 1 + abs(wi - 1)
            q_dis_vec = distance[..., h - 1, w_index]
            q_val_vec = image[..., h - 1, w_index]
            if use_speed:
                l2dist = get_l2_distance(p_val_vec, q_val_vec, 1)
                speed = (1.0 - lamda) + lamda / (l2dist + 1e-5)
                delta_d = math.sqrt(local_dist) / speed
            else:
                q_l1dist_z0_vec = torch.squeeze(torch.abs(p_val_vec - q_val_vec)) ** 2
                delta_d = torch.sqrt(local_dist + (lamda ** 2) * q_l1dist_z0_vec)
            distance[..., h, :] = torch.minimum(distance[..., h, :], (q_dis_vec + delta_d))


    for h in reversed(range(height - 1)):
        p_val_vec = image[..., h, :]
        for wi, w_index in enumerate([w_index_m1, w_index_z0, w_index_p1]):
            local_dist = 1 + abs(wi - 1)
            q_dis_vec = distance[..., h + 1, w_index]
            q_val_vec = image[..., h + 1, w_index]
            if use_speed:
                l2dist = get_l2_distance(p_val_vec, q_val_vec, 1)
                speed = (1.0 - lamda) + lamda / (l2dist + 1e-5)
                delta_d = math.sqrt(local_dist) / speed
            else:
                q_l1dist_z0_vec = get_l1_distance(p_val_vec, q_val_vec) ** 2
                delta_d = torch.sqrt(local_dist + (lamda ** 2) * q_l1dist_z0_vec)
            distance[..., h, :] = torch.minimum(distance[..., h, :], (q_dis_vec + delta_d))

    return distance

# refence implementation in python
@timing
def generalised_geodesic2d_raster_4scan(image, mask, v, lamda, iter):
    batch, _, _, _ = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()

    for it in range(iter):
        # top-down - height*, width
        distance = geodesic_topdown_pass(image, distance, lamda)

        # left-right - width*, height
        image = torch.transpose(image, dim0=3, dim1=2)
        distance = torch.transpose(distance, dim0=3, dim1=2)
        distance = geodesic_topdown_pass(image, distance, lamda)

        # transpose back to original - height, width
        image = torch.transpose(image, dim0=3, dim1=2)
        distance = torch.transpose(distance, dim0=3, dim1=2)

        # * indicates the current direction of pass

    return distance

# cpp implementation
@timing
def generalised_geodesic2d_raster_4scan_cpp(I, S, v, lamda, iter):
    return geodis.generalised_geodesic2d(I, S, v, lamda, iter)


def test_compare():
    img = np.asarray(
        Image.open("data/brain.png")
        .convert("L")
        .resize((1024, 1024), resample=Image.BILINEAR)
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
        img_np, msk_np, v=1e10, lamda=1.0, iter=2
    )
    # dst1 = (
    #     generalised_geodesic2d_raster_4scan(img, msk, v=1e10, lamda=1.0, iter=2)
    #     .squeeze_()
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    dst2 = (
        generalised_geodesic2d_raster_4scan_cpp(img, msk, v=1e10, lamda=1.0, iter=2)
        .squeeze_()
        .detach()
        .cpu()
        .numpy()
    )

    # dst2 = (
    #     generalised_geodesic2d_raster_4scan_cpp_init(
    #         img, msk, v=1e10, lamda=1.0, iter=2
    #     )
    #     .squeeze_()
    #     .detach()
    #     .cpu()
    #     .numpy()
    # )

    # np.testing.assert_allclose(dst2, dst2)

    diff = np.sum(np.abs(dst1 - dst2))
    print(diff)
    if diff > 0:
        print("Two implementations are not same, diff {}".format(diff))

    plt.subplot(1, 4, 1)
    plt.imshow(np.squeeze(img))
    plt.subplot(1, 4, 2)
    plt.imshow(np.squeeze(dst1))
    plt.subplot(1, 4, 3)
    plt.imshow(np.squeeze(dst2))
    plt.subplot(1, 4, 4)
    plt.imshow(np.squeeze(np.abs(dst1 - dst2)))
    plt.show()

    assert diff < 1


if __name__ == "__main__":
    test_compare()
