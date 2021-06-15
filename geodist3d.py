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


def get_l2_distance(p_val, q_val, dim=1):
    return torch.sqrt(torch.sum((p_val - q_val) ** 2, dim=dim))


def get_l1_distance(p_val, q_val, dim=1):
    return torch.sum(torch.abs(p_val - q_val), dim=dim)



@timing
def generalised_geodesic2d_raster_scan(image, mask, v, lamda, itr):
    batch, channel, height, width = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()
    use_speed = False
    for it in range(itr):
        d_td = [(-1, -1), (-1, 0), (-1, 1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]

        # top-down
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
                        delta_d = torch.sqrt(local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis)
                    cur_dis = q_dis + delta_d
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

        # bottom-up
        d_td = [(1, -1), (1, 0), (1, 1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]
        for h in reversed(range(height)):
            for w in range(width):
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
                        delta_d = torch.sqrt(local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis)
                    cur_dis = q_dis + delta_d
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

        # # left-right
        d_td = [(-1, -1), (0, -1), (1, -1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]
        for w in range(width):
            for h in range(height):
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
                        delta_d = torch.sqrt(local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis)
                    cur_dis = q_dis + delta_d
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

        # right-left
        d_td = [(-1, 1), (0, 1), (1, 1)]
        local_dis_td = [math.sqrt(a ** 2 + b ** 2) for a, b in d_td]
        for w in reversed(range(width)):
            for h in range(height):
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
                        delta_d = torch.sqrt(local_dis_td[idx] ** 2 + (lamda ** 2) * l1dis)
                    cur_dis = q_dis + delta_d
                    if cur_dis < distance[..., h, w]:
                        distance[..., h, w] = cur_dis

    return distance


@timing
def generalised_geodesic2d_raster_scan_vectorised(image, mask, v, lamda, itr):
    batch, channel, height, width = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()

    for it in range(itr):
        # top-down
        w_index_z0 = [x for x in range(width)]
        w_index_m1 = [(x-1) if (x-1) > 0 else 0 for x in w_index_z0]
        w_index_p1 = [(x+1) if (x+1) < width else width-1 for x in w_index_z0]

        w_dist_z0 = 1
        w_dist_pm1 = 2

        for h in range(1, height):
            new_dist = []

            p_val_vec = image[..., h, :]
            p_dis_vec = distance[..., h, :]
            new_dist.append(p_dis_vec)

            # read row-1
            q_dis_z0_vec = distance[..., h-1, w_index_z0]
            q_val_z0_vec = image[..., h-1, w_index_z0]
            q_l1dist_z0_vec = torch.squeeze(torch.abs(p_val_vec-q_val_z0_vec))  ** 2
            delta_d = torch.sqrt(w_dist_z0 + (lamda ** 2) * q_l1dist_z0_vec)
            new_dist.append(q_dis_z0_vec + delta_d)

            q_dis_m1_vec = distance[..., h-1, w_index_m1]
            q_val_m1_vec = image[..., h-1, w_index_m1]
            q_l1dist_m1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_m1_vec))  ** 2
            delta_d = torch.sqrt(w_dist_pm1 + (lamda ** 2) * q_l1dist_m1_vec)
            new_dist.append(q_dis_m1_vec + delta_d)

            q_dis_p1_vec = distance[..., h-1, w_index_p1]
            q_val_p1_vec = image[..., h-1, w_index_p1]
            q_l1dist_p1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_p1_vec)) ** 2
            delta_d = torch.sqrt(w_dist_pm1 + (lamda ** 2) * q_l1dist_p1_vec)
            new_dist.append(q_dis_p1_vec + delta_d)

            new_dist = torch.cat(new_dist, dim=0)
            min_dist, _ = torch.min(new_dist, dim=0)
            distance[..., h, :] = min_dist

        # bottom-up
        for h in reversed(range(height-1)):
            new_dist = []

            p_val_vec = image[..., h, :]
            p_dis_vec = distance[..., h, :]
            new_dist.append(p_dis_vec)

            # read row-1
            q_dis_z0_vec = distance[..., h+1, w_index_z0]
            q_val_z0_vec = image[..., h+1, w_index_z0]
            q_l1dist_z0_vec = torch.squeeze(torch.abs(p_val_vec-q_val_z0_vec))  ** 2
            delta_d = torch.sqrt(w_dist_z0 + (lamda ** 2) * q_l1dist_z0_vec)
            new_dist.append(q_dis_z0_vec + delta_d)

            q_dis_m1_vec = distance[..., h+1, w_index_m1]
            q_val_m1_vec = image[..., h+1, w_index_m1]
            q_l1dist_m1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_m1_vec))  ** 2
            delta_d = torch.sqrt(w_dist_pm1 + (lamda ** 2) * q_l1dist_m1_vec)
            new_dist.append(q_dis_m1_vec + delta_d)

            q_dis_p1_vec = distance[..., h+1, w_index_p1]
            q_val_p1_vec = image[..., h+1, w_index_p1]
            q_l1dist_p1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_p1_vec)) ** 2
            delta_d = torch.sqrt(w_dist_pm1 + (lamda ** 2) * q_l1dist_p1_vec)
            new_dist.append(q_dis_p1_vec + delta_d)

            new_dist = torch.cat(new_dist, dim=0)
            min_dist, _ = torch.min(new_dist, dim=0)
            distance[..., h, :] = min_dist

        # del local indices/distances as they are no longer needed
        del w_index_z0, w_index_p1, w_index_m1, w_dist_z0, w_dist_pm1

        # left-right
        h_index_z0 = [x for x in range(height)]
        h_index_m1 = [(x-1) if (x-1) > 0 else 0 for x in h_index_z0]
        h_index_p1 = [(x+1) if (x+1) < height else height-1 for x in h_index_z0]

        h_dist_z0 = 1
        h_dist_pm1 = 2

        for w in range(1, width):
            new_dist = []

            p_val_vec = image[..., :, w]
            p_dis_vec = distance[..., :, w]
            new_dist.append(p_dis_vec)

            # read col-1
            q_dis_z0_vec = distance[..., h_index_z0, w-1]
            q_val_z0_vec = image[..., h_index_z0, w-1]
            q_l1dist_z0_vec = torch.squeeze(torch.abs(p_val_vec-q_val_z0_vec))  ** 2
            delta_d = torch.sqrt(h_dist_z0 + (lamda ** 2) * q_l1dist_z0_vec)
            new_dist.append(q_dis_z0_vec + delta_d)

            q_dis_m1_vec = distance[..., h_index_m1, w-1]
            q_val_m1_vec = image[..., h_index_m1, w-1]
            q_l1dist_m1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_m1_vec))  ** 2
            delta_d = torch.sqrt(h_dist_pm1 + (lamda ** 2) * q_l1dist_m1_vec)
            new_dist.append(q_dis_m1_vec + delta_d)

            q_dis_p1_vec = distance[..., h_index_p1, w-1]
            q_val_p1_vec = image[..., h_index_p1, w-1]
            q_l1dist_p1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_p1_vec)) ** 2
            delta_d = torch.sqrt(h_dist_pm1 + (lamda ** 2) * q_l1dist_p1_vec)
            new_dist.append(q_dis_p1_vec + delta_d)

            new_dist = torch.cat(new_dist, dim=0)
            min_dist, _ = torch.min(new_dist, dim=0)
            distance[..., :, w] = min_dist

        # right-left
        h_index_z0 = [x for x in range(height)]
        h_index_m1 = [(x-1) if (x-1) > 0 else 0 for x in h_index_z0]
        h_index_p1 = [(x+1) if (x+1) < height else height-1 for x in h_index_z0]

        h_dist_z0 = 1
        h_dist_pm1 = 2

        for w in reversed(range(width-1)):
            new_dist = []

            p_val_vec = image[..., :, w]
            p_dis_vec = distance[..., :, w]
            new_dist.append(p_dis_vec)

            # read col-1
            q_dis_z0_vec = distance[..., h_index_z0, w+1]
            q_val_z0_vec = image[..., h_index_z0, w+1]
            q_l1dist_z0_vec = torch.squeeze(torch.abs(p_val_vec-q_val_z0_vec))  ** 2
            delta_d = torch.sqrt(h_dist_z0 + (lamda ** 2) * q_l1dist_z0_vec)
            new_dist.append(q_dis_z0_vec + delta_d)

            q_dis_m1_vec = distance[..., h_index_m1, w+1]
            q_val_m1_vec = image[..., h_index_m1, w+1]
            q_l1dist_m1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_m1_vec))  ** 2
            delta_d = torch.sqrt(h_dist_pm1 + (lamda ** 2) * q_l1dist_m1_vec)
            new_dist.append(q_dis_m1_vec + delta_d)

            q_dis_p1_vec = distance[..., h_index_p1, w+1]
            q_val_p1_vec = image[..., h_index_p1, w+1]
            q_l1dist_p1_vec = torch.squeeze(torch.abs(p_val_vec-q_val_p1_vec)) ** 2
            delta_d = torch.sqrt(h_dist_pm1 + (lamda ** 2) * q_l1dist_p1_vec)
            new_dist.append(q_dis_p1_vec + delta_d)

            new_dist = torch.cat(new_dist, dim=0)
            min_dist, _ = torch.min(new_dist, dim=0)
            distance[..., :, w] = min_dist

    return distance


@timing
def geodistk_geodesic_distance_2d(I, S, lamb, iter):
    return GeodisTK.geodesic2d_raster_scan(I, S, lamb, iter)


if __name__ == "__main__":
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

    dst1 = (
        generalised_geodesic2d_raster_scan(
            image=img.to('cuda'), mask=msk.to('cuda'), v=1e10, lamda=1.0, itr=2
        )
        .squeeze_()
        .detach()
        .cpu()
        .numpy()
    )
    dst2 = (
        generalised_geodesic2d_raster_scan_vectorised(
            image=img.to('cuda'), mask=msk.to('cuda'), v=1e10, lamda=1.0, itr=2
        )
        .squeeze_()
        .detach()
        .cpu()
        .numpy()
    )
    print(np.sum(np.abs(dst1-dst2)))
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(dst1))
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(dst2))
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(np.abs(dst1-dst2)))
    plt.show()

