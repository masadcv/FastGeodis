import GeodisTK
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import wraps
from time import time
import SimpleITK as sitk
import geodis

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
def generalised_geodesic_distance_3d(I, S, spacing, v, lamb, iter):
    """
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    """
    return GeodisTK.generalised_geodesic3d_raster_scan(I, S, spacing, v, lamb, iter)

@timing
def generalised_geodesic3d_raster_2scan(image, mask, spacing, v, lamda, iter):
    batch, _, depth, height, width = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()

    spacing_squared = [a * a for a in spacing]

    use_speed = True
    for it in range(iter):
        d_td = [
            (-1, -1, -1),
            (-1, -1, 0),
            (-1, -1, 1),
            (-1, 0, -1),
            (-1, 0, 0),
            (0, -1, -1),
            (0, -1, 0),
            (0, -1, 1),
            (0, 0, -1),
            (1, -1, -1),
            (1, -1, 0),
            (1, -1, 1),
            (1, 0, -1),
        ]
        local_dis_td = [
            math.sqrt(
                abs(a) * spacing_squared[0]
                + abs(b) * spacing_squared[1]
                + abs(c) * spacing_squared[2],
            )
            for a, b, c in d_td
        ]

        # forward scan
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    p_val = image[..., d, h, w]

                    c_d_td = [(d + dd, h + dh, w + dw) for dd, dh, dw in d_td]
                    for idx, (d_d_td, h_d_td, w_d_td) in enumerate(c_d_td):
                        if (
                            d_d_td < 0
                            or d_d_td >= depth
                            or h_d_td < 0
                            or h_d_td >= height
                            or w_d_td < 0
                            or w_d_td >= width
                        ):
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
        d_td = [
            (-1, 0, 1),
            (-1, 1, -1),
            (-1, -1, 0),
            (-1, 1, 1),
            (0, 0, 1),
            (0, 1, -1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, -1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        local_dis_td = [
            math.sqrt(
                abs(a) * spacing_squared[0]
                + abs(b) * spacing_squared[1]
                + abs(c) * spacing_squared[2],
            )
            for a, b, c in d_td
        ]

        for d in reversed(range(depth)):
            for h in reversed(range(height)):
                for w in reversed(range(width)):
                    p_val = image[..., d, h, w]

                    c_d_td = [(d + dd, h + dh, w + dw) for dd, dh, dw in d_td]
                    for idx, (d_d_td, h_d_td, w_d_td) in enumerate(c_d_td):
                        if (
                            d_d_td < 0
                            or d_d_td >= depth
                            or h_d_td < 0
                            or h_d_td >= height
                            or w_d_td < 0
                            or w_d_td >= width
                        ):
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


def geodesic_frontback_pass(image, distance, spacing, lamda):
    _, _, depth, height, width = image.shape

    w_index_z0 = [x for x in range(width)]
    w_index_m1 = [(x - 1) if (x - 1) > 0 else 0 for x in w_index_z0]
    w_index_p1 = [(x + 1) if (x + 1) < width else width - 1 for x in w_index_z0]

    h_index_z0 = [x for x in range(height)]
    h_index_m1 = [(x - 1) if (x - 1) > 0 else 0 for x in h_index_z0]
    h_index_p1 = [(x + 1) if (x + 1) < height else height - 1 for x in h_index_z0]

    # back-front
    for z in range(1, depth):
        p_val_vec = image[..., z, :, :]
        for hi, h_index in enumerate([h_index_m1, h_index_z0, h_index_p1]):
            for wi, w_index in enumerate([w_index_m1, w_index_z0, w_index_p1]):
                local_dist = spacing[0] * spacing[0]
                local_dist += ((float(hi) - 1.0) ** 2) * spacing[1] * spacing[1]
                local_dist += ((float(wi) - 1.0) ** 2) * spacing[2] * spacing[2]

                q_dis_vec = distance[..., z - 1, h_index, :][..., w_index]
                q_val_vec = image[..., z - 1, h_index, :][..., w_index]
                if 1 == 1:
                    l2dist = get_l1_distance(p_val_vec, q_val_vec, 1)
                    speed = (1.0 - lamda) + lamda / (l2dist + 1e-5)
                    delta_d = math.sqrt(local_dist) / speed
                else:
                    q_l1dist_z0_vec = get_l1_distance(p_val_vec, q_val_vec, 1) ** 2
                    delta_d = torch.sqrt(local_dist + (lamda ** 2) * q_l1dist_z0_vec)

                distance[..., z, :, :] = torch.minimum(distance[..., z, :, :], (q_dis_vec + delta_d))

    # front-back
    for z in reversed(range(depth - 1)):
        p_val_vec = image[..., z, :, :]
        for hi, h_index in enumerate([h_index_m1, h_index_z0, h_index_p1]):
            for wi, w_index in enumerate([w_index_m1, w_index_z0, w_index_p1]):
                local_dist = spacing[0] * spacing[0]
                local_dist += ((float(hi) - 1.0) ** 2) * spacing[1] * spacing[1]
                local_dist += ((float(wi) - 1.0) ** 2) * spacing[2] * spacing[2]

                q_dis_vec = distance[..., z + 1, h_index, :][..., w_index]
                q_val_vec = image[..., z + 1, h_index, :][..., w_index]
                if 1 == 1:
                    l2dist = get_l1_distance(p_val_vec, q_val_vec, 1)
                    speed = (1.0 - lamda) + lamda / (l2dist + 1e-5)
                    delta_d = math.sqrt(local_dist) / speed
                else:
                    q_l1dist_z0_vec = get_l1_distance(p_val_vec, q_val_vec, 1) ** 2
                    delta_d = torch.sqrt(local_dist + (lamda ** 2) * q_l1dist_z0_vec)
                distance[..., z, :, :] = torch.minimum(distance[..., z, :, :], (q_dis_vec + delta_d))

    return distance


@timing
def generalised_geodesic3d_raster_4scan_vectorised(image, mask, spacing, v, lamda, iter):
    batch, _, _, _, _ = image.shape
    assert batch == 1, "At the moment, only batch=1 supported - received {}".format(
        batch
    )

    # initialise distance with soft mask
    distance = v * mask.clone()

    for it in range(iter):
        # front-back - depth*, height, width
        distance = geodesic_frontback_pass(image, distance, spacing, lamda)

        # top-bottom - height*, depth, width
        image = torch.transpose(image, dim0=3, dim1=2)
        distance = torch.transpose(distance, dim0=3, dim1=2)
        distance = geodesic_frontback_pass(image, distance, spacing, lamda)
        # transpose back to original depth, height, width
        image = torch.transpose(image, dim0=3, dim1=2)
        distance = torch.transpose(distance, dim0=3, dim1=2)

        # left-right -  width*, height, depth
        image = torch.transpose(image, dim0=4, dim1=2)
        distance = torch.transpose(distance, dim0=4, dim1=2)
        distance = geodesic_frontback_pass(image, distance, spacing, lamda)
        # transpose back to original depth, height, width
        image = torch.transpose(image, dim0=4, dim1=2)
        distance = torch.transpose(distance, dim0=4, dim1=2)

        # * indicates the current direction of pass

    return distance

# cpp implementation
@timing
def generalised_geodesic3d_raster_4scan_cpp(image, mask, spacing, v, lamda, iter):
    return geodis.generalised_geodesic3d(image, mask, spacing, v, lamda, iter)


def test_compare_original_imp():
    input_name = "data/img3d.nii.gz"
    img_sitk = sitk.ReadImage(input_name)
    img_np = sitk.GetArrayFromImage(img_sitk)
    spacing_raw = img_sitk.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    img_np = np.asarray(img_np, np.float32)
    # img_np = img_np[18:38, 63:183, 93:233]
    msk_np = np.zeros_like(img_np, np.float32)
    msk_np[10][60][70] = 1.0
    msk_np = 1.0 - msk_np

    img = np.expand_dims(np.expand_dims(img_np, axis=0), axis=0)
    msk = np.expand_dims(np.expand_dims(msk_np, axis=0), axis=0)

    img = torch.from_numpy(img.astype(np.float32))
    msk = torch.from_numpy(msk.astype(np.float32))

    img_np = np.squeeze(img.detach().cpu().numpy())
    msk_np = np.squeeze(msk.detach().cpu().numpy())

    dst1 = generalised_geodesic_distance_3d(
        img_np, msk_np, spacing, v=1e10, lamb=1.0, iter=2
    )

    dst1 = (
        generalised_geodesic3d_raster_4scan_cpp(
            image=img.to("cpu"),
            mask=msk.to("cpu"),
            spacing=spacing,
            v=1e10,
            lamda=1.0,
            iter=2,
        )
        .squeeze_()
        .detach()
        .cpu()
        .numpy()
    )

    dst2 = (
        generalised_geodesic3d_raster_4scan_vectorised(
            image=img.to("cpu"),
            mask=msk.to("cpu"),
            spacing=spacing,
            v=1e10,
            lamda=1.0,
            iter=2,
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

    slice_idx = 1
    img_slice = img_np[slice_idx]
    dst1 = dst1[slice_idx]
    dst2 = dst2[slice_idx]

    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(dst1))
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(dst2))
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(np.abs(dst1 - dst2)))
    plt.show()

    assert diff < 1


if __name__ == "__main__":
    test_compare_original_imp()
