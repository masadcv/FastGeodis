from functools import partial, wraps
import unittest
import torch

try:
    import FastGeodis
except:
    print(
        "Unable to load FastGeodis for unittests\nMake sure to install using: python setup.py install"
    )
    exit()

DEVICES_TO_RUN = ["cpu", "cuda"]
CONF_2D_CPU = [("cpu", 2, bas) for bas in [16, 32, 64]]
CONF_2D_CUDA = [("cuda", 2, bas) for bas in [16, 32, 64]]
CONF_2D = CONF_2D_CPU + CONF_2D_CUDA

CONF_3D_CPU = [("cpu", 3, bas) for bas in [16, 32, 64]]
CONF_3D_CUDA = [("cuda", 3, bas) for bas in [16, 32, 64]]
CONF_3D = CONF_3D_CPU + CONF_3D_CUDA
CONF_ALL = CONF_2D + CONF_3D

CONF_ALL_CPU = CONF_2D_CPU + CONF_3D_CPU

# Fast March is compute intensive, so limit 2d and 3d cases to run with it
CONF_2D_CPU_FM = [CONF_2D_CPU[0]]
CONF_3D_CPU_FM = [CONF_3D_CPU[0]]
CONF_ALL_CPU_FM = CONF_2D_CPU_FM + CONF_3D_CPU_FM



def skip_if_no_cuda(obj):
    return unittest.skipUnless(torch.cuda.is_available(), "Skipping CUDA-based tests")(
        obj
    )


def run_cuda_if_available(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if args[1] == "cuda":
            if torch.cuda.is_available():
                return fn(*args, **kwargs)
            else:
                raise unittest.SkipTest("skipping as cuda device not found")
        else:
            return fn(*args, **kwargs)

    return wrapper


def fastgeodis_generalised_geodesic_distance_2d(image, softmask, v, lamb, iter):
    return FastGeodis.generalised_geodesic2d(image, softmask, v, lamb, iter)


def fastgeodis_generalised_geodesic_distance_3d(
    image, softmask, v, lamb, iter, spacing
):
    return FastGeodis.generalised_geodesic3d(image, softmask, spacing, v, lamb, iter)


def fastgeodis_signed_generalised_geodesic_distance_2d(image, softmask, v, lamb, iter):
    return FastGeodis.signed_generalised_geodesic2d(image, softmask, v, lamb, iter)


def fastgeodis_signed_generalised_geodesic_distance_3d(
    image, softmask, v, lamb, iter, spacing
):
    return FastGeodis.signed_generalised_geodesic3d(
        image, softmask, spacing, v, lamb, iter
    )


def toivanen_signed_generalised_geodesic_distance_2d(image, softmask, v, lamb, iter):
    return FastGeodis.signed_generalised_geodesic2d_toivanen(
        image, softmask, v, lamb, iter
    )


def toivanen_signed_generalised_geodesic_distance_3d(
    image, softmask, v, lamb, iter, spacing
):
    return FastGeodis.signed_generalised_geodesic3d_toivanen(
        image, softmask, spacing, v, lamb, iter
    )


def pixelqueue_signed_generalised_geodesic_distance_2d(image, softmask, lamb, iter):
    return FastGeodis.signed_geodesic2d_pixelqueue(image, softmask, lamb)


def pixelqueue_signed_generalised_geodesic_distance_3d(
    image, softmask, lamb, iter, spacing
):
    return FastGeodis.signed_geodesic3d_pixelqueue(
        image, softmask, spacing, lamb
    )

def fastmarch_signed_generalised_geodesic_distance_2d(image, softmask, lamb, iter):
    return FastGeodis.signed_geodesic2d_fastmarch(image, softmask, lamb)


def fastmarch_signed_generalised_geodesic_distance_3d(
    image, softmask, lamb, iter, spacing
):
    return FastGeodis.signed_geodesic3d_fastmarch(
        image, softmask, spacing, lamb
    )

def toivanen_generalised_geodesic_distance_2d(image, softmask, v, lamb, iter):
    return FastGeodis.generalised_geodesic2d_toivanen(image, softmask, v, lamb, iter)


def toivanen_generalised_geodesic_distance_3d(image, softmask, v, lamb, iter, spacing):
    return FastGeodis.generalised_geodesic3d_toivanen(
        image, softmask, spacing, v, lamb, iter
    )


def pixelqueue_geodesic_distance_2d(image, softmask, lamb, iter):
    return FastGeodis.geodesic2d_pixelqueue(image, softmask, lamb)


def pixelqueue_geodesic_distance_3d(image, softmask, lamb, iter, spacing):
    return FastGeodis.geodesic3d_pixelqueue(
        image, softmask, spacing, lamb
    )

def fastmarch_geodesic_distance_2d(image, softmask, lamb, iter):
    return FastGeodis.geodesic2d_fastmarch(image, softmask, lamb)


def fastmarch_geodesic_distance_3d(image, softmask, lamb, iter, spacing):
    return FastGeodis.geodesic3d_fastmarch(
        image, softmask, spacing, lamb
    )


def fastgeodis_GSF_2d(image, softmask, theta, v, lamb, iter):
    return FastGeodis.GSF2d(image, softmask, theta, v, lamb, iter)


def fastgeodis_GSF_3d(image, softmask, theta, v, lamb, iter, spacing):
    return FastGeodis.GSF3d(image, softmask, theta, spacing, v, lamb, iter)


def toivanen_GSF_2d(image, softmask, theta, v, lamb, iter):
    return FastGeodis.GSF2d_toivanen(image, softmask, theta, v, lamb, iter)


def toivanen_GSF_3d(image, softmask, theta, v, lamb, iter, spacing):
    return FastGeodis.GSF3d_toivanen(image, softmask, theta, spacing, v, lamb, iter)


def pixelqueue_GSF_2d(image, softmask, theta, lamb, iter):
    return FastGeodis.GSF2d_pixelqueue(image, softmask, theta, lamb)


def pixelqueue_GSF_3d(image, softmask, theta, lamb, iter, spacing):
    return FastGeodis.GSF3d_pixelqueue(image, softmask, theta, spacing, lamb)

def fastmarch_GSF_2d(image, softmask, theta, lamb, iter):
    return FastGeodis.GSF2d_fastmarch(image, softmask, theta, lamb)


def fastmarch_GSF_3d(image, softmask, theta, lamb, iter, spacing):
    return FastGeodis.GSF3d_fastmarch(image, softmask, theta, spacing, lamb)


def get_simple_shape(base_dim, num_dims):
    return [1, 1] + [
        base_dim,
    ] * num_dims


def get_fastgeodis_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastgeodis_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(fastgeodis_generalised_geodesic_distance_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_signed_fastgeodis_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastgeodis_signed_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(
            fastgeodis_signed_generalised_geodesic_distance_3d, spacing=spacing
        )
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_toivanen_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return toivanen_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(toivanen_generalised_geodesic_distance_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_signed_toivanen_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return toivanen_signed_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(
            toivanen_signed_generalised_geodesic_distance_3d, spacing=spacing
        )
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_pixelqueue_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return pixelqueue_geodesic_distance_2d
    elif num_dims == 3:
        return partial(pixelqueue_geodesic_distance_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_signed_pixelqueue_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return pixelqueue_signed_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(
            pixelqueue_signed_generalised_geodesic_distance_3d, spacing=spacing
        )
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))

def get_fastmarch_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastmarch_geodesic_distance_2d
    elif num_dims == 3:
        return partial(fastmarch_geodesic_distance_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_signed_fastmarch_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastmarch_signed_generalised_geodesic_distance_2d
    elif num_dims == 3:
        return partial(
            fastmarch_signed_generalised_geodesic_distance_3d, spacing=spacing
        )
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_GSF_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastgeodis_GSF_2d
    elif num_dims == 3:
        return partial(fastgeodis_GSF_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_GSF_toivanen_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return toivanen_GSF_2d
    elif num_dims == 3:
        return partial(toivanen_GSF_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))


def get_GSF_pixelqueue_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return pixelqueue_GSF_2d
    elif num_dims == 3:
        return partial(pixelqueue_GSF_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))



def get_GSF_fastmarch_func(num_dims, spacing=[1.0, 1.0, 1.0]):
    if num_dims == 2:
        return fastmarch_GSF_2d
    elif num_dims == 3:
        return partial(fastmarch_GSF_3d, spacing=spacing)
    else:
        raise ValueError("Unsupported num_dims received: {}".format(num_dims))

