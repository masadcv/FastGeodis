import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import FastGeodis

device = "cuda" if torch.cuda.is_available else "cpu"
image = np.asarray(Image.open("data/img2d.png"), np.float32)

image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
image_pt = image_pt.to(device)
mask_pt = torch.ones_like(image_pt)
mask_pt[..., 100, 100] = 0

v = 1e10
iterations = 2

lamb = 1.0 # <-- Geodesic distance transform
geodesic_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, iterations
)
geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())

lamb=0.0 # <-- Euclidean distance transform
euclidean_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, iterations
)
euclidean_dist = np.squeeze(euclidean_dist.cpu().numpy())

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.imshow(geodesic_dist)
plt.plot(100, 100, "mo")

plt.subplot(1, 3, 3)
plt.imshow(euclidean_dist)
plt.plot(100, 100, "mo")

plt.show()
