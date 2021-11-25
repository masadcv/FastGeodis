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
lamb = 1.0
iterations = 2
geodesic_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, 1 - lamb, iterations
)
geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.imshow(geodesic_dist)
plt.plot(100, 100, "mo")
plt.show()
