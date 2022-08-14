#!/usr/bin/env /usr/bin/python3
import numpy as np
import torch
from PIL import Image

import FastGeodis

device = "cuda" if torch.cuda.is_available() else "cpu"
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

print(geodesic_dist)
