import torch
import geodis
import time

ab = torch.rand((1, 1, 512, 512))

N = 5
tic = time.time()
for i in range(N):
    geodis.generalised_geodesic2d(ab, ab, 0.1, 0.1, 10)

print('took: {}'.format((time.time() - tic)/N))