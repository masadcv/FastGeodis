# FastGeodis: Fast Generalised Geodesic Distance Transform


## Installation instructions

`pip install git+https://github.com/masadcv/FastGeodis`

TODO:

`pip install FastGeodis`

## Example usage

### Fast Geodesic Distance Transform
The following demonstrates a simple example showing FastGeodis usage:
```
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
```

For more usage examples see:
- **Simple 2D Geodesic Distance example**: [`samples/simpledemo2d.py`](./samples/simpledemo2d.py) 
- **2D Geodesic Distance**: [`samples/demo2d.py`](./samples/demo2d.py) 
- **3D Geodesic Distance**: [`samples/demo3d.py`](./samples/demo3d.py)
- **2D GSF Segmentation Smoothing**: [`samples/demoGSF2d_SmoothingSegExample.ipynb`](./samples/demoGSF2d_SmoothingSegExample.ipynb)
 

## Execution Time
<img src="figures/experiment_2d.png?raw=true" width="400" />
<img src="figures/experiment_3d.png?raw=true" width="400" />

<br><br>

## Initial Results Comparison
### 2D case
<img src="figures/fast_marching_compare_2d.png?raw=true?raw=true" width="800" />

<img src="figures/fast_marching_compare_3d_jointhist.png?raw=true" width="400" />

### 3D case

<img src="figures/fast_marching_compare_3d.png?raw=true" width="800" />

<img src="figures/fast_marching_compare_2d_jointhist.png?raw=true" width="400" />

<br><br>
## Citation
TODO