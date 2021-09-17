#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: reubendo
"""
import dijkstra3d
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import os
import torch
import FastGeodis

def fastgeodis_generalised_geodesic_distance_3d(I, S, spacing, v, l_grad, l_eucl, iter):
    return FastGeodis.generalised_geodesic3d(I, S, spacing, v, l_grad, l_eucl, iter)
    
if __name__ == "__main__":
    # Reproducibility purposes
    np.random.seed(0)

    # run on GPU when available
    run_cuda = torch.cuda.is_available()


    connectivity = 26
    l_eucl = 1
    l_grad = 1
    spacing = [1, 0.5, 2]

    
    for N_seed in [1,10, 100, 1000]:
        times_dijkstra3d = []
        times_fastgeodis_cpu = []
        times_fastgeodis_gpu = []

        score_relative = []
        for shape in tqdm([(k,k,k) for k in range(10,210,10)]):

            # Creating image
            img = np.arange(np.prod(shape))
            np.random.shuffle(img)
            img = img.reshape(shape).astype(np.float32)
            

            N_seed = np.min([N_seed,np.prod(shape)-1])

            # Sources (extreme points along the x axis)
            sourcesind = np.random.choice(np.arange(np.prod(shape)), size=N_seed, replace=False)
            sources = np.unravel_index(sourcesind, shape)
            sources = np.stack(sources,1).tolist()

            #S = np.zeros_like(img).astype(np.float32)
            S = np.zeros_like(img, np.uint8)
            for source in sources:
                S[source[0]][source[1]][source[2]] = 1

            
            t = time.time()
            field_dijkstra3d = dijkstra3d.distance_field(
                data=img,
                prob=np.zeros_like(img),
                source=sources, 
                connectivity=connectivity, 
                spacing=spacing, 
                l_grad=l_grad, 
                l_eucl=l_eucl,
                l_prob=0.0)
            times_dijkstra3d.append(time.time() - t)

            
            It = torch.from_numpy(img).unsqueeze_(0).unsqueeze_(0).float().to("cpu")
            St = torch.from_numpy(1 - S.astype(np.float32)).unsqueeze_(0).unsqueeze_(0).to("cpu")
            
            t = time.time()
            D_raster_cpu = np.squeeze(
                fastgeodis_generalised_geodesic_distance_3d(
                    It, St, spacing, 1e10, l_grad, l_eucl, 4
                ).numpy()
            )
            times_fastgeodis_cpu.append(time.time() - t)

            if run_cuda:
                It = It.to("cuda")# torch.from_numpy(img).unsqueeze_(0).unsqueeze_(0).float().to("cpu")
                St = St.to("cuda")
                
                t = time.time()
                D_raster_gpu = np.squeeze(
                    fastgeodis_generalised_geodesic_distance_3d(
                        It, St, spacing, 1e10, l_grad, l_eucl, 4
                    ).cpu().numpy()
                )
                times_fastgeodis_gpu.append(time.time() - t)

                np.testing.assert_equal(D_raster_cpu, D_raster_gpu)

            score_relative.append((abs(field_dijkstra3d-D_raster_gpu)/(D_raster_gpu+1e-7)*100).mean())
            
        

                        
        list_x = [np.prod((k,k,k)) for k in range(10,210,10)]  

        fig,ax = plt.subplots()

        ax.set_title(f'Time in (s) per number of voxels - Seeds: {N_seed} random points')
        ax.set_xlabel('Spatial size')
        ax.set_ylabel('Execution time (seconds)')
        ax.plot(list_x, times_fastgeodis_cpu, marker='o', color='m',  label='FastGeodesic (cpu)')
        if run_cuda:
            ax.plot(list_x, times_fastgeodis_gpu, marker='o', color='g',  label='FastGeodesic (gpu)')
        ax.plot(list_x, times_dijkstra3d, marker='o', color='r', label='Djikstra3D')

        ax.legend()


        ax2=ax.twinx()
        ax2.scatter(list_x, score_relative, marker='s', color='b', edgecolors='none', label='Mean elative error')
        ax2.set_ylabel("Relative error (%)",color="blue")
        #ax2.legend()

        fig.tight_layout()
        #fig.show()
        fig.savefig(os.path.join('figures',  f'speed_{N_seed}.png'))
        