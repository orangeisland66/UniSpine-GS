
import os
import torch
from plyfile import PlyData

def count_gaussians(ply_path):
    plydata = PlyData.read(ply_path)
    return plydata.elements[0].count

def main():
    # Point this to any trained model result.
    ply_path = '/data1/sunchao/CQH/UniSpine-GS_new_all_ct_v2/output/chest/2026_02_09_16_55_01/point_cloud/iteration_20000/point_cloud.ply'
    
    if not os.path.exists(ply_path):
        print(f"File not found: {ply_path}")
        return

    n_gaussians = count_gaussians(ply_path)
    
    # Number of attributes per Gaussian.
    # xyz: 3
    # features_dc: 3 (1*3)
    # features_rest: 45 (15*3, max_sh_degree=3)
    # scaling: 3
    # rotation: 4
    # opacity: 1
    # Total per gaussian: 3 + 3 + 45 + 3 + 4 + 1 = 59
    
    params_per_gaussian = 59
    total_params = n_gaussians * params_per_gaussian
    
    print(f"{'Model':<25} | {'Gaussians':<15} | {'Parameters':<15}")
    print("-" * 59)
    print(f"{'UniSpine-GS (chest)':<25} | {n_gaussians:<15,} | {total_params:<15,}")

if __name__ == "__main__":
    main()
