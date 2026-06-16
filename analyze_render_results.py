
import os
import numpy as np
from PIL import Image
import glob

def analyze_images(base_path):
    gt_path = os.path.join(base_path, "gt")
    renders_path = os.path.join(base_path, "renders")
    
    gt_files = sorted(glob.glob(os.path.join(gt_path, "*.png")))
    render_files = sorted(glob.glob(os.path.join(renders_path, "*.png")))
    
    if not gt_files or not render_files:
        print(f"No images found in {base_path}")
        return

    print(f"Analyzing {len(gt_files)} images in {base_path}...")
    
    total_l1 = 0
    total_mse = 0
    gt_mean_intensity = 0
    render_mean_intensity = 0
    non_zero_pixels = 0
    total_pixels = 0
    
    for gt_f, render_f in zip(gt_files, render_files):
        try:
            gt_img = np.array(Image.open(gt_f)).astype(np.float32) / 255.0
            render_img = np.array(Image.open(render_f)).astype(np.float32) / 255.0
            
            # Simple aggregations
            gt_mean_intensity += np.mean(gt_img)
            render_mean_intensity += np.mean(render_img)
            
            diff = np.abs(gt_img - render_img)
            total_l1 += np.mean(diff)
            total_mse += np.mean(diff ** 2)
            
            # Check how much of the image is "empty" (black background)
            # Ultrasound images often have a lot of black.
            # Let's say pixel < 0.05 is "black"
            mask = gt_img > 0.05
            non_zero_pixels += np.sum(mask)
            total_pixels += gt_img.size
            
        except Exception as e:
            print(f"Error processing {gt_f}: {e}")

    num_images = len(gt_files)
    avg_gt_mean = gt_mean_intensity / num_images
    avg_render_mean = render_mean_intensity / num_images
    avg_l1 = total_l1 / num_images
    avg_psnr = -10 * np.log10(total_mse / num_images)
    avg_foreground_ratio = non_zero_pixels / total_pixels

    print("-" * 30)
    print(f"Results for {base_path}:")
    print(f"  Average GT Intensity: {avg_gt_mean:.4f}")
    print(f"  Average Render Intensity: {avg_render_mean:.4f}")
    print(f"  Average L1 Error: {avg_l1:.4f}")
    print(f"  Estimated PSNR: {avg_psnr:.2f} dB")
    print(f"  Foreground Ratio (pixels > 0.05): {avg_foreground_ratio:.2%}")
    print("-" * 30)

    if avg_foreground_ratio < 0.01:
        print("WARNING: Very low foreground ratio. The images might be mostly black.")
    elif avg_foreground_ratio > 0.9:
        print("WARNING: Very high foreground ratio. The images might be full.")
    
    # Check specific sample
    sample_idx = len(gt_files) // 2
    gt_sample = np.array(Image.open(gt_files[sample_idx])).astype(np.float32) / 255.0
    render_sample = np.array(Image.open(render_files[sample_idx])).astype(np.float32) / 255.0
    
    print(f"Sample Image ({os.path.basename(gt_files[sample_idx])}) Stats:")
    print(f"  GT Range: [{gt_sample.min():.4f}, {gt_sample.max():.4f}]")
    print(f"  Render Range: [{render_sample.min():.4f}, {render_sample.max():.4f}]")

def analyze_ply(ply_path):
    if not os.path.exists(ply_path):
        print(f"PLY file not found: {ply_path}")
        return
        
    print(f"Analyzing Point Cloud: {ply_path}")
    # Simple PLY header parser
    num_vertex = 0
    with open(ply_path, 'rb') as f:
        header_ended = False
        while not header_ended:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('element vertex'):
                num_vertex = int(line.split()[-1])
            if line == 'end_header':
                header_ended = True
    
    print(f"  Number of vertices: {num_vertex}")
    if num_vertex < 1000:
        print("WARNING: Very few points. Reconstruction likely failed.")
    
    analyze_ply_content(ply_path) # Call detailed analysis

def analyze_ply_content(ply_path):
    try:
        import plyfile
        plydata = plyfile.PlyData.read(ply_path)
        vertex = plydata['vertex']
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        
        print(f"Detailed Point Cloud Analysis:")
        print(f"  X range: [{x.min():.4f}, {x.max():.4f}] (span: {x.max()-x.min():.4f})")
        print(f"  Y range: [{y.min():.4f}, {y.max():.4f}] (span: {y.max()-y.min():.4f})")
        print(f"  Z range: [{z.min():.4f}, {z.max():.4f}] (span: {z.max()-z.min():.4f})")
        
        in_unit_cube = np.sum((x >= -5) & (x <= 5) & (y >= -5) & (y <= 5) & (z >= -5) & (z <= 5))
        print(f"  Points in [-5, 5]^3: {in_unit_cube} ({in_unit_cube/len(x):.2%})")

    except ImportError:
        print("plyfile module not installed, skipping detailed analysis")
    except Exception as e:
        print(f"Error reading PLY content: {e}")

def analyze_pickle(pickle_path):
    try:
        import pickle
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # print("Keys:", data.keys())
        if "image" in data:
            img = data["image"]
            print(f"  Volume Shape: {img.shape}")
            print(f"  Volume Range: [{img.min():.4f}, {img.max():.4f}]")
            print(f"  Volume Mean: {np.mean(img):.4f}")
            
        # Check projection ranges
        if "train" in data and "projections" in data["train"]:
            projs_train = data["train"]["projections"]
            print(f"  Train Projections Shape: {projs_train.shape}")
            print(f"  Train Projections Range: [{projs_train.min():.4f}, {projs_train.max():.4f}]")
            print(f"  Train Projections Mean: {np.mean(projs_train):.4f}")

        if "val" in data and "projections" in data["val"]:
            projs_val = data["val"]["projections"]
            print(f"  Val Projections Shape: {projs_val.shape}")
            print(f"  Val Projections Range: [{projs_val.min():.4f}, {projs_val.max():.4f}]")
            print(f"  Val Projections Mean: {np.mean(projs_val):.4f}")

    except Exception as e:
        print(f"Error analyzing pickle: {e}")

if __name__ == "__main__":
    base_dir = "/data1/sunchao/CQH/UniSpine-GS/output/IMG_20251125_1"
    pickle_path = "/data1/sunchao/CQH/UniSpine-GS/data/ultrasound/IMG_20251125_1.pickle"
    
    print("=== Analyzing Test Set ===")
    analyze_images(os.path.join(base_dir, "test/ours_20000"))
    
    print("\n=== Analyzing Train Set ===")
    analyze_images(os.path.join(base_dir, "train/ours_20000"))
    
    print("\n=== Analyzing Point Cloud ===")
    analyze_ply(os.path.join(base_dir, "point_cloud/iteration_20000/point_cloud.ply"))
    
    print("\n=== Analyzing Pickle Data ===")
    analyze_pickle(pickle_path)
