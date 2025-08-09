import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Stack attention map.")
    parser.add_argument("--save-dir", type=str, default="Data/Synthetic", help="Directory to save images (and heatmaps if enabled).")
    parser.add_argument("--object-heatmap-path", type=str, default=None, help="Path to the object token heatmaps.")
    parser.add_argument("--fg-heatmap-path", type=str, default=None, help="Path to the foreground learnable token heatmaps.")
    parser.add_argument("--bg-heatmap-path", type=str, default=None, help="Path to the background learnable token heatmaps.")
    parser.add_argument("--stack-heatmap-save-path", type=str, default="daam_stack_heatmaps", help="Path to save the stacked heatmaps.")
    parser.add_argument("--inv-heatmap-save-path", type=str, default="daam_inv_heatmaps", help="Path to save the inverted heatmaps of the learnable background token.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    object_heatmap_path = os.path.join(args.save_dir,args.object_heatmap_path)
    fg_heatmap_path =  os.path.join(args.save_dir,args.fg_heatmap_path)
    bg_heatmap_path =  os.path.join(args.save_dir,args.bg_heatmap_path)
    stack_heatmap_save_path = os.path.join(args.save_dir,args.stack_heatmap_save_path)
    inv_heatmap_save_path = os.path.join(args.save_dir,args.inv_heatmap_save_path)

    os.makedirs(stack_heatmap_save_path, exist_ok=True)
    os.makedirs(inv_heatmap_save_path, exist_ok=True)

    object_heatmap_files = os.listdir(object_heatmap_path)
    fg_heatmap_files = os.listdir(fg_heatmap_path)
    bg_heatmap_files = os.listdir(bg_heatmap_path)

    for object_heatmap_file, fg_heatmap_file, bg_heatmap_file in tqdm(zip(object_heatmap_files, fg_heatmap_files, bg_heatmap_files)):
        obj_heatmap = Image.open(os.path.join(object_heatmap_path, object_heatmap_file))
        obj_heatmap_arr = np.asarray(obj_heatmap)
        fg_heatmap = Image.open(os.path.join(fg_heatmap_path, fg_heatmap_file))
        fg_heatmap_arr = np.asarray(fg_heatmap)
        bg_heatmap = Image.open(os.path.join(bg_heatmap_path, bg_heatmap_file))
        bg_heatmap_arr = np.asarray(bg_heatmap)

        inv_bg_heatmap_arr = 255 - bg_heatmap_arr

        stack_heatmap_arr = np.stack([obj_heatmap_arr, fg_heatmap_arr, inv_bg_heatmap_arr], axis=-1)
        stack_heatmap = Image.fromarray(stack_heatmap_arr)
        stack_heatmap.save(os.path.join(stack_heatmap_save_path, object_heatmap_file))
        inv_bg_heatmap = Image.fromarray(inv_bg_heatmap_arr)
        inv_bg_heatmap.save(os.path.join(inv_heatmap_save_path, bg_heatmap_file))


