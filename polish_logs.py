#!/usr/bin/env python3
"""
Video Processing Script for Robot Inference Logs

This script processes saved images from robot inference runs and converts them into videos.
It supports video image recordings directory structures:
video_recordings/
└── task_name/
    └── iter_X/
        ├── images/
        │   ├── 000001_head.jpg
        │   ├── 000001_left_wrist.jpg
        │   ├── 000001_right_wrist.jpg
        │   ├── 000001_combined.jpg
        │   └── ...
        └── task_name_inference_TIMESTAMP.log

Usage:
    python3 -m pip install imageio[ffmpeg]
    python polish_logs.py [--base-dirs PATH] [--overwrite] [--help]

Examples:
    python polish_logs.py  # Process default directories
    python polish_logs.py --base-dirs ./custom_recordings --overwrite
"""
import os

class Config:
    DEFAULT_VIDEO_RECORDINGS_DIR = "AgiBot-World-Submission/CogACT/video_recordings"
    DEFAULT_FPS = 2
    VIDEO_OUTPUT_FORMAT = 'mp4'

def merge_images_to_video(task_log_dir, task_name):
    """Merge all images in the directory into one final video

    """
    try:
        import imageio
        print(f" Merging images into video...")
        
        # Create final video writer with same fps as segments
        final_video_path = os.path.join(task_log_dir, f"{task_name}_inference_complete.{Config.VIDEO_OUTPUT_FORMAT}")
        final_writer = imageio.get_writer(final_video_path, fps=Config.DEFAULT_FPS)
        
        # Read and merge all images
        images = sorted([img for img in os.listdir(task_log_dir) if "combined" in img and img.endswith(".jpg")])
        for img_name in images:
            img_path = os.path.join(task_log_dir, img_name)
            print(f"   Processing image {img_name}...")
            image = imageio.imread(img_path)
            final_writer.append_data(image)
        
        final_writer.close()
        print(f"✅ Images merged successfully: {final_video_path}")
        
    except Exception as e:
        print(f"❌ Error merging images: {e}")

def iter_root_folder_to_get_dirs(video_recording_base_dir):
    # video_recording_base_dir = "AgiBot-World-Submission/CogACT/video_recordings"
    from pathlib import Path
    # list the folder named "iter_x", in level deep 2
    root = Path(video_recording_base_dir)
    # dirs = [f for f in root.iterdir() if f.is_dir() and f.name.startswith("iter_")]
    dirs = [f for f in root.glob("*/iter_*") if f.is_dir()]
    return dirs

def get_task_name_info_from_dirname(dirname):
    # task_name is parent folder name of "iter_x"
    task_name = dirname.split("/")[-2]
    task_log_dir = dirname
    return task_name, task_log_dir

def get_image_num(task_log_dir):
    # count number of files contains "inference_segment_*.mp4"
    files = os.listdir(task_log_dir)
    # print(f"Files in {task_log_dir}: {files}")
    segment_files = [f for f in files if "combined" in f and f.endswith(".jpg")]
    
    return len(segment_files)

def has_final_video(task_log_dir, task_name):
    final_video_path = os.path.join(task_log_dir, f"{task_name}_inference_complete.mp4")
    return os.path.exists(final_video_path)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Process robot inference logs to merge images into videos.")
    parser.add_argument("--base-dirs", type=str, nargs='*', default=[Config.DEFAULT_VIDEO_RECORDINGS_DIR],
                        help="Base directories to search for task logs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing final videos if they exist.")
    args = parser.parse_args()

    # If we have multiple segments, merge them
    dirs = iter_root_folder_to_get_dirs(args.base_dirs[0])
    # print(f"Found {len(dirs)} task directories to process.")
    for dir in dirs:
        task_name, task_log_dir = get_task_name_info_from_dirname(str(dir))
        total_segments = get_image_num(task_log_dir)
        # print(f"Found {total_segments} combined images in {task_log_dir}")
        if total_segments > 0:
            if has_final_video(task_log_dir, task_name) and not args.overwrite:
                print(f"Final video already exists for {task_name} in {task_log_dir}, skipping merge.")
                continue
            merge_images_to_video(task_log_dir, task_name)
        else:
            print(f"⚠️ No images found in {task_log_dir}, skipping merge.")
