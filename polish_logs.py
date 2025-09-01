#!/usr/bin/env python3
"""
Video Processing Script for Robot Inference Logs

This script processes saved images from robot inference runs and converts them into videos.
It also provides functionality to clean up image files after video generation.

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
    python polish_logs.py [--base-dirs PATH] [--overwrite] [--cleanup-images] [--dry-run] [--help]

Examples:
    python polish_logs.py  # Process default directories
    python polish_logs.py --base-dirs ./custom_recordings --overwrite
    python polish_logs.py --cleanup-images --dry-run  # Show what images would be deleted
    python polish_logs.py --cleanup-images  # Delete all images after processing, keep only MP4s


    # See what images would be deleted (dry run)
    python polish_logs.py --cleanup-images --dry-run

    # Process videos and delete all images afterward
    python polish_logs.py --cleanup-images

    # Process with custom directory and cleanup
    python polish_logs.py --base-dir ./custom_recordings --cleanup-images --overwrite
"""
import os

class Config:
    DEFAULT_VIDEO_RECORDINGS_DIR = "AgiBot-World-Submission/CogACT/video_recordings"
    VIDEO_OUTPUT_FORMAT = 'mp4'
    DEFAULT_FPS = 20
    SAVE_MODE = 'per_joint_step'  # Options: 'per_inference', 'per_joint_step'
    # DEFAULT_FPS = 2
    # SAVE_MODE = 'per_inference'  # Options: 'per_inference', 'per_joint_step'

def merge_images_to_video(task_log_dir, task_name):
    """Merge all images in the directory into one final video

    """
    try:
        import imageio
        print(f" Merging images into video...")
        
        # Create final video writer with same fps as segments
        if Config.SAVE_MODE == "per_inference":
            final_video_path = os.path.join(task_log_dir, f"{task_name}_inference_complete.{Config.VIDEO_OUTPUT_FORMAT}")
        elif Config.SAVE_MODE == "per_joint_step":
            final_video_path = os.path.join(task_log_dir, f"{task_name}_joint_step_complete.{Config.VIDEO_OUTPUT_FORMAT}")
        final_writer = imageio.get_writer(final_video_path, fps=Config.DEFAULT_FPS)
        
        # Read and merge all images
        # timestamp = f"{count:06d}", filename format: 000001_combine.jpg
        # step_timestamp = f"{count:06d}_{step_index:06d}", filename format: 000001_000000_combined.jpg
        # IMAGE = "per_inference"
        if Config.SAVE_MODE == "per_inference":
            images = sorted([img for img in os.listdir(task_log_dir) if "combined" in img and img.endswith(".jpg") and len(img.split("_"))==2])
        elif Config.SAVE_MODE == "per_joint_step":
            images = sorted([img for img in os.listdir(task_log_dir) if "combined" in img and img.endswith(".jpg") and len(img.split("_"))==3])
        
        # images = sorted([img for img in os.listdir(task_log_dir) if "combined" in img and img.endswith(".jpg")])
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
    if Config.SAVE_MODE == "per_inference":
        final_video_path = os.path.join(task_log_dir, f"{task_name}_inference_complete.mp4")
    elif Config.SAVE_MODE == "per_joint_step":
        final_video_path = os.path.join(task_log_dir, f"{task_name}_joint_step_complete.mp4")
    else:
        raise ValueError(f"Unknown SAVE_MODE: {Config.SAVE_MODE}")
    return os.path.exists(final_video_path)

def cleanup_images(task_log_dir, task_name, dry_run=False):
    """Delete all image files in the directory while keeping MP4 videos
    
    Args:
        task_log_dir: Directory containing images and videos
        task_name: Name of the task (for logging purposes)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    try:
        files = os.listdir(task_log_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        images_to_delete = []
        
        for file in files:
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in image_extensions):
                images_to_delete.append(file)
        
        if not images_to_delete:
            print(f"📁 No image files found in {task_log_dir}")
            return
        
        if dry_run:
            print(f"🔍 DRY RUN - Would delete {len(images_to_delete)} image files from {task_name}:")
            for img in images_to_delete:
                print(f"   - {img}")
        else:
            print(f"🗑️  Deleting {len(images_to_delete)} image files from {task_name}...")
            deleted_count = 0
            for img in images_to_delete:
                img_path = os.path.join(task_log_dir, img)
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    print(f"   ✅ Deleted: {img}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {img}: {e}")
            
            print(f"✅ Successfully deleted {deleted_count}/{len(images_to_delete)} image files")
    
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Process robot inference logs to merge images into videos.")
    parser.add_argument("--base-dir", type=str, default= Config.DEFAULT_VIDEO_RECORDINGS_DIR,
                        help="Base directories to search for task logs.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing final videos if they exist.")
    parser.add_argument("--cleanup-images", action="store_true",
                        help="Delete all image files after processing, keeping only MP4 videos.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting (use with --cleanup-images).")
    args = parser.parse_args()

    # If we have multiple segments, merge them
    dirs = iter_root_folder_to_get_dirs(args.base_dir)
    # print(f"Found {len(dirs)} task directories to process.")
    for dir in dirs:
        task_name, task_log_dir = get_task_name_info_from_dirname(str(dir))
        total_segments = get_image_num(task_log_dir)
        # print(f"Found {total_segments} combined images in {task_log_dir}")
        if total_segments > 0:
            if has_final_video(task_log_dir, task_name) and not args.overwrite:
                print(f"Final video already exists for {task_name} in {task_log_dir}, skipping merge.")
            else:
                merge_images_to_video(task_log_dir, task_name)
            
            # Cleanup images if requested
            if args.cleanup_images:
                cleanup_images(task_log_dir, task_name, dry_run=args.dry_run)
        else:
            print(f"⚠️ No images found in {task_log_dir}, skipping merge.")
            # Still offer to cleanup if there are other images
            if args.cleanup_images:
                cleanup_images(task_log_dir, task_name, dry_run=args.dry_run)
