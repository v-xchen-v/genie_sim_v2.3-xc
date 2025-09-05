#!/bin/bash

# Simple backup and restore script for video_recordings and output folders
# Usage: 
#   ./backup_outputs.sh backup   - Backup both folders
#   ./backup_outputs.sh restore  - Restore both folders from latest backup

set -e  # Exit on any error

# Auto-detect environment and set base directory
if [ -d "/root/workspace/main" ]; then
    BASE_DIR="/root/workspace/main"
elif [ -d "/home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc" ]; then
    BASE_DIR="/home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc"
else
    echo "ERROR: Cannot detect environment"
    exit 1
fi

# Directories to backup/restore
VIDEO_DIR="$BASE_DIR/AgiBot-World-Submission/CogACT/video_recordings"
OUTPUT_DIR="$BASE_DIR/source/geniesim/benchmark/output"
BACKUP_DIR="$BASE_DIR/backups"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to backup folders
backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "Creating backup with timestamp: $timestamp"
    echo "Backup directory: $BACKUP_DIR"
    
    # Backup video_recordings
    if [ -d "$VIDEO_DIR" ]; then
        echo "Backing up video_recordings..."
        cd "$(dirname "$VIDEO_DIR")"
        zip -r "$BACKUP_DIR/video_recordings_$timestamp.zip" "$(basename "$VIDEO_DIR")" > /dev/null
        echo "✓ Video recordings backed up to: video_recordings_$timestamp.zip"
    else
        echo "⚠ Video recordings directory not found: $VIDEO_DIR"
    fi
    
    # Backup output
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Backing up benchmark output..."
        cd "$(dirname "$OUTPUT_DIR")"
        zip -r "$BACKUP_DIR/benchmark_output_$timestamp.zip" "$(basename "$OUTPUT_DIR")" > /dev/null
        echo "✓ Benchmark output backed up to: benchmark_output_$timestamp.zip"
    else
        echo "⚠ Benchmark output directory not found: $OUTPUT_DIR"
    fi
    
    echo "Backup completed!"
}

# Function to restore folders
restore() {
    echo "Looking for latest backups in: $BACKUP_DIR"
    
    # Find latest video backup
    local latest_video=$(ls -t "$BACKUP_DIR"/video_recordings_*.zip 2>/dev/null | head -1)
    # Find latest output backup  
    local latest_output=$(ls -t "$BACKUP_DIR"/benchmark_output_*.zip 2>/dev/null | head -1)
    
    if [ -z "$latest_video" ] && [ -z "$latest_output" ]; then
        echo "ERROR: No backup files found in $BACKUP_DIR"
        exit 1
    fi
    
    # Restore video_recordings
    if [ -n "$latest_video" ]; then
        echo "Restoring video_recordings from: $(basename "$latest_video")"
        
        # Remove existing directory
        if [ -d "$VIDEO_DIR" ]; then
            rm -rf "$VIDEO_DIR"
        fi
        
        # Extract backup
        cd "$(dirname "$VIDEO_DIR")"
        unzip -q "$latest_video"
        echo "✓ Video recordings restored"
    else
        echo "⚠ No video recordings backup found"
    fi
    
    # Restore benchmark output
    if [ -n "$latest_output" ]; then
        echo "Restoring benchmark output from: $(basename "$latest_output")"
        
        # Remove existing directory
        if [ -d "$OUTPUT_DIR" ]; then
            rm -rf "$OUTPUT_DIR"
        fi
        
        # Extract backup
        cd "$(dirname "$OUTPUT_DIR")"
        unzip -q "$latest_output"
        echo "✓ Benchmark output restored"
    else
        echo "⚠ No benchmark output backup found"
    fi
    
    echo "Restore completed!"
}

# Function to list backups
list() {
    echo "Available backups in: $BACKUP_DIR"
    echo ""
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]; then
        echo "No backups found"
        return
    fi
    
    echo "Video recordings backups:"
    ls -la "$BACKUP_DIR"/video_recordings_*.zip 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}' || echo "  None found"
    
    echo ""
    echo "Benchmark output backups:"
    ls -la "$BACKUP_DIR"/benchmark_output_*.zip 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes, " $6 " " $7 " " $8 ")"}' || echo "  None found"
}

# Main function
case "${1:-}" in
    backup)
        backup
        ;;
    restore)
        restore
        ;;
    list)
        list
        ;;
    *)
        echo "Simple Backup/Restore Script"
        echo ""
        echo "Usage: $0 {backup|restore|list}"
        echo ""
        echo "Commands:"
        echo "  backup   - Create zip backups of video_recordings and output folders"
        echo "  restore  - Restore folders from latest backup files"
        echo "  list     - List available backup files"
        echo ""
        echo "Directories:"
        echo "  Video:  $VIDEO_DIR"
        echo "  Output: $OUTPUT_DIR"
        echo "  Backup: $BACKUP_DIR"
        echo ""
        echo "Example:"
        echo "  $0 backup    # Create backups"
        echo "  $0 restore   # Restore from latest backups"
        exit 1
        ;;
esac
