#!/bin/bash

# Multi-batch run script for CogACT inference with different configurations
# This script runs multiple batchrun.sh executions in sequence with different inference configs
# and organizes output files to avoid conflicts

set -e  # Exit on any error

# # Auto-detect environment and set base directory
# if [ -d "/root/workspace/main" ]; then
#     # Running inside container
#     BASE_DIR="/root/workspace/main"
#     echo() {
#         echo "[INFO] (Container) $1"
#     }
# elif [ -d "/home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc" ]; then
#     # Running outside container
#     BASE_DIR="/home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc"
#     echo() {
#         echo "[INFO] (Host) $1"
#     }
# else
#     echo "[ERROR] Cannot detect environment. Neither container nor host base directory found."
#     echo "  Container path: /root/workspace/main"
#     echo "  Host path: /home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc"
#     exit 1
# fi
BASE_DIR="$(pwd)"
# Configuration based on detected environment
COGACT_DIR="$BASE_DIR/AgiBot-World-Submission/CogACT"
BENCHMARK_OUTPUT_DIR="$BASE_DIR/source/geniesim/benchmark/output"
SCRIPT_DIR="$BASE_DIR/scripts"
BACKUP_DIR="$BASE_DIR/multi_run_outputs"

echo "Detected base directory: $BASE_DIR"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Function to print colored output
print_header() {
    echo "========================================"
    echo "$1"
    echo "========================================"
}

print_error() {
    echo "[ERROR] $1" >&2
}

# Function to backup and rename output directories
backup_outputs() {
    local run_id="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local run_backup_dir="$BACKUP_DIR/run_${timestamp}_${run_id}"
    
    echo "Backing up outputs for run $run_id to $run_backup_dir"
    mkdir -p "$run_backup_dir"
    
    # Backup video_recordings from CogACT
    if [ -d "$COGACT_DIR/video_recordings" ]; then
        echo "Backing up CogACT video_recordings..."
        cp -r "$COGACT_DIR/video_recordings" "$run_backup_dir/video_recordings_${run_id}_${timestamp}"
        # Clean original directory but keep the folder structure
        find "$COGACT_DIR/video_recordings" -mindepth 1 -delete
    fi
    
    # Backup inference_logs from CogACT
    if [ -d "$COGACT_DIR/inference_logs" ]; then
        echo "Backing up CogACT inference_logs..."
        cp -r "$COGACT_DIR/inference_logs" "$run_backup_dir/cogact_inference_logs_${run_id}_${timestamp}"
        # Clean original directory but keep the folder structure
        find "$COGACT_DIR/inference_logs" -mindepth 1 -delete
    fi
    
    # Backup benchmark output
    if [ -d "$BENCHMARK_OUTPUT_DIR" ]; then
        echo "Backing up benchmark output..."
        cp -r "$BENCHMARK_OUTPUT_DIR" "$run_backup_dir/benchmark_output_${run_id}_${timestamp}"
        # Clean original directory but keep the folder structure
        find "$BENCHMARK_OUTPUT_DIR" -mindepth 1 -delete 2>/dev/null || true
    fi
    
    echo "Backup completed for run $run_id"
}

# Function to restore original config
restore_original_config() {
    if [ -f "$COGACT_DIR/inference_config.yaml.original" ]; then
        echo "Restoring original inference_config.yaml"
        cp "$COGACT_DIR/inference_config.yaml.original" "$COGACT_DIR/inference_config.yaml"
        rm "$COGACT_DIR/inference_config.yaml.original"
    fi
}

# Function to switch inference config
switch_config() {
    local config_file="$1"
    local run_id="$2"
    
    echo "Switching to configuration: $config_file (Run ID: $run_id)"
    
    # Backup original config if not already backed up
    if [ ! -f "$COGACT_DIR/inference_config.yaml.original" ]; then
        cp "$COGACT_DIR/inference_config.yaml" "$COGACT_DIR/inference_config.yaml.original"
    fi
    
    # Copy the specific config to the main location
    if [ -f "$COGACT_DIR/$config_file" ]; then
        cp "$COGACT_DIR/$config_file" "$COGACT_DIR/inference_config.yaml"
        echo "Successfully switched to $config_file"
    else
        print_error "Configuration file $config_file not found!"
        return 1
    fi
}

# Function to run batch execution
run_batch() {
    local run_id="$1"
    local config_file="$2"
    local task_name="${3:-all}"
    local model_name="${4:-CogACT}"
    
    print_header "Starting Run $run_id with config $config_file"
    
    # Switch to the specific config
    switch_config "$config_file" "$run_id"
    
    # Add run info to a log file
    echo "$(date): Run $run_id started with config $config_file, task: $task_name, model: $model_name" >> "$BACKUP_DIR/multi_run.log"
    
    # Run the batch script
    echo "Executing batchrun.sh -1 $task_name $model_name"
    cd "$BASE_DIR"  # Change to base directory
    
    if ./scripts/batchrun.sh -1 "$task_name" "$model_name"; then
        echo "Run $run_id completed successfully"
        echo "$(date): Run $run_id completed successfully" >> "$BACKUP_DIR/multi_run.log"
    else
        print_error "Run $run_id failed!"
        echo "$(date): Run $run_id FAILED" >> "$BACKUP_DIR/multi_run.log"
    fi
    
    # Backup outputs after the run
    backup_outputs "$run_id"
    
    echo "Run $run_id finished and outputs backed up"
}

# Cleanup function for script interruption
cleanup() {
    echo "Script interrupted, cleaning up..."
    restore_original_config
    ./scripts/autorun.sh clean 2>/dev/null || true
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_header "Multi-Batch Run Script for CogACT"
    
    # Validate that we're in the right directory
    if [ ! -f "$COGACT_DIR/inference_config.yaml" ]; then
        print_error "inference_config.yaml not found in $COGACT_DIR"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_DIR/batchrun.sh" ]; then
        print_error "batchrun.sh not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Configuration sets to run
    # Format: "run_id:config_file:task_name:model_name"
    # You can modify this array to add/remove/change configurations
    configs=(
        "1_port28015:inference_config.28015.yaml:all:CogACT"
        "2_port31015:inference_config.31015.yaml:all:CogACT"
        "3_port31115:inference_config.31115.yaml:all:CogACT"
        "4_port28115:inference_config.28115.yaml:all:CogACT"
        "5_port28215:inference_config.28215.yaml:all:CogACT"
        "6_port28315:inference_config.28315.yaml:all:CogACT"
        # "3:inference_config.24xxx.yaml:all:CogACT"
        # "4:inference_config.19xxx.yaml:all:CogACT"
    )
    
    echo "Found ${#configs[@]} configuration sets to run"
    
    # Create log file
    echo "$(date): Multi-batch run started" > "$BACKUP_DIR/multi_run.log"
    
    # Execute each configuration
    for config_set in "${configs[@]}"; do
        IFS=':' read -r run_id config_file task_name model_name <<< "$config_set"
        
        echo "Preparing to run configuration set: $config_set"
        
        # Run the batch with current config
        run_batch "$run_id" "$config_file" "$task_name" "$model_name"
        
        # Wait a bit between runs
        echo "Waiting 10 seconds before next run..."
        sleep 10
    done
    
    # Restore original configuration
    restore_original_config
    
    print_header "All batch runs completed!"
    echo "Results are stored in: $BACKUP_DIR"
    echo "Check the log file: $BACKUP_DIR/multi_run.log"
    
    # Show summary
    echo ""
    echo "Summary of runs:"
    cat "$BACKUP_DIR/multi_run.log"
}

# Show usage if help requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0"
    echo ""
    echo "This script runs multiple batchrun.sh executions in sequence with different"
    echo "inference configurations and organizes output files to avoid conflicts."
    echo ""
    echo "Environment Detection:"
    echo "  - Container: /root/workspace/main"
    echo "  - Host: /home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc"
    echo "  - Current: $BASE_DIR"
    echo ""
    echo "The script will:"
    echo "  1. Run batchrun.sh -1 all CogACT with different inference configs"
    echo "  2. After each run, backup video_recordings and output directories"
    echo "  3. Restore the original inference_config.yaml at the end"
    echo ""
    echo "Output directories are backed up to: $BACKUP_DIR"
    echo "Each run gets a unique timestamp and run ID."
    echo ""
    echo "Configurations to run are defined in the script itself."
    echo "Edit the 'configs' array in the script to modify what runs."
    exit 0
fi

# Run main function
main "$@"
