#!/bin/bash
# Usage: ./scripts/zip_submission_pack.sh <SUBFOLDER_PATH> <OUTPUT_ZIP_NAME>
# Example: ./scripts/zip_submission_pack.sh AgiBot-World-Submission AgiBot-World.zip

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <SUBFOLDER_PATH> <OUTPUT_ZIP_NAME>"
    exit 1
fi

SUBFOLDER="$1"
OUTPUT_ZIP="$2"

# Temporary clean copy
CLEAN_FOLDER="${SUBFOLDER}_clean"

echo "Cleaning and preparing $SUBFOLDER → $CLEAN_FOLDER ..."
rm -rf "$CLEAN_FOLDER"
rsync -av "$SUBFOLDER/" "$CLEAN_FOLDER/" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".git" \
  --exclude=".gitignore" \
  --exclude=".DS_Store" \
  --exclude="*.egg-info" \
  --exclude="build" \
  --exclude="dist" \
  --exclude="*.db" \
  --exclude="*.log"

echo "Creating zip archive: $OUTPUT_ZIP ..."
rm -f "$OUTPUT_ZIP"
zip -r "$OUTPUT_ZIP" "$CLEAN_FOLDER"

echo "Done! Archive saved as $OUTPUT_ZIP"
