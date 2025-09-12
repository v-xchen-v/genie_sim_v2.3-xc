#!/bin/bash
# Usage: ./scripts/zip_submission_pack.sh <SUBFOLDER_PATH> <OUTPUT_ZIP_NAME>
# Example: ./scripts/zip_submission_pack.sh AgiBot-World-Submission/CogACT AgiBot-World.zip
# Tips:
# check the temp file filesize: watch -n 1 ls -lh AgiBot-World.zip
# check the file list in the zip: unzip -l AgiBot-World.zip

set -euo pipefail

# if [ $# -ne 2 ]; then
#     echo "Usage: $0 <SUBFOLDER_PATH> <OUTPUT_ZIP_NAME>"
#     exit 1
# fi

SUBFOLDER="AgiBot-World-Submission/CogACT" # $1
OUTPUT_ZIP="AgiBot-World.zip" # $2

# Temporary clean copy
CLEAN_FOLDER="CogACT_clean" # "${SUBFOLDER}_clean"

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
# zip -r "$OUTPUT_ZIP" "$CLEAN_FOLDER"
zip -r -0 "$OUTPUT_ZIP" "$CLEAN_FOLDER" # no compression to speed up zipping
# remove the temporary clean copy
# rm -rf "$CLEAN_FOLDER"

# move the zip to the submission folder
mv "$OUTPUT_ZIP" "submission/$OUTPUT_ZIP"
# copy requirements.txt in SUBFOLDERPATH to the submission folder
cp "${SUBFOLDER}/requirements.txt" "submission/"

echo "Done! Archive saved as $OUTPUT_ZIP"
