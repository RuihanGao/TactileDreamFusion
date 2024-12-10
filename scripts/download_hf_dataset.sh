#!/bin/bash

# enable faster huggingface data transfer and download
export HF_HUB_ENABLE_HF_TRANSFER=1

# set data directory
data_dir="data"
# create data directory if it does not exist
mkdir -p "$data_dir"

# set cache directory
cache_dir="$data_dir/.cache"
export CACHE_DIR="$cache_dir"

# create a for loop to download different .zip files
zip_files=(
    # Uncomment the desired files to download
    "tactile_textures.zip"
    "base_meshes.zip"
)

for zip_file in "${zip_files[@]}"; do
    # ensure cache directory exists
    mkdir -p "$CACHE_DIR"

    # export the ZIP_FILE environment variable
    export ZIP_FILE="$zip_file"

    # use the cache_dir variable in the Python script
    python -c "
import os
from huggingface_hub import snapshot_download

# Get environment variables
cache_dir = os.getenv('CACHE_DIR')
zip_file = os.getenv('ZIP_FILE')

# Validate environment variables
if not cache_dir:
    raise ValueError('CACHE_DIR environment variable is not set')
if not zip_file:
    raise ValueError('ZIP_FILE environment variable is not set')

# Download the specific zip file
snapshot_download(
    repo_id='Ruihan28/TactileDreamFusion',
    allow_patterns=[zip_file],
    repo_type='dataset',
    local_dir='data/',
    cache_dir=cache_dir
)
"

    # unzip the downloaded file and remove the .zip file
    filename="$data_dir/$zip_file"
    if [[ -f "$filename" ]]; then
        unzip -o "$filename" -d "$data_dir/"
        rm "$filename"
    else
        echo "Warning: File $filename not found. Skipping unzip."
    fi
done

# remove cache directory
rm -rf "$cache_dir"
