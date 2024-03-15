#!/bin/bash

# List of gzipped bed files to decompress
files=(
    "ENCFF298ANC.bed.gz"
    "ENCFF768XXW.bed.gz"
    "ENCFF397KNY.bed.gz"
    "ENCFF825KFE.bed.gz"
    "ENCFF092DWH.bed.gz"
    "ENCFF538PLU.bed.gz"
    "ENCFF931AKV.bed.gz"
    "ENCFF139HDN.bed.gz"
    "ENCFF615ZGF.bed.gz"
    "ENCFF986UZO.bed.gz"
)

# Loop through each file and decompress it with gunzip
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Decompressing $file..."
        gunzip "$file"
    else
        echo "File $file does not exist, skipping..."
    fi
done

echo "Decompression complete."
