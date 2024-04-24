#!/bin/bash

# Define the path to the reference genome using the absolute path
reference_genome="/Users/hungtruong/Desktop/School/cs490/FinalProject/Assignment-1-CS490/hg38.fa"

# Loop through each BED file in the Data directory
for bed_file in *.bed; do
    # Define the output file name by replacing the .bed extension with .txt
    output_file="${bed_file%.bed}.txt"
    
    # Use bedtools to extract sequences and save them to the output file
    bedtools getfasta -fi "$reference_genome" -bed "$bed_file" -fo "$output_file"
    
    echo "Generated $output_file from $bed_file"
done

# Move back to the original directory
cd ..

