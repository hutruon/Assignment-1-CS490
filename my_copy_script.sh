#!/bin/bash

# Get source and destination directory from user input
SOURCE=$1
DESTINATION=$2

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 source_directory destination_directory"
    exit 1
fi

# Copy files from source to destination
cp -r $SOURCE/* $DESTINATION/

echo "Files copied from $SOURCE to $DESTINATION successfully."
