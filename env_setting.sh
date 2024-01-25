#!/bin/bash

# Update system packages
sudo apt-get update

# Install htop and screen
sudo apt-get install -y htop screen

# Install Miniconda
# Create a temporary directory for the Miniconda installer
mkdir -p ~/miniconda_temp
cd ~/miniconda_temp

# Download the Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Run the Miniconda installer
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize Conda
$HOME/miniconda3/bin/conda init bash

# Clean up the installer
cd ..
rm -rf ~/miniconda_temp

# Check if Conda was installed successfully
if [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Miniconda installed successfully."
else
    echo "Miniconda installation failed."
fi
