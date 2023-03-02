#!/bin/bash
# Add additional installs or env setups if needed

sudo apt-get update
sudo apt-get -y upgrade
sudo apt install awscli -y
export AWS_PROFILE=devseed # you need to manually add creds on vm

# Conda
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh -b
rm Mambaforge-Linux-x86_64.sh
export PATH="$HOME/mambaforge/bin:$PATH"
echo 'export PATH="$HOME/mambaforge/bin:$PATH"' >> ~/.bashrc
printf "alias jserve='jupyter lab --allow-root --no-browser'\n" >> /home/ubuntu/.bashrc #start jupyter

# Conda packages
mamba install -n base -c conda-forge jupyterlab_widgets jupyterlab nb_conda_kernels ipykernel ipywidgets black isort -y

# GitHub CLI
sudo curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update
sudo apt-get -y install gh