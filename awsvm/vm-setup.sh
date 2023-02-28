#!/bin/bash
# Add additional installs or env setups if needed

sudo apt-get update
sudo apt-get -y upgrade 

# Docker
sudo apt-get -y install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
sudo echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
sudo chmod 666 /var/run/docker.sock
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

# mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
# mamba install -c fastchan fastai -y
# mamba install -c conda-forge ipykernel ipywidgets black isort  jupyterlab_code_formatter -y
# mamba deactivate

# GitHub CLI
sudo curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
sudo echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update
sudo apt-get -y install gh

# CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda