FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    wget

# Set environment variables
ENV PATH="/root/mambaforge/bin:$PATH"

# Install Conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm Mambaforge-Linux-x86_64.sh \
    && echo 'export PATH="/root/mambaforge/bin:$PATH"' >> ~/.bashrc

# Install Conda packages
RUN /bin/bash -c "source ~/.bashrc && mamba install -n base -c conda-forge jupyterlab_widgets jupyterlab nb_conda_kernels ipykernel ipywidgets black isort -y"

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get -y install gh

# Set the working directory to /app
WORKDIR /slickformer

# Set the volume to mount the local directory where the Dockerfile is in
VOLUME /slickformer

# Start Jupyter Lab
CMD ["/bin/bash", "-c", "source ~/.bashrc && jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 8888 --notebook-dir=/slickformer"]