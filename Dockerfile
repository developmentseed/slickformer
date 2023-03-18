FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    wget \
    libxml2 \
    git \
    xdg-utils \
    libjpeg-turbo8 \
    libgl1 -y
# last two not installed by conda for some reason for opencv

# Set environment variables
ENV PATH="/root/mambaforge/bin:$PATH"

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get -y install gh

# Install Conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm Mambaforge-Linux-x86_64.sh \
    && echo 'export PATH="/root/mambaforge/bin:$PATH"' >> ~/.bashrc

# Install Conda packages for jupyter server
RUN mamba install -n base -c conda-forge jupyterlab_widgets jupyterlab nb_conda_kernels ipywidgets black isort -y

##### Install custom Conda packages

# Copy the environment.yaml file to the image
COPY environment.yaml /

# allows installation if cuda not on host, vague explanation 
# here: https://conda-forge.org/docs/maintainer/knowledge_base.html#cuda-builds
ENV CONDA_OVERRIDE_CUDA=11.7

# Create a new conda environment based on the environment.yaml file
RUN mamba env create -f /environment.yaml --quiet

# Activate the new environment
SHELL ["conda", "run", "-n", "slickformer", "/bin/bash", "-c"]

# # Log in to Weights and Biases
# RUN wandb login

# Set the working directory to /slickformer
WORKDIR /slickformer

# Set the volume to mount the local directory where the Dockerfile is in
VOLUME /slickformer

# Copy the library file to the image
COPY ceruleanml /slickformer
COPY setup.py /slickformer

RUN pip install -e .

# so we can activate envs in vscode remote container connection
RUN conda init

# Start Jupyter Lab
CMD ["/bin/bash", "-c", "jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 8888 --notebook-dir=/slickformer"]