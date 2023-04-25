FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Create a non-root user
ARG USERNAME=work
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    wget \
    libxml2 \
    git \
    xdg-utils \
    libgl1 -y
# last two not installed by conda for some reason for opencv

# Set environment variables
ENV PATH="/home/$USERNAME/mambaforge/bin:$PATH"
RUN mkdir -p /home/$USERNAME/

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get -y install gh

# Install Conda
USER $USERNAME
WORKDIR /home/$USERNAME
ENV CONDA_DIR="/home/$USERNAME/mambaforge"
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    && bash Mambaforge-Linux-x86_64.sh -b \
    && rm Mambaforge-Linux-x86_64.sh \
    && echo 'export PATH="/home/$USERNAME/mambaforge/bin:$PATH"' >> ~/.bashrc

# Install Conda packages for jupyter server
RUN mamba install -n base -c conda-forge jupyterlab_widgets jupyterlab nb_conda_kernels ipywidgets black isort -y

##### Install custom Conda packages

# Copy the environment.yaml file to the image
COPY --chown=$USER_UID:users environment.yaml /home/$USERNAME/environment.yaml

# allows installation if cuda not on host, vague explanation 
# here: https://conda-forge.org/docs/maintainer/knowledge_base.html#cuda-builds
ENV CONDA_OVERRIDE_CUDA=11.7

# Create a new conda environment based on the environment.yaml file
RUN mkdir -p /home/$USERNAME/tmp && \
    chown $USERNAME:$USER_GID /home/$USERNAME/tmp

ENV TMPDIR=/home/$USERNAME/tmp

RUN TMPDIR=$TMPDIR mamba env create -f /home/$USERNAME/environment.yaml --quiet

# Activate the new environment
SHELL ["conda", "run", "-n", "slickformer", "/bin/bash", "-c"]

# # Log in to Weights and Biases
# RUN wandb login

WORKDIR /home/$USERNAME/slickformer

VOLUME  /home/$USERNAME/slickformer

# Copy the library file to the image
COPY ceruleanml /home/$USERNAME/slickformer
COPY setup.py /home/$USERNAME/slickformer
COPY scripts/download_models_and_configs.py /home/$USERNAME/slickformer
RUN conda run -n slickformer pip install -e .

# so we can activate envs in vscode remote container connection
RUN conda init

RUN python /home/$USERNAME/slickformer/download_models_and_configs.py

# Start Jupyter Lab
CMD ["/bin/bash", "-c", "umask 002 && jupyter lab --allow-root --no-browser --ip 0.0.0.0 --port 8888 --notebook-dir=$HOME/slickformer"]
