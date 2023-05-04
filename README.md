# slickformer

### Running dataset creation

First, you'll need to sync the data from s3. If you're a devseeder, use your devseed profile. If not, sorry! This is a WIP.

```
export AWS_PROFILE=devseed
aws s3 sync s3://slickformer/cv2-training/ data/cv2/
aws s3 sync s3://slickformer/aux_datasets/ data/aux_datasets/
aws s3 sync s3://slickformer/partition_lists/ data/partition_lists/
```

### Running the jupyter server locally with slickformer dependencies to make the dataset

Start the docker container, but with the skytruth AWS_PROFILE. this is necessary to create the dataset locally.

```
docker build -t slickserver .
```
Don't use `--gpus all` if you're doing this on a machine without a GPU. and Omit the transformers line if not editing transformers lib
```
docker run -it --rm \
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/home/work/slickformer \
    -p 8888:8888 \
    -e AWS_PROFILE=devseed \
    --gpus all slickserver
```

this will start a jupyter server. You can connect to the container with a VSCode Remote session for the next step.

Then, from the docker container with the slickformer conda environment activated, run

`bash scripts make_datasets.sh`

This will create the train, validation, and test sets without tiling, and extract the annotations from the Photopea image layers into a COCO JSON.

After creating the dataset, you can start the jupyter server to work with it in jupyter or VSCode Remote using the same command.

```
docker run -it --rm \
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/slickformer \
    -p 8888:8888 \
    -e AWS_PROFILE=devseed \
    --gpus all slickserver
```

### Pushing the container

to create an ecr repository and push the image after logging in:

`aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin YourAccountID.dkr.ecr.eu-central-1`

```
aws ecr create-repository \
    --repository-name slickformer \     
    --image-scanning-configuration scanOnPush=true \
    --region eu-central-1

docker tag slickserver-pl:latest YourAccountID.dkr.ecr.eu-central-1.amazonaws.com/slickformer

docker push YourAccountID.dkr.ecr.eu-central-1.amazonaws.com/slickformer
```
