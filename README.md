# slickformer

### Running dataset creation

First, you'll need to sync the data from s3.

```
export AWS_PROFILE=devseed
aws s3 sync s3://slickformer/cv2-training/ data/cv2/
aws s3 sync s3://slickformer/aux_datasets/ data/aux_datasets/
aws s3 sync s3://slickformer/partition_lists/ data/partition_lists/
```

then, from the docker container with the slickformer conda environment activated, run

`bash scripts make_datasets.sh`

This will create the train, validation, and test sets without tiling, and extract the annotations from the Photopea image layers into a COCO JSON.

### Running the jupyter server locally with slickformer dependencies

```
docker build -t slickserver .
```

```
docker run -it --rm \
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/slickformer \
    -p 8888:8888 \
    -e AWS_PROFILE=skytruth \
    --gpus all slickserver
```

### Pulling the container from ECR

`docker sts get-caller-identity` to get your account id (not the user ID!)

then

`aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin YourAccountID.dkr.ecr.eu-central-1`

then 

```
docker pull YourAccountID.dkr.ecr.eu-central-1.amazonaws.com/slickformer

```

you can then start it with

```
docker run -it --rm \
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/slickformer \
    -p 8888:8888 \
    -e AWS_PROFILE=devseed \
    --gpus all slickserver
```

### Pushing the container

to remake the ecr repository and push the image after logging in with

`aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin YourAccountID.dkr.ecr.eu-central-1`

```
aws ecr create-repository \
    --repository-name slickformer \     
    --image-scanning-configuration scanOnPush=true \
    --region eu-central-1

docker tag slickserver-pl:latest YourAccountID.dkr.ecr.eu-central-1.amazonaws.com/slickformer

docker push YourAccountID.dkr.ecr.eu-central-1.amazonaws.com/slickformer
```
