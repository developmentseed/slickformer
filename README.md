# slickformer


### Running the jupyter server with slickformer dependencies

```
docker build -t slickserver .
```

```
docker run -it --rm \
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/slickformer \
    -p 8888:8888 \
    -e AWS_PROFILE=devseed \
    --gpus all slickserver
```