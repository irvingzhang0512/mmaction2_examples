# create container
# -d 后台运行
# -it /bin/bash 保持容器持续运行
# --name 指定容器名称
# -p 8022:22 指定ssh端口，方便后续使用ssh远程开发
# --gpus 指定使用的GPU
docker run -it -d \
    -p 8022:22 \
    --name zyy_mmaction2 \
    --gpus all \
    mmaction2 \
    /bin/bash

docker run -d \
    -p 8022:22 \
    --name zyy_mmaction2 \
    --gpus all \
    mmaction2