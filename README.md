# XGNN

XGNN是一个高性能的分布式GNN框架，同时单机支持处理千亿边的大规模数据集。

1. 分布式模块采用模型并行与数据并行的混合并行模式，并通过流水线处理重叠通信与计算开销。
2. 单机实现了高性能的图聚合kernel，大大提高了GCN、GIN等GNN算法的计算效率。同时为了支持处理千亿边级别的大规模图数据，XGNN将图拓扑和图特征进行partition，然后利用内存映射将图拓扑和图特征分块映射到内存中。


# 安装

提供两种安装方式：
1. 利用Docker构建镜像
2. 利用conda安装环境

## 构建Docker镜像

1. 首先利用Docker构建XGNN运行的虚拟环境，Docker的安装请[参考](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)。如下我们构建了一个叫做`xgnn`的镜像，之后的所有实验都是在该镜像上完成的。
```shell
cd xgnn/docker
docker build -f Dockerfile -t xgnn .
```
2. 启动docker容器：
```shell
docker run --gpus all -e NCCL_SOCKET_IFNAME=eno1 --rm -dit --shm-size="5g" --network=host --name xgnn xgnn /bin/bash
```
* --gpus all：表示使用所有gpu
* -e NCCL_SOCKET_IFNAME=eno1: 设置需要使用的网卡
* --shm-size="5g": 设置共享内存大小
* --network=host: 使用主机的网络

3. docker容器内安装xgnn：
```shell
# 1. 将xgnn复制到docker
docker cp xgnn xgnn:/home
# 2. 进入容器
docker exec -it xgnn bash
cd /home/xgnn
# 3. 安装xgnn
python setup.py install
```

## 基于conda安装

1. 安装cmake:
```shell
version=3.18
build=0
mkdir ~/temp
cd ~/temp
wget https://cmake.org/files/v$version/cmake-$version.$build-Linux-x86_64.sh 
sudo mkdir /opt/cmake
sudo sh cmake-$version.$build-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
cd ~
rm -rf ~/temp
```
2. 安装conda:
```shell
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export PATH=/opt/conda/bin:$PATH

apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

export TINI_VERSION=v0.16.1
source ~/.bashrc
```
3. 安装相关依赖包：
```shell
conda create -n xgnn python=3.7.5
conda activate xgnn
conda install -y astunparse numpy ninja pyyaml mkl \
	mkl-include setuptools cffi \
	typing_extensions future six \
	requests dataclasses \
	pytest nose cython scipy \
	networkx matplotlib nltk \
	tqdm pandas scikit-learn && \
	conda install -y -c pytorch magma-cuda102
# 单机依赖包
apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    libgoogle-perftools-dev \
    protobuf-compiler && \
	rm -rf /var/lib/apt/lists/*
```
4. 编译安装pytorch:
```shell
mkdir ~/temp
cd ~/temp
# 下载能够支持多版本参数的PyTorch源码
git clone --recursive https://github.com/Ningsir/pytorch.git -b multi-version
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

# 编译安装PyTorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
USE_NINJA=OFF python setup.py install --cmake

cd ~
rm -rf ~/temp
```

> 可能出现的问题：报错缺少valgrind.h文件
>
> 解决办法：cd third_party/valgrind && git checkout VALGRIND_3_18_0
5. 安装dgl:
```shell
conda install -y -c dglteam dgl-cuda10.2=0.7.1
```
# License
