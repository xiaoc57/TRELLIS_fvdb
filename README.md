
<h1 align="center">TRELLIS_fvdb: Unofficial Implementation of Structured 3D Latents for Scalable and Versatile 3D Generation in fvdb.</h1>

<img src="assets/logo.webp" width="100%" align="center">
<h1 align="center">Structured 3D Latents<br>for Scalable and Versatile 3D Generation</h1>
<p align="center"><a href="https://arxiv.org/abs/2412.01506"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://trellis3d.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/JeffreyXiang/TRELLIS'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>
<p align="center"><img src="assets/teaser.png" width="100%"></p>

<span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> is a large 3D asset generation model. It takes in text or image prompts and generates high-quality 3D assets in various formats, such as Radiance Fields, 3D Gaussians, and meshes. The cornerstone of <span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> is a unified Structured LATent (<span style="font-size: 16px; font-weight: 600;">SL</span><span style="font-size: 12px; font-weight: 700;">AT</span>) representation that allows decoding to different output formats and Rectified Flow Transformers tailored for <span style="font-size: 16px; font-weight: 600;">SL</span><span style="font-size: 12px; font-weight: 700;">AT</span> as the powerful backbones. We provide large-scale pre-trained models with up to 2 billion parameters on a large 3D asset dataset of 500K diverse objects. <span style="font-size: 16px; font-weight: 600;">T</span><span style="font-size: 12px; font-weight: 700;">RELLIS</span> significantly surpasses existing methods, including recent ones at similar scales, and showcases flexible output format selection and local 3D editing capabilities which were not offered by previous models.

***Check out our [Project Page](https://trellis3d.github.io) for more videos and interactive demos!***

<!-- Features -->
## 🌟 Features
- **High Quality**: It produces diverse 3D assets at high quality with intricate shape and texture details.
- **Versatility**: It takes text or image prompts and can generate various final 3D representations including but not limited to *Radiance Fields*, *3D Gaussians*, and *meshes*, accommodating diverse downstream requirements.
- **Flexible Editing**: It allows for easy editings of generated 3D assets, such as generating variants of the same object or local editing of the 3D asset.
- **fvdb**: It provides faster startup and more straightforward sparse tensor operations. 

## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 16GB of memory is necessary. The code has been verified on NVIDIA A40 GPU.  
- **Software**:   
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain submodules. The code has been tested with CUDA versions 11.8 and 12.2.  
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.8 or higher is required. 


# trellis_fvdb
使用更高级的稀疏数据处理库fvdb实现了trellis的多个模型

## 能够完成的任务
稀疏结构使用的模型是不需要修改的，
而将slat相关的数据结构由sparse Tensor修改成了FVDB中的VDBTensor

# install

```bash
    install fvdb
    install pointcept
    install xformers
    install flash-attn not
    git submodule add https://github.com/AcademySoftwareFoundation/openvdb.git openvdb
    git checkout feature/fvdb
    git add .gitmodules openvdb
    git commit -m "add openvdb" 

    cd openvdb/fvdb
    conda env create -f env/dev_environment.yml
    conda activate fvdb


    pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

    export MAX_JOBS=$(free -g | awk '/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}')
    python setup.py develop

    cd ../../
    pip install -r requirements.txt
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

    export CUDA_VISIBLE_DEVICES=1

    # install Pointcept
    pip install Pointcept/
    cd Pointcept/libs/pointops
    python setup.py install
    cd ../../../

    # reinstall !!!!!!!!!!!
    pip install spconv-cu120
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    pip install torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    
    python -m sp2sl.tests.test_trellis_fvdb_vae
```

## fvdb的独特问题
单一数据处理时因为dijk会有问题
当ddp时因为TorchDeviceBuffer::create会有问题
特征不对应问题
如果是单一数据集需要jagged——like
卷积weight的问题


# TODO
[x] valid step 
test_step
几个事情
第一个事情就是对于obj数据集
测试部分
可变学习率？
[x] 数据增强方法

<!-- 多节点多卡 (不要想了，没有nvlink效率不高) -->

shapenet方法，我有shapenetv1 高斯，物体，怎么训练这个？

最后就是结构化的高斯能不能用于压缩方法。


目前的问题，

- [x] 高斯太少，应该使用32
- [x] 高斯范围太大，应该放到一个邻域内
- [x] 是否使用球谐函数的高阶
- [x] 优化了使用多少张图像的参数输入
- [x] 是否直接使用ood数据
位置要不要使用一个高频函数

 <!-- - x 数据集初始化 参数
 - x 数据集block 如何block
 - x 优化模型结构
 - x 模型加载检查点
 - x gs_decoder 冻结参数
 - grid 正则化方法
 - 检查grid与高斯的对应关系
 - 高斯分block的方法  --># TRELLIS_fvdb
