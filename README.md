# 3DGS: 3D Gaussian Splatting

## 简介

`3DGS` 项目旨在实现 3D Gaussian Splatting 技术，用于高效的 3D 数据处理和渲染。本项目基于 Python 和 PyTorch，主要对3DGS的模型部分进行复现。

## 安装

### 实验环境

- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

### 安装步骤

1.创建并激活 Conda 虚拟环境：

```
conda create -n gaussian_splatting python=3.7
conda activate gaussian_splatting
conda install -c conda-forge vs2022_win-64
```

2.安装 PyTorch 和相关库：

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

或者使用 Conda：

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3.安装其他依赖：

```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install plyfile
pip install tqdm
```
或者（未测试过）：

```
conda env create --file environment.yml
conda activate gaussian_splatting
```

## 依赖项注明

- [Differential Gaussian Rasterization][graphdeco-inria/diff-gaussian-rasterization (github.com)](https://github.com/graphdeco-inria/diff-gaussian-rasterization)用于光栅化
- [SIBR Viewers][JayFoxRox/SIBR_viewers (github.com)](https://github.com/JayFoxRox/SIBR_viewers)用于渲染交互界面，[开发文档][SIBR Website (inria.fr)](https://sibr.gitlabpages.inria.fr/?page=index.html&version=0.9.6)

## 训练方法
### 训练
```
python train.py --config configs/your_config.yaml
```
### 渲染
```
python render.py --config configs/your_config.yaml --output_dir <path_to_output>
```
### 评估
```
python evaluate.py --model_paths <path_to_model>
```

## 目录结构
```
3DGS/
├── arguments                        # 存储脚本运行所需参数的模块
├── bin                              # 存储可执行文件或二进制文件的目录
├── gaussian_renderer                # 3D Gaussian Splatting 线上渲染相关代码
├── main                             # 训练、渲染、评估代码（主要代码入口）
├── resources                        # 存储项目资源文件
├── scene                            # 场景相关的代码和数据（包括场景类以及3DGS模型类）
├── shaders                          # 着色器
├── SIBR_viewers                     # 结果查看交互器
├── submodules                       # 依赖子模块
├── utils                            # 工具函数
├── viewers                          # 可视化相关
├── .gitignore
├── .gitmodules
├── 3D Gaussian Splatting.md         # 简要流程介绍文档
└── environment.yml                  # 可选环境配置文件
```
## 结果录屏
[录屏链接][https://www.alipan.com/s/QpDz2NEUHks]
