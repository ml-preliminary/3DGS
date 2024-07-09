# 3D Gaussian Splatting

## Overview

- A ==PyTorch-based optimizer== to produce a 3D Gaussian model from SfM inputs
- A ==network viewer== that allows to connect to and visualize the optimization process
- An ==OpenGL-based real-time viewer== to render trained models in real-time.
- A script to help you ==turn your own images into optimization-ready SfM data sets==

## 核心代码流程

~~~伪代码
FUNCTION 3DGaussianSplatting(pcd: BasicPointCloud, camera_params, training_params, spatial_lr_scale: float):
    # 初始化阶段
    xyz, features_dc, features_rest, scaling, rotation, opacity = InitAttributes(pcd, spatial_lr_scale)
    i = 0

    WHILE NOT converged DO
        # 获取训练视图和图像
        V, Ĩ = SampleTrainingView()

        # 栅格化生成图像
        I = Rasterize(xyz, features_dc, features_rest, scaling, rotation, opacity, V)
        
        # 计算损失并反向传播
        L = ComputeLoss(I, Ĩ)
        BackpropAndStep(L)

        # 检查是否需要细化
        IF IsRefinementIteration(i) THEN
            FOR EACH Gaussian IN (xyz, scaling, features_dc, features_rest, opacity) DO
                # 移除不符合条件的高斯分布
                IF Gaussian.opacity < threshold OR IsTooLarge(Gaussian) THEN
                    RemoveGaussian(Gaussian)
                END IF

                # 密度调整
                IF GradientLargerThanThreshold(Gaussian) THEN
                    IF Gaussian.scaling > size_threshold THEN
                        SplitGaussian(Gaussian)
                    ELSE
                        CloneGaussian(Gaussian)
                    END IF
                END IF
            END FOR
        END IF

        # 更新学习率
        UpdateLearningRate(i)
        i = i + 1
    END WHILE
    RETURN final_image
~~~

## 预备

### 球谐（Spherical Harmonics）

==定义==

![image-20240706195922938](C:\Users\lllly\AppData\Roaming\Typora\typora-user-images\image-20240706195922938.png)

在计算机图形学中，球谐函数常用于近似复杂的光照环境，如环境光照映射。在3D Gaussian Splatting中，如果需要考虑由于环境光照引起的间接效果，可以利用球谐函数对这些光照条件进行有效编码和重建。

## 高斯场景类：`Class GaussianModel`

### `def setup_functions(self)`

~~~python
def setup_functions(self):
        # 协方差矩阵作投影
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation) # 缩放+旋转矩阵
            actual_covariance = L @ L.transpose(1, 2)  # 计算协方差矩阵
            symm = strip_symmetric(actual_covariance) # 提取对称部分（由于数值计算中的不精确性，直接计算的协方差矩阵 actual_covariance 可能会略微偏离对称性。）
            return symm
        
        # 将缩放的激活函数设置为指数函数，逆激活函数设置为对数函数。
        # 输入的缩放值会通过指数函数转换为正值，而对其反向操作则是对数函数。
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid # 不透明度激活函数为sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
~~~

### `def __init__(self, sh_degree : int)`

~~~python
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0 # 目前的球谐阶数，用于近似三维表面或场的分布
        self.max_sh_degree = sh_degree  # 最大可能达到的的球谐阶数
        self._xyz = torch.empty(0) # 每个Gaussian的中心坐标
        self._features_dc = torch.empty(0) # 球谐的直流分量（dc = Direct Current），代表球谐函数的零阶部分，通常是常数项
        self._features_rest = torch.empty(0) # 球谐的其他高阶特征（除了直流分量）
        self._scaling = torch.empty(0) # 缩放参数，用于调整高斯分布的宽度或大小
        self._rotation = torch.empty(0) # 旋转参数（一系列四元数），高斯分布在三维空间中的旋转信息
        self._opacity = torch.empty(0) # 不透明度（经历sigmoid前的）
        self.max_radii2D = torch.empty(0) # 在某个相机视野里出现过的（像平面上的）最大2D半径，详见train.py里面gaussians.max_radii2D[visibility_filter] = ...一行
        self.xyz_gradient_accum = torch.empty(0) # 每个Gaussian的坐标梯度积累，当它太大的时候要对Gaussian进行分裂或复制（论文5.2中提到）
        self.denom = torch.empty(0) # 与累积梯度配合使用，表示统计了多少次累积梯度，算平均梯度时除掉这个（denom = denominator，分母）
        self.optimizer = None # 优化器（论文中采用Adam，见附录B Algorithm 1的伪代码）
        self.percent_dense = 0 # 参与控制Gaussian密集程度的超参数
        self.spatial_lr_scale = 0 # 坐标的学习率要乘上这个，抵消在不同尺度下应用同一个学习率带来的问题
        self.setup_functions()
~~~

































