#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import numpy as np

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        """
        初始化 Camera 类，负责管理相机的内外参和图像数据。
        :param colmap_id: Colmap 中的相机 ID。
        :param R: 旋转矩阵。
        :param T: 平移矩阵。
        :param FoVx: 水平视场角。
        :param FoVy: 垂直视场角。
        :param image: 图像数据。
        :param gt_alpha_mask: 真实图像的 alpha 蒙版。
        :param image_name: 图像名称。
        :param uid: 相机唯一标识符。
        :param trans: 变换矩阵。
        :param scale: 缩放比例。
        :param data_device: 数据设备（默认为 "cuda"）。
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # 设置数据设备
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # 处理原始图像数据
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 应用 alpha 蒙版
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # 计算视图变换矩阵和投影矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        """
        初始化 MiniCam 类，负责管理小型相机的基本参数和变换矩阵。
        :param width: 图像宽度。
        :param height: 图像高度。
        :param fovy: 垂直视场角。
        :param fovx: 水平视场角。
        :param znear: 近剪裁面。
        :param zfar: 远剪裁面。
        :param world_view_transform: 世界到视图的变换矩阵。
        :param full_proj_transform: 完整的投影变换矩阵。
        """
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

