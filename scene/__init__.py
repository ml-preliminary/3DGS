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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化 Scene 类，负责加载和管理场景数据。
        :param args: 包含模型路径和源路径等参数的 ModelParams 对象。
        :param gaussians: 高斯模型对象。
        :param load_iteration: 要加载的特定迭代，如果为 None，则不加载。
        :param shuffle: 是否打乱相机顺序。
        :param resolution_scales: 分辨率缩放比例列表。
        """
        self.model_path = args.model_path  # 模型路径
        self.loaded_iter = None  # 加载的迭代次数
        self.gaussians = gaussians  # 高斯模型

        # 如果指定了加载迭代次数，则加载对应的迭代
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}  # 训练相机字典
        self.test_cameras = {}  # 测试相机字典

        # 加载场景信息，根据源路径判断场景类型
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果未加载特定迭代，则初始化场景
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 根据参数决定是否打乱相机顺序
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # 训练相机随机打乱
            random.shuffle(scene_info.test_cameras)  # 测试相机随机打乱

        self.cameras_extent = scene_info.nerf_normalization["radius"]  # 场景归一化半径

        # 加载训练和测试相机
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 如果加载了特定迭代，则加载对应的点云
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)  # 根据点云创建高斯模型

    def save(self, iteration):
        """
        保存当前迭代的点云。
        :param iteration: 当前迭代次数。
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        """
        获取指定缩放比例的训练相机列表。
        :param scale: 缩放比例。
        :return: 训练相机列表。
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        """
        获取指定缩放比例的测试相机列表。
        :param scale: 缩放比例。
        :return: 测试相机列表。
        """
        return self.test_cameras[scale]
