#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
使用 pytorch-grad-cam 生成热图的示例代码

要使用自己的模型和图片生成热图，需要替换以下内容：

1. 模型加载部分：
   - 替换预训练的 ResNet50 为自己的模型
   - 加载自己的模型权重

2. 目标层选择：
   - 替换 model.layer4[-1] 为模型的最后一个卷积层
   - 不同模型架构的推荐层：
     * ResNet: model.layer4[-1]
     * VGG: model.features[-1]
     * DenseNet: model.features[-1]
     * MobileNet: model.features[-1]
     * ViT: model.blocks[-1].norm1
     * Swin Transformer: model.layers[-1].blocks[-1].norm1

3. 图片加载和预处理：
   - 替换示例图片路径为自己的图片
   - 根据模型调整预处理参数（mean 和 std）

4. 目标类别：
   - 替换类别索引为模型中的实际类别
   - 如果模型不是分类模型，可能需要自定义 targets

5. 输出路径：
   - 修改保存热图的路径
'''

import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    #############################################################
    # 1. 模型加载 - 替换为自己的模型
    #############################################################
    # 示例：使用预训练的 ResNet50 模型
    # 替换为：
    # model = 模型类()
    # model.load_state_dict(torch.load('模型权重路径.pth'))
    model = models.resnet50(weights="IMAGENET1K_V1")  # 示例模型
    model = model.to(device)
    model.eval()  # 设置为评估模式，但不要使用 torch.no_grad()
    
    #############################################################
    # 2. 目标层选择 - 替换为模型的最后一个卷积层
    #############################################################
    # 示例：ResNet50 的最后一个卷积层
    # 替换为：target_layers = [model.最后一个卷积层]
    target_layers = [model.layer4[-1]]  # ResNet50 的最后一个卷积层
    
    #############################################################
    # 3. 图片加载和预处理 - 替换为自己的图片
    #############################################################
    # 示例：加载项目中的示例图片
    # 替换为：image_path = "图片路径.jpg"
    image_path = "./examples/both.png"  # 使用项目中的示例图片
    
    # 加载图片并转换为RGB格式
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR转RGB
    if rgb_img is None:
        print(f"错误：无法加载图片 {image_path}")
        return
    print(f"图片尺寸: {rgb_img.shape}")
    
    # 图像归一化
    rgb_img = np.float32(rgb_img) / 255  # 归一化到0-1范围
    
    # 图像预处理 - 根据模型调整mean和std参数
    # ImageNet预训练模型通常使用以下参数：
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # 如果模型使用不同的预处理，请相应调整
    input_tensor = preprocess_image(rgb_img,
                                   mean=[0.485, 0.456, 0.406],  # 根据模型调整
                                   std=[0.229, 0.224, 0.225])   # 根据模型调整
    input_tensor = input_tensor.to(device)
    
    #############################################################
    # 4. 目标类别 - 替换为模型中的实际类别
    #############################################################
    # 示例：指定ImageNet中的"狗"类别（类别索引243）
    # 替换为：targets = [ClassifierOutputTarget(目标类别索引)]
    # 
    # 如果想解释模型预测的最高概率类别，可以使用：
    # targets = None  # 这将自动选择最高概率的类别
    #
    # 如果模型不是分类模型，可能需要自定义targets
    # 参考pytorch_grad_cam/utils/model_targets.py中的其他实现
    targets = [ClassifierOutputTarget(243)]  # 示例：ImageNet中的"狗"类别
    
    #############################################################
    # 5. 生成热图
    #############################################################
    # 使用 with 语句创建 GradCAM 对象，确保资源正确释放
    # 可选参数：
    # - reshape_transform: 对于特殊架构（如Transformer）可能需要提供
    # - aug_smooth=True: 使用测试时增强来平滑CAM结果
    # - eigen_smooth=True: 使用主成分分析来平滑结果
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # 生成 CAM
        # 可选参数：
        # - aug_smooth=True: 使用测试时增强来平滑结果
        # - eigen_smooth=True: 使用主成分分析来平滑结果
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # 取第一个批次的结果
        
        # 将 CAM 叠加到原始图像上
        # use_rgb=True 表示输入图像是RGB格式
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        #############################################################
        # 6. 保存结果 - 修改为想要的输出路径
        #############################################################
        output_path = "./dog_gradcam.jpg"  # 替换为想要的输出路径
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        print(f"热图已保存到: {output_path}")
    
    #############################################################
    # 7. 第二个示例 - 可以根据需要移除或修改
    #############################################################
    # 这部分展示了如何对另一个图片/类别生成热图
    # 在实际应用中，可以根据需要修改或移除此部分
    
    # 加载另一个示例图片
    image_path = "./examples/both.png"  # 替换为另一个图片
    print(f"加载第二个示例图片: {image_path}")
    second_img = cv2.imread(image_path, 1)
    if second_img is None:
        print(f"错误：无法加载图片 {image_path}")
        return
    print(f"图片尺寸: {second_img.shape}")
    rgb_img = second_img[:, :, ::-1]  # BGR转RGB
    rgb_img = np.float32(rgb_img) / 255
    
    # 预处理图像
    input_tensor = preprocess_image(rgb_img,
                                   mean=[0.485, 0.456, 0.406],  # 根据模型调整
                                   std=[0.229, 0.224, 0.225])   # 根据模型调整
    input_tensor = input_tensor.to(device)
    
    # 指定另一个目标类别
    # 示例：ImageNet中的"猫"类别（类别索引282）
    targets = [ClassifierOutputTarget(282)]  # 替换为另一个目标类别
    
    # 再次使用 with 语句创建新的 GradCAM 对象
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # 生成 CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 将 CAM 叠加到原始图像上
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 保存结果
        output_path = "./cat_gradcam.jpg"  # 替换为想要的输出路径
        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        print(f"热图已保存到: {output_path}")

#############################################################
# 其他可能的高级用法：
#
# 1. 使用不同的CAM方法：
#    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, 
#                             AblationCAM, XGradCAM, EigenCAM, LayerCAM
#
# 2. 对于非分类任务（如目标检测、语义分割）：
#    参考文档：https://jacobgil.github.io/pytorch-gradcam-book
#
# 3. 对于特殊架构（如Transformer）：
#    需要提供reshape_transform参数
#
# 4. 评估CAM的质量：
#    使用pytorch_grad_cam.metrics包中的工具
#############################################################

if __name__ == "__main__":
    main()
