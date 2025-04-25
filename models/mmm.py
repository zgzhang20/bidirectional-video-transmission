import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # 准备预训练模型（Mask R-CNN模型）
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取预训练模型的打分模块的输入维度，也就是特征提取模块的输出维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 将预训练模型的预测部分修改为FastR-CNN的预测部分（Fast R-CNN与Faster R-CNN的预测部分相同）
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 获取预训练模型中像素级别预测器的输入维度
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    num_classes = 2

    # 使用自己的参数生成Mask预测器替换预训练模型中的Mask预测器部分
    # 三个参数，输入维度，中间层维度，输出维度（类别个数）
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

