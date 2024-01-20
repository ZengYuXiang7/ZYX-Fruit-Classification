import torch.nn as nn
import torchvision.models as models


def get_model(model_name, num_classes):
    # 加载预训练模型
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=True)
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(pretrained=True)
    elif model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(pretrained=True)
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 修改模型的尾部全连接层
    if 'resnet' in model_name:
        # 如果模型是 ResNet
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif 'densenet' in model_name:
        # 如果模型是 DenseNet
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif 'efficientnet' in model_name or 'mobilenet' in model_name:
        # 如果模型是 EfficientNet 或 MobileNet
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model
