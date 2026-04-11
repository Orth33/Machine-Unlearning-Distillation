import torch.nn as nn
from torchvision.models import efficientnet_b0 as tv_efficientnet_b0


def efficientnet_b0(num_classes=1000, imagenet=False, pretrained=False):
    use_pretrained = imagenet or pretrained
    try:
        from torchvision.models import EfficientNet_B0_Weights

        weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
        model = tv_efficientnet_b0(weights=weights)
    except ImportError:
        model = tv_efficientnet_b0(pretrained=use_pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
