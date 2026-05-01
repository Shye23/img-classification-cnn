from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

def get_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    return model