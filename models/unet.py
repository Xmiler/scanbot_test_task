from torch import nn
from segmentation_models_pytorch.unet.model import Unet


def get_resnet18_greyscale():
    model = Unet(encoder_name='resnet18')
    model.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model