"""Base Model for Semantic Segmentation"""

import torch.nn as nn
from core.nn import JPU
from .base_models.resnet import *


__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet18':
            self.pretrained = resnet18(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'resnet34':
            self.pretrained = resnet34(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet152':
            self.pretrained = resnet152(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c1, c2, c3, c4
    def gscnn_base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        y = self.pretrained.bn1(x)
        z = self.pretrained.relu(y)
        c0 = self.pretrained.maxpool(z)
        c1 = self.pretrained.layer1(c0)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
            return c0,c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred
