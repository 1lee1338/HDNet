"""Model store which handles pretrained models """

from .HDNet import *

__all__ = [ 'get_segmentation_model']

def get_segmentation_model(model, **kwargs):
    models = {
        'hdnet':get_hdnet,
    }
    return models[model](**kwargs)
