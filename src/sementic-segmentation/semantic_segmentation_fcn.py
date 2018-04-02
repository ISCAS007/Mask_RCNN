# -*- coding: utf-8 -*-
"""
FCN for sementic segmenation
"""
from semantic_segmentation_basic import semantic_segmentation_basic

class semantic_segmentation_fcn(semantic_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)

    def train(self):
        pass

    def get_model(self):
        pass
