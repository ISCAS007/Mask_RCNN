# -*- coding: utf-8 -*-
"""
FCN for sementic segmenation
"""
import semantic_segmentation_basic
from models.utils.metrics import sparse_accuracy_ignoring_last_label as sparse_acc
import keras
from keras.layers import Conv2D, Conv2DTranspose
from dataset_rob2018 import dataset_rob2018
from models.model_fcn import *


class semantic_segmentation_fcn(semantic_segmentation_basic.semantic_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)
        self.dataset = dataset_rob2018(config)
        self.config['class_number'] = self.dataset.num_classes
        self.model = self.get_model()

#    @staticmethod
#    def get_metrics():
#        metrics = semantic_segmentation_basic.semantic_segmentation_basic.get_metrics()
#        metrics.append(sparse_acc)
#        
#        return metrics

    def train(self):
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=self.get_metrics())

        dataset = self.dataset
        train_main_input_paths, val_main_input_paths = dataset.get_train_val()

        print('train dataset size', len(train_main_input_paths))
        print('test dataset size', len(val_main_input_paths))

        batch_size = self.config['batch_size']
        print('batch size is', batch_size)
        self.model.fit_generator(generator=dataset.batch_gen_images(train_main_input_paths, batch_size),
                                 steps_per_epoch=len(
                                     train_main_input_paths)//batch_size,
                                 epochs=self.config['epoches'],
                                 verbose=1,
                                 callbacks=self.get_callbacks(self.config),
                                 validation_data=dataset.batch_gen_images(
                                     val_main_input_paths, batch_size),
                                 validation_steps=len(val_main_input_paths)//batch_size)

    def get_model(self):
        model_fcn = self.config['model_fcn']
        input_shape = self.config['target_size']+(3,)
        model = globals()[model_fcn](weight_decay=self.config['weight_decay_fcn'],
                                     input_shape=input_shape,
                                     batch_momentum=self.config['batchnorm_momentum'],
                                     classes=self.config['class_number'])

        return model


if __name__ == '__main__':
    config = semantic_segmentation_basic.get_default_config()
    config['dataset_name'] = 'cityscapes'
    config['dataset_train_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['task'] = 'semantic'
    config['model_name'] = 'semantic_segmentation_fcn'
    config['batch_size'] = 20
#    config['decoder']='mobilenet'
    config['epoches'] = 30
    config['test_mean_iou']=True

    for model_fcn in get_model_names():
        config['model_fcn'] = model_fcn
        config['note'] = config['model_fcn']
        if config['model_fcn'] is 'AtrousFCN_Resnet50_16s':
            config['weight_decay_fcn'] = 0.0001/2
        else:
            config['weight_decay_fcn'] = 1e-4
    
        config['batchnorm_momentum'] = 0.95
    
        net = semantic_segmentation_fcn(config)
        net.train()
