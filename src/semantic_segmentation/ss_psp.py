#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:12:32 2018

@author: yzbx
Total params: 65,437,056
Trainable params: 65,378,816
Non-trainable params: 58,240
"""

import semantic_segmentation_basic
from models.utils.metrics import sparse_accuracy_ignoring_last_label as sparse_acc
import keras
from keras.layers import Conv2D, Conv2DTranspose
from dataset_rob2018 import dataset_rob2018
from models.PSPNet import PSPNet50

class semantic_segmentation_psp(semantic_segmentation_basic.semantic_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)
        self.dataset = dataset_rob2018(config)
        self.config['class_number'] = self.dataset.num_classes
        self.model = self.get_model()
        print(self.model.summary())
#    @staticmethod
#    def get_metrics():
#        metrics = semantic_segmentation_basic.semantic_segmentation_basic.get_metrics()
#        metrics.append(sparse_acc)
#
#        return metrics

    def train(self):
        if self.config['test_mean_iou'] == True:
            metrics=self.get_metrics(self.config['class_number'])
        else:
            metrics=self.get_metrics()
        
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=metrics)

        dataset = self.dataset
        train_main_input_paths, val_main_input_paths = dataset.get_train_val()

        print('train dataset size', len(train_main_input_paths))
        print('test dataset size', len(val_main_input_paths))

        batch_size = self.config['batch_size']
        print('batch size is', batch_size)
        
#        trainimg_dir=valimg_dir=os.path.join(self.config['dataset_train_root'],'image_2')
#        trainmsk_dir=valmsk_dir=os.path.join(self.config['dataset_train_root'],'semantic')
#        # set generater
#        train_gen = data_gen_small(
#                trainimg_dir,
#                trainmsk_dir,
#                train_main_input_paths,
#                batch_size,
#                self.config['input_shape'],
#                self.config['class_number'])
#        val_gen = data_gen_small(
#                valimg_dir,
#                valmsk_dir,
#                val_main_input_paths,
#                batch_size,
#                self.config['input_shape'],
#                self.config['class_number'])
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
        input_shape = self.config['target_size']+(3,)
        model = PSPNet50(input_shape=input_shape,
                         n_labels=self.config['class_number'],
                         reshape_output=self.config['reshape_output'])

        return model


if __name__ == '__main__':
    config = semantic_segmentation_basic.get_default_config()
    config['dataset_name'] = 'cityscapes'
    config['dataset_train_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['task'] = 'semantic'
    config['model_name'] = 'semantic_segmentation_psp'
    config['batch_size'] = 4
#    config['decoder']='mobilenet'
    config['epoches'] = 30
    config['test_mean_iou'] = True
    # model input size, use None for convinience, but in fact use 224x224
    config['input_shape'] = [512, 512, 3]
    # model output size, for benchmark, we need post-processing
    config['target_size'] = (512, 512)
    
    config['note'] = 'psp'
    net = semantic_segmentation_psp(config)
    
    app = 'version'
    if app == 'train':
        # train
        net.train()
    elif app == 'showcase':
        # showcase
        weight_load_dir_or_file = checkpoint_dir = os.path.join(config['checkpoint_dir'],
                                  config['dataset_name'],
                                  config['model_name'],
                                  config['note'])
        config['weight_load_dir_or_file'] = weight_load_dir_or_file
        net.showcase(n=3)
    elif app == 'version':
        net.show_version()
