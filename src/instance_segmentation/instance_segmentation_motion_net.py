# -*- coding: utf-8 -*-
"""
motion net structure for instance segmentation
"""

import os
import sys
lib_path = os.path.abspath(os.path.join('../semantic_segmentation'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)
import basic_net
#from ..semantic_segmentation.models import model_basic
import keras
from keras.layers import Conv2D, Conv2DTranspose
#from ..semantic_segmentation.dataset_rob2018 import dataset_rob2018
from models import model_basic
from dataset_rob2018 import dataset_rob2018, show_images
import numpy as np
import matplotlib.pyplot as plt


class is_simple(basic_net.instance_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)
        self.dataset = dataset_rob2018(config)
        self.config['class_number'] = self.dataset.num_classes
        self.model = self.get_model()

    def get_model(self):
        input_layers = model_basic.get_encoder(self.config)
        merge_type = self.config['merge_type'].lower()
        drop_ratio = self.config['dropout_ratio']
        data_format = self.config['data_format']

        if data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        layer_depth = len(input_layers)
        deconv_output = None
        model = None
        for i in range(layer_depth):
            if i == 0:
                merge_output = input_layers[-i - 1].output
                merge_output = keras.layers.Dropout(
                    rate=drop_ratio)(merge_output)
            else:
                assert deconv_output is not None
                merge_output = self.merge(inputs=[input_layers[-i - 1].output,
                                                  deconv_output],
                                          mode=merge_type)

                merge_output = keras.layers.Dropout(
                    rate=drop_ratio)(merge_output)

                if merge_type == 'concat':
                    merge_output = Conv2D(filters=input_layers[-i - 1].output_shape[channel_axis],
                                          kernel_size=(3, 3),
                                          activation=None,
                                          padding='same',
                                          data_format=data_format)(merge_output)
                    merge_output = keras.layers.BatchNormalization()(merge_output)
                    merge_output = keras.layers.Activation(
                        'relu')(merge_output)
                    merge_output = keras.layers.Dropout(
                        rate=drop_ratio)(merge_output)

            if i < layer_depth - 1:
                # input_filters=merge_output.shape[channel_axis].value
                # target_filters=input_layers[-i-2].output_shape[channel_axis]
                deconv_output = Conv2DTranspose(filters=input_layers[-i - 2].output_shape[channel_axis],
                                                kernel_size=(2, 2),
                                                strides=(2, 2),
                                                activation=None,
                                                padding='same',
                                                data_format=data_format)(merge_output)
                deconv_output = keras.layers.BatchNormalization()(deconv_output)
                deconv_output = keras.layers.Activation('relu')(deconv_output)
                deconv_output = keras.layers.Dropout(
                    rate=drop_ratio)(deconv_output)

            else:
                y_category = Conv2D(filters=self.config['class_number'],
                                    kernel_size=(3, 3),
                                    activation='softmax',
                                    padding='same',
                                    data_format=data_format,
                                    name='y_category')(merge_output)

                y_offset = Conv2D(filters=2,
                                  kernel_size=(3, 3),
                                  activation='tanh',
                                  padding='same',
                                  data_format=data_format,
                                  name='y_offset')(merge_output)

                if self.config['sub_version'] == 'two_stage':
                    y_center = Conv2D(filters=1,
                                      kernel_size=(3, 3),
                                      activation='sigmoid',
                                      padding='same',
                                      data_format=data_format,
                                      name='y_center')(y_offset)
                else:
                    y_center = Conv2D(filters=1,
                                      kernel_size=(3, 3),
                                      activation='sigmoid',
                                      padding='same',
                                      data_format=data_format,
                                      name='y_center')(merge_output)

                model = keras.models.Model(inputs=input_layers[0].input,
                                           outputs=[y_category, y_offset, y_center])

        return model

    def train(self):
        metrics = self.get_metrics(self.config['class_number'])

        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=metrics)

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

    def showcase(self, n=3):
        weights_file = self.get_weight_path(
            self.config['weight_load_dir_or_file'])
        if weights_file is None:
            return 0

        self.model.load_weights(weights_file)

        count = 0
        for imgs,paths,x in self.dataset.batch_gen_inputs_for_test(self.config['batch_size'], shuffle=True):
            y_category, y_offset, y_center = self.model.predict(x)
            for img,p, a, b, c in zip(imgs,paths, y_category, y_offset, y_center):
                instance_img, xy_local_max = self.get_instance_segmentation_by_kmeans(
                    a, b, c)
                instance_img = instance_img.astype(np.int64)
                semantic_img = np.argmax(a, axis=-1).astype(np.int64)
                offset_x=(b[:,:,0]+1.0)*0.5
                offset_y=(b[:,:,1]+1.0)*0.5
                if len(xy_local_max) > 0:
                    h, w = instance_img.shape
                    center_img = np.zeros((h, w), np.uint8)
    
                    d = 5
                    for y, x in xy_local_max:
                        for i in range(x-d, x+d):
                            for j in range(y-d, y+d):
                                if i >= 0 and j >= 0 and i < w and j < h:
                                    center_img[j, i] = 255
                else:
                    center_img=c.reshape(self.config['target_size'])
                
                print('center_img shape',center_img.shape,center_img.dtype)
                print('unique center_img',np.unique(center_img))
                img=img.astype(np.uint8)
                show_images([img, instance_img, semantic_img, center_img,offset_x,offset_y],
                            ['image_2', 'instance', 'semantic', 'center','offset_x','offset_y'])
                count = count+1
                if count >= n:
                    break

            if count >= n:
                break

        return 1


if __name__ == '__main__':
    config = basic_net.get_default_config()
#    config['dataset_name']='kitti2015'
    config['dataset_name'] = 'cityscapes'
    config['dataset_train_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['dataset_test_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/test'
    config['task'] = 'instance'
    config['model_name'] = 'is_simple'
    config['batch_size'] = 20
    config['encoder'] = 'mobilenet'
    config['epoches'] = 30
    config['category_weight'] = 100
    config['center_threshold'] = 0.001
    config['center_min_gap'] = 5

    app = 'showcase'
    for sub_version in ['one_stage', 'two_stage']:
        config['sub_version'] == sub_version
        config['note'] = sub_version

        if app == 'train':
            # train
            net = is_simple(config)
            net.train()
        elif app == 'showcase':
            # showcase
            weight_load_dir_or_file = os.path.join(
                '/home/yzbx/tmp/logs/instance/cityscapes/is_simple', sub_version)
            config['weight_load_dir_or_file'] = weight_load_dir_or_file
            net = is_simple(config)
            net.showcase(n=3)
