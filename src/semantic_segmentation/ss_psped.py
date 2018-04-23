# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
motion net structure for semantic segmentation
"""

import semantic_segmentation_basic
from models import model_basic
import keras
from keras.layers import Conv2D, Conv2DTranspose
from dataset_rob2018 import dataset_rob2018
import numpy as np
import os
from keras.backend import tf as ktf

class Interp(keras.layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config
    
class semantic_segmentation_psped(semantic_segmentation_basic.semantic_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)
        self.dataset = dataset_rob2018(config)
        self.config['class_number'] = self.dataset.num_classes
        data_format = self.config['data_format']

        if data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1
        self.model=self.get_model()

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
        print('batch size is',batch_size)
        self.model.fit_generator(generator=dataset.batch_gen_images(train_main_input_paths, batch_size),
                                 steps_per_epoch=len(
                                     train_main_input_paths)//batch_size,
                                 epochs=self.config['epoches'],
                                 verbose=1,
                                 callbacks=self.get_callbacks(self.config),
                                 validation_data=dataset.batch_gen_images(
                                     val_main_input_paths, batch_size),
                                 validation_steps=len(val_main_input_paths)//batch_size)
    @staticmethod
    def interp_block(prev_layer,strides,target_shape):
        x=keras.layers.AveragePooling2D(pool_size=strides,strides=strides)(prev_layer)
        x=keras.layers.Conv2D(filters=512,kernel_size=(1,1),strides=(1,1),use_bias=False)(x)
        x=keras.layers.BatchNormalization(momentum=0.95, epsilon=1e-5)(x)
        x=keras.layers.Activation('relu')(x)
        
#        print('target_shape is',target_shape)
        x=Interp(target_shape)(x)
        
        return x
    
    def build_pyramid_pooling_module(self,prev_layer):
        """Build the Pyramid Pooling Module."""
        # ---PSPNet concat layers with Interpolation
        target_size=(prev_layer.shape[1].value,prev_layer.shape[2].value)
        interp_block1 = self.interp_block(prev_layer, 2, target_size)
        interp_block2 = self.interp_block(prev_layer, 4, target_size)
        interp_block3 = self.interp_block(prev_layer, min(8,target_size[0]), target_size)
    
        # concat all these layers. resulted
        # shape=(1,feature_map_size_x,feature_map_size_y,4096)
        x = keras.layers.Concatenate()([prev_layer,
                             interp_block1,
                             interp_block2,
                             interp_block3])
        return x
        
    def get_model(self):
        # h, w, c = self.config['input_shape']
        backbone_layers = model_basic.get_encoder(self.config)
        merge_type = self.config['merge_type'].lower()
        drop_ratio = self.config['dropout_ratio']
        data_format = self.config['data_format']

        if data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        psp_start_layer=self.config['psp_start_layer']
        input_layers=[]
        for i in range(psp_start_layer):
            input_layers.append(backbone_layers[i])
        
        psp_layer=self.build_pyramid_pooling_module(backbone_layers[psp_start_layer].output)
        input_layers.append(psp_layer)
        layer_depth = len(input_layers)
        deconv_output = None
        model = None
        for i in range(layer_depth):
            if i == 0:
#                print('input_layer class is',input_layers[-i-1].__class__)
#                merge_output = input_layers[-i - 1].output
                merge_output = input_layers[-i-1]
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
                    merge_output = keras.layers.Activation('relu')(merge_output)
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
                outputs = Conv2D(filters=self.config['class_number'],
                                 kernel_size=(3, 3),
                                 activation='softmax',
                                 padding='same',
                                 data_format=data_format)(merge_output)

                model = keras.models.Model(inputs=input_layers[0].input,
                                           outputs=outputs)

        return model

if __name__ == '__main__':
    config=semantic_segmentation_basic.get_default_config()
#    config['dataset_name']='kitti2015'
    config['dataset_name']='cityscapes'
    config['dataset_train_root']='/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['task']='semantic'
    config['model_name']='semantic_segmentation_psped'
    config['batch_size']=20
    config['encoder']='mobilenet'
    config['epoches']=30
    config['test_mean_iou']=True
    
    config['input_shape'] = [224, 224, 3]
    
    config['psp_start_layer']=5
    config['note']='_'.join([config['encoder'],'psp'+str(config['psp_start_layer'])])
    
    app = 'train'
    net=semantic_segmentation_psped(config)
    if app == 'train':
        # train
        for i in range(4):
            config['psp_start_layer']=i+1
            config['note']='_'.join([config['encoder'],'psp'+str(config['psp_start_layer'])])
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
        