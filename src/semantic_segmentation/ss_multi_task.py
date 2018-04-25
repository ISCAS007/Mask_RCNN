# -*- coding: utf-8 -*-
"""
motion net structure for semantic segmentation
semantic_segmentation_multi_task with inner loss
"""
import semantic_segmentation_basic
from models import model_basic
import keras
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from dataset_rob2018 import dataset_rob2018
import numpy as np
import os
import keras.backend as K
import tensorflow as tf
from keras.backend import tf as ktf

tf.app.flags.DEFINE_string('app', 'version', 'application name')
tf.app.flags.DEFINE_integer('batch_size', 20, 'Batch size')
tf.app.flags.DEFINE_integer('epoches', 30, 'epoches')

FLAGS = tf.app.flags.FLAGS
tg=tf.data.Dataset.from_generator

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
    
class CustomRegularization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def call(self ,x ,mask=None):
        ld=x[0]
        rd=x[1]
        bce = keras.losses.mse(ld, rd)
        loss2 = K.sum(bce)
        self.add_loss(loss2,x)
        #you can output whatever you need, just update output_shape adequately
        #But this is probably useful
        return bce

    def compute_output_shape(self, input_shape):
        return tuple([None,1])

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

class semantic_segmentation_multi_task(semantic_segmentation_basic.semantic_segmentation_basic):
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
    
    def regen(self,gen):
        assert self.config['reshape_output']==False
        for x,y in gen:
            b,h,w,c=y.shape
            outputs=[y]
            for i in range(self.config['psp_start_layer']):
                cr_i=np.random.rand(b,1)
                outputs.append(cr_i)
        
            yield x,outputs
        
    def train(self):
        if self.config['test_mean_iou'] == True:
            metrics=self.get_metrics(self.config['class_number'])
        else:
            metrics=self.get_metrics()
        losses=['mse']
        for i in range(self.config['psp_start_layer']):
            losses.append(zero_loss)
            
        self.model.compile(loss=losses,
                           optimizer='adam',
                           metrics={'main_output':metrics})

        dataset = self.dataset
        train_main_input_paths, val_main_input_paths = dataset.get_train_val()

        print('train dataset size', len(train_main_input_paths))
        print('test dataset size', len(val_main_input_paths))

        batch_size = self.config['batch_size']
        print('batch size is',batch_size)
        self.model.fit_generator(generator=self.regen(dataset.batch_gen_images(train_main_input_paths, batch_size)),
                                 steps_per_epoch=len(
                                     train_main_input_paths)//batch_size,
                                 epochs=self.config['epoches'],
                                 verbose=1,
                                 callbacks=self.get_callbacks(self.config),
                                 validation_data=self.regen(dataset.batch_gen_images(
                                     val_main_input_paths, batch_size)),
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
        
        merge_layers=[]
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
                
                merge_layers.append(merge_output)
            else:
                main_output = Conv2D(filters=self.config['class_number'],
                                 kernel_size=(3, 3),
                                 activation='softmax',
                                 padding='same',
                                 name='main_output',
                                 data_format=data_format)(merge_output)
                
                seg_outputs = [main_output]
                merge_layers.reverse()
                for idx,layer in enumerate(merge_layers):
                    aux_output=Conv2D(filters=self.config['class_number'],
                                     kernel_size=(3, 3),
                                     activation='softmax',
                                     padding='same',
                                     name='aux_output_'+str(idx),
                                     data_format=data_format)(layer)
                    seg_outputs.append(aux_output)
                
                net_outputs = [main_output]
                for idx,output in enumerate(seg_outputs):
                    if idx==0:
                        continue
                    else:
                        pool_output=keras.layers.AvgPool2D(pool_size=(2,2),strides=(2,2))(seg_outputs[idx-1])
                        cr = CustomRegularization(name='cr_'+str(idx))([pool_output,output])
                        net_outputs.append(cr)
                
                if self.config['reshape_output']:
                    b,h,w,c=main_output.get_shape().as_list()
                    main_output = Reshape((h*w,c), input_shape=(h,w,c))(main_output)
                    
                model = keras.models.Model(inputs=input_layers[0].input,
                                           outputs=net_outputs)

        return model

def main(argv=None):
    config=semantic_segmentation_basic.get_default_config()
#    config['dataset_name']='kitti2015'
    config['dataset_name']='cityscapes'
    config['dataset_train_root']='/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['task']='semantic'
    config['model_name']='semantic_segmentation_psped'
    config['batch_size']=20
    config['encoder']='resnet50'
    config['epoches']=30
    config['test_mean_iou']=True
    
    config['input_shape'] = [224, 224, 3]
    config['reshape_output']=False
    config['psp_start_layer']=3
    config['note']='_'.join([config['encoder'],'psp'+str(config['psp_start_layer'])])
    
    net=semantic_segmentation_multi_task(config)
    if FLAGS.app == 'train':
        net.train()
    elif FLAGS.app=='version':
        net.show_version()

if __name__ == "__main__":
    tf.app.run()