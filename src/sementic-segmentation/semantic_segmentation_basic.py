# -*- coding: utf-8 -*-
"""
basic class for semantic segmentation
"""
import os
import time
import json
import keras
import dataset

def get_default_config():
    config={}
    
    return config

def get_dataset(config):
    """
    [Robust Vision Challenge 2018](http://www.robustvision.net/)
    [devkit](http://cvlibs.net:3000/ageiger/rob_devkit.git)
    dataset=[cityscapes,kitti2015,scannet,wilddash]
    """
    if config['dataset']=='cityscapes':
        pass
    elif config['dataset']=='kitti2015':
        pass
    elif config['dataset']=='scannet':
        pass
    elif config['dataset']=='wilddash':
        pass
    else:
        pass
class semantic_segmentation_basic():
    def __init__(self):
        self.name=self.__class__.__name__
        self.version=self.get_version(self.name)
        
    @staticmethod
    def get_backbones():
        return ['vgg16','vgg19','mobilenet']
    
    @staticmethod
    def get_datasets():
        return ['cdnet2014','bmcnet']
    
    @staticmethod
    def get_models():
        return ['MotionNet_TwoOutput']

    @staticmethod
    def get_version(class_name):
        version_dict={}
        version_dict['semantic_segmentation_basic']="0.0"
        version_dict['semantic_segmentation_fcn']="0.1"
        version_dict['semantic_segmentation_unet']="0.2"
        version_dict['semantic_segmentation_skip_net']="0.3"
        version_dict['semantic_segmentation_dilation_net']="0.4"
        version_dict['semantic_segmentation_segnet']="0.5"
        version_dict['semantic_segmentation_enet']="0.6"
        return version_dict[class_name]

    @staticmethod
    def get_callbacks(config):
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        
        log_dir=os.path.join(config['log_dir'],
                             config['dataset_name'],
                             config['model_name'],
                             config['note'],
                             time_str)
        
        checkpoint_dir=os.path.join(config['checkpoint_dir'],
                                     config['dataset_name'],
                                     config['model_name'],
                                     config['note'],
                                     time_str)
        
        os.makedirs(log_dir,exist_ok=True)
        os.makedirs(checkpoint_dir,exist_ok=True)
        
        # write config to config.txt
        config_path=os.path.join(checkpoint_dir,'config.txt')
        config_file=open(config_path,'w')
        json.dump(config,config_file,sort_keys=True)
        
        tensorboard_log = keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint = keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=False, 
                    save_weights_only=False, 
                    mode='auto',
                    period=config['epoches']//3)
        
        return [tensorboard_log,checkpoint]

    def show_version(self):
        print('config is'+'*'*100+'\n')
        json.dumps(self.config)
        self.model.summary()
        
    @staticmethod
    def merge(inputs, mode):
        if mode == "add" or mode == 'sum':
            return keras.layers.add(inputs)
        elif mode == "subtract":
            return keras.layers.subtract(inputs)
        elif mode == "multiply":
            return keras.layers.multiply(inputs)
        elif mode == "max":
            return keras.layers.maximum(inputs)
        elif mode == "min":
            return keras.layers.minimum(inputs)
        elif mode == "concatenate" or mode == 'concat':
            return keras.layers.concatenate(inputs)
        else:
            print('warning: unknown merge type %s' % mode)
            assert False