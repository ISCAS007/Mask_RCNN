# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
basic class for instance segmentation
"""
import os
import time
import json
import keras
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max

lib_path=os.path.abspath(os.path.join('../semantic_segmentation'))
sys.path.append(lib_path)
#from ..semantic_segmentation import metrics_fmeasure
#from ..semantic_segmentation import metrics_iou
import metrics_fmeasure
import metrics_iou
import glob

def get_default_config():
    config = {}
    config = {}
    config['encoder'] = 'vgg16'
    # model input size, use None for convinience, but in fact use 224x224
    config['input_shape'] = [None, None, None]
    # model output size, for benchmark, we need post-processing
    config['target_size'] = (224, 224)
    config['optimizer'] = 'adam'
    config['learning_rate'] = 0.01
    config['log_dir'] = os.path.join(
        os.getenv('HOME'), 'tmp', 'logs', 'instance')
    config['checkpoint_dir'] = config['log_dir']
    config['model_name'] = 'instance_segmentation_motion_net'
    config['note'] = 'default'
    config['epoches'] = 30
    config['batch_size'] = 32
    config['merge_type'] = 'concat'
    config['dropout_ratio'] = 0.1
    config['data_format'] = 'channels_last'
    config['sub_version'] = ''

    # dataset
    config['dataset_name'] = 'kitti2015'
    config['dataset_train_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    config['task'] = 'instance'

    return config


def get_dataset(config):
    """
    [Robust Vision Challenge 2018](http://www.robustvision.net/)
    [devkit](http://cvlibs.net:3000/ageiger/rob_devkit.git)
    dataset=[cityscapes,kitti2015,scannet,wilddash]
    """
    if config['dataset'] == 'cityscapes':
        pass
    elif config['dataset'] == 'kitti2015':
        pass
    elif config['dataset'] == 'scannet':
        pass
    elif config['dataset'] == 'wilddash':
        pass
    else:
        pass

def show_result(y_category,y_offset,y_center):
    """
    y_category [h,w,class_propabilities]
    y_offset [h,w,2] for x and y
    y_center [h,w] for y_center(x,y)=1 if (x,y) is center points
    """
    assert y_category.ndim==3
    h,w,c=y_category.shape
#    image_category=np.zeros((h,w),dtype=np.uint8)
    image_category=np.argmax(y_category,axis=-1).astype(np.uint8)
    image_offset=np.zeros((h,w,3),dtype=np.float32)
    image_offset[:,:,0:2]=(y_offset+1)/2.0
    image_center=y_center
    
    fig,axs=plt.subplots(2,2)
    axs[0,0].imshow(image_category)
    axs[0,0].title('category')
    axs[0,1].imshow(image_offset)
    axs[0,1].title('offset')
    axs[1,0].imshow(image_center)
    axs[1,0].title('center')
    
    plt.show()
def get_newest_file(files):
        t=0
        newest_file=None
        for full_f in files:
            if os.path.isfile(full_f):
                file_create_time = os.path.getctime(full_f)
                if file_create_time > t:
                    t = file_create_time
                    newest_file = full_f
        
        return newest_file
    
class instance_segmentation_basic():
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)

    @staticmethod
    def get_backbones():
        return ['vgg16', 'vgg19', 'mobilenet']

    @staticmethod
    def get_datasets():
        return ['cityscapes', 'kitti2015', 'scannet', 'wilddash']

    @staticmethod
    def get_models():
        return ['is_simple']

    @staticmethod
    def get_version(class_name):
        version_dict = {}
        version_dict['instance_segmentation_basic'] = "0.0"
        version_dict['is_simple'] = '0.1'
        return version_dict[class_name]

    @staticmethod
    def get_callbacks(config):
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())

        log_dir = os.path.join(config['log_dir'],
                               config['dataset_name'],
                               config['model_name'],
                               config['note'],
                               time_str)

        checkpoint_dir = os.path.join(config['checkpoint_dir'],
                                      config['dataset_name'],
                                      config['model_name'],
                                      config['note'],
                                      time_str)

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # write config to config.txt
        config_path = os.path.join(checkpoint_dir, 'config.txt')
        config_file = open(config_path, 'w')
        json.dump(config, config_file, sort_keys=True)

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

        return [tensorboard_log, checkpoint]

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

    @staticmethod
    def get_metrics(class_num=None):
        if class_num is None:
            return {'y_category': [metrics_fmeasure.precision,
                                   metrics_fmeasure.recall,
                                   metrics_fmeasure.fmeasure],
                    'y_offset': 'mse',
                    'y_center': 'mse'}
        else:
            tf_iou_metric = metrics_iou.slow_mean_iou(class_num)

            return {'y_category': [tf_iou_metric,
                                   metrics_fmeasure.precision,
                                   metrics_fmeasure.recall,
                                   metrics_fmeasure.fmeasure],
                    'y_offset': 'mse',
                    'y_center': 'mse'}
            
    
    def get_instance_segmentation_by_kmeans(self,y_category,y_offset,y_center):
        assert y_offset.ndim==3
        h,w,c=y_offset.shape
        
        y_category=np.argmax(y_category,axis=-1)
        assert h==y_category.shape[0]
        assert w==y_category.shape[1]
        
        category_weight=self.config['category_weight']
        center_threshold=self.config['center_threshold']
        center_min_gap=self.config['center_min_gap']
        
        data=[]
        for i in range(h):
            for j in range(w):
                label=category_weight*y_category[i,j]
                x_center=j+y_offset[i,j,0]
                y_center=i+y_offset[i,j,1]
                data.append([label,i,j,x_center,y_center])
        
        xy_local_max=peak_local_max(y_center,min_distance=center_min_gap,threshold_abs=center_threshold)
        if len(xy_local_max)==0:
            print('warning: xy_local_max is empty')
            n_clusters=len(np.unique(y_category))
        else:
            n_clusters=len(xy_local_max)
        
        data=np.array(data,dtype=np.float32)
        kmeans=KMeans(n_clusters=n_clusters).fit(data)
        instance_img=kmeans.labels_.reshape(h,w)
        
        return instance_img,xy_local_max
    
    
    
    @staticmethod
    def get_weight_path(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            files = glob.glob(os.path.join(checkpoint_path,'*.hdf5'))
            if len(files)==0:
                files = glob.glob(os.path.join(checkpoint_path,'*','*.hdf5'))
            newest_file = get_newest_file(files)

            if newest_file is None:
                print('no weight file find in path',checkpoint_path)
                return None
            else:
                return newest_file
        elif os.path.isfile(checkpoint_path):
            return checkpoint_path
        else:
            print('checkpoint_path is not a dir or a file!!!')
            sys.exit(-1)