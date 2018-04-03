# -*- coding: utf-8 -*-
"""
http://www.robustvision.net/
Robust Vision Challenge 2018 Workshop at CVPR 2018 in Salt Lake City
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import math
import keras

def get_rob2018_dataset(config):
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
        print('target dataset not in rob2018',config['dataset'])
        assert False

def show_images(images,titles,fignum=1):
    n=len(images)
    x=y=math.ceil(math.sqrt(n))
    plt.figure(fignum)
    for idx,img in enumerate(images):
        plt.subplot(x,y,idx+1)
        plt.imshow(img)
        plt.title(titles[idx])
    plt.show()

def preprocess_image_bgr(x):
    """
    batch_size,height,width,channel=x.shape
    scale x from [0,255] to [-1,1]
    """
    assert x.ndim==4
    x=2*(x.astype(np.float32)/255.0-0.5)
    
    return x
    
def preprocess_image_mask(y,num_classes):
    """
    batch_size,height,width=y.shape
    """
    assert y.ndim==3
    
    b,h,w=y.shape
    y_category=keras.utils.to_categorical(y,num_classes)
    y_category = y_category.reshape([b, h, w, num_classes])
    return y_category
    
class dataset_rob2018():
    def __init__(self,config):
        self.dataset_name=config['dataset_name']
        self.dataset_train_root=config['dataset_train_root']
        self.task=config['task']
        self.target_size=config['target_size']
        self.num_classes={}
        self.num_classes['kitti2015']=34
        assert self.task in ['instance','semantic']
        
    def batch_gen_images(self,image_2_list,batch_size):
        tasks=['image_2',self.task]
        for image_2_batch,task_batch in self.batch_gen_paths(image_2_list,batch_size):
            images_batchs=[]
            for _list,_task in zip([image_2_batch,task_batch],tasks):
                img_list=[]
                for _f in _list:
                    assert os.path.exists(_f)
                    if _task=='image_2':
                        img=cv2.imread(_f)
                    else:
                        img=cv2.imread(_f,cv2.IMREAD_GRAYSCALE)
                    
                    img_list.append(img)
                
                if _task=='image_2':
                    x=[cv2.resize(img,self.target_size,interpolation=cv2.INTER_LINEAR) for img in img_list]
                    x=np.asarray(x,dtype=np.float32)
                    images_batchs.append(preprocess_image_bgr(x))
                elif _task=='semantic':
                    y=[cv2.resize(img,self.target_size,interpolation=cv2.INTER_NEAREST) for img in img_list]
                    y=np.asarray(y,dtype=np.float32)
                    images_batchs.append(preprocess_image_mask(y,self.num_classes[self.dataset_name]))
                else:
                    print('undefined precessing for task',_task)
                    assert False
            
            assert len(images_batchs)==2
            assert images_batchs[0].ndim==4
            assert images_batchs[1].ndim==4
            yield images_batchs
            
    def batch_gen_paths(self,image_2_list,batch_size):
        """
        input relative image_2 list
        output batch path
        """
        n=len(image_2_list)
        while True:
            random.shuffle(image_2_list)
            paths=[]
            for task in ['image_2',self.task]:
                task_root=os.path.join(self.dataset_train_root,task)
                abs_list=[os.path.join(task_root,f) for f in image_2_list]
                paths.append(abs_list)
            
            for idx in range(0,n,batch_size):
                if idx+batch_size<=n:
                    yield [paths[0][idx:idx+batch_size],paths[1][idx:idx+batch_size]]
    
    def get_train_val(self):
        train_txt=os.path.join(self.dataset_train_root,'train.txt')
        val_txt=os.path.join(self.dataset_train_root,'val.txt')
        
        lists=[]
        if os.path.exists(train_txt) and os.path.exists(val_txt):
            for txt in [train_txt,val_txt]:
                file=open(txt,'r')
                lines=file.readlines()
                _list=[line.strip() for line in lines]
                lists.append(_list)
                file.close()
        else:
            print('warning: cannot find train txt and val txt',train_txt,val_txt)
            print('create them by random')
            image_dir=os.path.join(self.dataset_train_root,'image_2')
            _list=os.listdir(image_dir)
            img_suffix=('png','jpg','jpeg','bmp')
            img_list=[f for f in _list if f.lower().endswith(img_suffix)]
            random.shuffle(img_list)
            list_size=len(img_list)
            train_size=list_size*3//5
            lists.append(img_list[:train_size])
            lists.append(img_list[train_size:])
            for _list,_file in zip(lists,['train.txt','val.txt']):
                file_path=os.path.join(self.dataset_train_root,_file)
                file=open(file_path,'w')
                for line in _list:
                    file.write(line+'\n')
                file.close()
            
        return lists

    def view(self):
        """
        xxxdataset
        ├── testing
        │   └── image_2
        └── training
            ├── image_2
            ├── instance
            ├── semantic
            └── semantic_rgb
        """
        
        f=None
        for _dir in os.listdir(self.dataset_train_root):
            sub_dir=os.path.join(self.dataset_train_root,_dir)
            if not os.path.isdir(sub_dir):
                continue
            
            f_list=os.listdir(sub_dir)
            if len(f_list)==0:
                print('warning: empty dir',sub_dir)
                continue
                
            f=random.choice(f_list)
            break
        
        imgs=[]
        titles=[]
        if f is not None:
            for _dir in os.listdir(self.dataset_train_root):
                img_file=os.path.join(self.dataset_train_root,_dir,f)
                if os.path.exists(img_file):
                    img=cv2.imread(img_file)
                    imgs.append(img)
                    titles.append(_dir)
                else:
                    print('cannot find image file',img_file)
        
        show_images(imgs,titles)
    
    def get_class_range(self):
        """
        max class id is 33
        min class id is 0
        """
        sub_dir=os.path.join(self.dataset_train_root,self.task)
        f_list=os.listdir(sub_dir)
        max_class=0
        min_class=100
        for f in f_list:
            img_file=os.path.join(sub_dir,f)
            img=cv2.imread(img_file)
            max_class=max(np.max(img),max_class)
            min_class=min(np.min(img),min_class)
            
        print('max class id is',max_class)
        print('min class id is',min_class)
    
if __name__ == '__main__':
    config={}
    config['dataset']='kitti2015'
    config['dataset_train_root']='/media/sdb/CVDataset/ObjectSegmentation/Kitti2015_archives/data_semantics/training'
    config['task']='semantic'
    
    dataset=dataset_rob2018(config)
    dataset.get_class_range()
#    dataset.view()
    
    train_list,val_list=dataset.get_train_val()
    dataset.batch_gen_images(train_list,2)