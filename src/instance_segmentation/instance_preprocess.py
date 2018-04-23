# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def get_center_map(img):
    """
    offset image may cannot save as normal image,use pickle instead
    """
    h,w=img.shape
    x=np.arange(0,w)
    y=np.arange(0,h)
    ones_x=[1 for i in range(h)]
    ones_y=[1 for i in range(w)]
    x_img=np.array(np.matrix([ones_x]).transpose()*np.matrix(x),dtype=np.float32)
    y_img=np.array(np.matrix(y).transpose()*np.matrix([ones_y]),dtype=np.float32)
    
    instances=np.unique(img)
    center_points=[]
    offset_img=np.zeros((h,w,2),dtype=np.float32)
    for obj in instances:
        mask_img=(img==obj).astype(np.uint8)
        m=cv2.moments(mask_img,binaryImage=True)
        mean_x=m['m10']/m['m00']
        mean_y=m['m01']/m['m00']
        center_points.append((mean_x,mean_y))
        
#        print(obj,mean_x,mean_y)
        offset_img[:,:,0]+=(x_img-mean_x)*mask_img
        offset_img[:,:,1]+=(y_img-mean_y)*mask_img
#        print(np.unique(offset_img))
    data={'shape':img.shape,
          'offset_image':offset_img,
          'center_points':center_points,
          'instances':instances}
    
    return data

def show_center_map(data,instance_img):
    img=data['offset_image']
    h,w=data['shape']
    img[:,:,0]=img[:,:,0]+w
    img[:,:,1]=img[:,:,1]+h
    offset_image=np.zeros((h,w,3),dtype=np.float32)
    offset_image[:,:,0]=img[:,:,0]/(2.0*w)
    offset_image[:,:,1]=img[:,:,1]/(2.0*h)
    plt.subplot(1,2,1)
    plt.imshow(offset_image)
    plt.title('offset image')
    plt.subplot(1,2,2)
    plt.imshow(instance_img)
    plt.title('instance image')
    plt.show()
    
if __name__ == '__main__':
    root='/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    image_dir = os.path.join(root, 'instance')
    _list = os.listdir(image_dir)
#            print('_list size is',len(_list))
    img_suffix = ('png', 'jpg', 'jpeg', 'bmp')
    img_list = [f for f in _list if f.lower().endswith(img_suffix)]
    
    pickle_dir=os.path.join(root,'centered_data')
    os.makedirs(pickle_dir,exist_ok=True)
    
    n=len(img_list)
    for idx,f in enumerate(img_list):
        img_path=os.path.join(image_dir,f)
        f_pickle=f.split('.')[0]+'.pkl'
        pickle_path=os.path.join(pickle_dir,f_pickle)
        
        if os.path.exists(pickle_path):
            print('file exists, exit...',pickle_path)
            break
        
        img=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        assert img.dtype == 'uint16'
        data=get_center_map(img)
        output=open(pickle_path,'wb')
        pickle.dump(data,output)
        output.close()
        
        print('%d/%d(%0.2f%%) save to %s'%(idx,n,(100.0*idx/n),pickle_path))
#        show_center_map(data,img)
#        break

    