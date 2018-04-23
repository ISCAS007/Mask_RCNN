# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def show_images(images, titles, fignum=1):
    n = len(images)
    x = y = math.ceil(math.sqrt(n))
    plt.figure(fignum)
    for idx, img in enumerate(images):
        plt.subplot(x, y, idx+1)
        plt.imshow(img)
        plt.title(titles[idx])
    plt.show()
    
def show_figures(images, titles, fignum=1):
    for idx, img in enumerate(images):
        plt.figure(fignum)
        plt.imshow(img)
        plt.title(titles[idx])
        fignum=fignum+1
        
    plt.show()

def get_edge(a,method):
    if method=='canny':
        edge=cv2.Canny(a,0,1).astype(np.uint8)
    elif method=='filter2d-x-y-3':
        kernel=np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.uint8)
        edge_1=cv2.filter2D(a,cv2.CV_64F,kernel)
        kernel=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=np.uint8)
        edge_2=cv2.filter2D(a,cv2.CV_64F,kernel)
        edge=np.logical_or(edge_1!=0,edge_2!=0).astype(np.uint8)
    elif method=='filter2d-x-y-1':
        kernel=np.array([[0,1,0],[0,0,0],[0,-1,0]],dtype=np.uint8)
        edge_1=cv2.filter2D(a,cv2.CV_64F,kernel)
        kernel=np.array([[0,0,0],[1,0,-1],[0,0,0]],dtype=np.uint8)
        edge_2=cv2.filter2D(a,cv2.CV_64F,kernel)
        edge=np.logical_or(edge_1!=0,edge_2!=0).astype(np.uint8)
    elif method=='laplacian':
        edge=np.absolute(cv2.Laplacian(a,cv2.CV_64F,ksize=3))
        edge=(edge!=0).astype(np.uint8)
    elif method=='scharr':
        edge=np.absolute(cv2.Scharr(a,cv2.CV_64F,1,0))
        edge=(edge!=0).astype(np.uint8)
    elif method=='sobel':
        edge=np.absolute(cv2.Soble(a,cv2.CV_64F,1,0,ksize=3))
        edge=(edge!=0).astype(np.uint8)
    else:
        print('unknown method for edge detection',method)
        assert False
        
    return edge
            
def get_instance_gradiants(img,method='canny'):
    assert img.dtype == np.dtype('uint16')
    img_semantic=(img//256).astype(np.uint8)
    img_instance=(img%256).astype(np.uint8)
    
    edges=[]
    
    for a in [img_semantic,img_instance]:
        edges.append(get_edge(a,method))
    
    return np.logical_or(edges[0],edges[1]).astype(np.uint8)
            

def show_contour(img_file,root_dir,method='canny'):
#    img_file='WildDash_za0000_100000.png'
#    root_dir='/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    img_2_path=os.path.join(root_dir,'image_2',img_file)
    img_2=cv2.imread(img_2_path,cv2.IMREAD_UNCHANGED)
    print('img_2_path',img_2_path)
    
    img_file=img_file.replace('.jpg','.png')
    instance_path=os.path.join(root_dir,'instance',img_file)
    print('instance_path',instance_path)
    img_instance=cv2.imread(instance_path,cv2.IMREAD_UNCHANGED)
    print('img_instance',img_instance.dtype)
    
    semantic_path=os.path.join(root_dir,'semantic',img_file)
    print('semantic_path',semantic_path)
    img_semantic=cv2.imread(semantic_path,cv2.IMREAD_UNCHANGED)
    print('img_semantic',img_semantic.dtype)
    
    edge_semantic=get_edge(img_semantic,method)
    edge_instance = get_instance_gradiants(img_instance,method)
#    edge_semantic=np.clip(a=edge_semantic,a_min=0,a_max=1).astype(np.uint8)
    
    show_figures([img_2,img_semantic,img_instance,edge_semantic,edge_instance],['image_2','semantic','instance','edge-semantic','edge-instance'])
    
    
if __name__ == '__main__':
    root_dir='/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
    img_files=os.listdir(os.path.join(root_dir,'image_2'))
    img_files=[a for a in img_files if a.startswith('Cityscapes')]
    img_file=np.random.choice(img_files)
    show_contour(img_file,root_dir)