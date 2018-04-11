# sementic segmentation

# download and transform weight for fcn
models/utils/get_weights_path.py
models/utils/transfer_FCN.py

however, DenseNet_FCN cannot be useed due to the update of keras_contrib

# download and transform dataset for rob2018
```shell
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0645_02/color')
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0141_01/color')
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0591_02/color')
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0230_00/color')
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0261_03/color')
('warning: cannot find path color_path', 'temp_unpack_dir/scannet/train/scene0583_01/color')

(env3) ➜  datasets_kitti2015 ls training/image_2 | grep Wild |wc
     70      70    1890
(env3) ➜  datasets_kitti2015 ls training/image_2 | grep Scan |wc
  24366   24366  877176
(env3) ➜  datasets_kitti2015 ls training/image_2 | grep Kitti |wc
    200     200    4800
(env3) ➜  datasets_kitti2015 ls training/image_2 | grep City |wc 
   3475    3475  130570
(env3) ➜  datasets_kitti2015 ls training/image_2 | wc           
  28111   28111 1014436
```