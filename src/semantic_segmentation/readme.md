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
  24353   24353  876708
(env3) ➜  datasets_kitti2015 ls training/image_2 | grep Kitti |wc
    200     200    4800
(env3) ➜  datasets_kitti2015 ls training/image_2 | grep City |wc 
   3475    3475  130570
(env3) ➜  datasets_kitti2015 ls training/image_2 | wc           
  28098   28098 1013968
```

# [ScanNet](http://dovahkiin.stanford.edu/adai/documentation)
```shell
# predicted labels
34	wall
35	floor
36	cabinet
37	bed
38	chair
39	sofa
40	table
41	door
42	window
43	bookshelf
44	picture
45	counter
47	desk
49	curtain
57	refridgerator
61	shower curtain
66	toilet
67	sink
69	bathtub
72	otherfurniture

# all labels, NYUv2 40-label set with an offset of +33
34	wall
35	floor
36	cabinet
37	bed
38	chair
39	sofa
40	table
41	door
42	window
43	bookshelf
44	picture
45	counter
46	blinds
47	desk
48	shelves
49	curtain
50	dresser
51	pillow
52	mirror
53	floor mat
54	clothes
55	ceiling
56	books
57	refridgerator
58	television
59	paper
60	towel
61	shower curtain
62	box
63	whiteboard
64	person
65	nightstand
66	toilet
67	sink
68	lamp
69	bathtub
70	bag
71	otherstructure
72	otherfurniture
73	otherprop
```