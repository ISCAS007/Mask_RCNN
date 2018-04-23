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
import pickle
import randomcolor
import sys

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from dataset.labels_cityscapes import labels


def get_rob2018_dataset_names():
    return ['cityscapes', 'kitti2015', 'scannet', 'wilddash']


# def get_rob2018_dataset_class_num(dataset_name):
#    """
#    return class number for target dataset
#    """
#    class_num_dict = {'cityscapes': 0,
#                      'kitti2015': 0,
#                      'scannet': 34,
#                      'wilddash': 0}
#    return class_num_dict[dataset_name]


def show_images(images, titles, fignum=1):
    n = len(images)
    x = y = math.ceil(math.sqrt(n))
    plt.figure(fignum)
    for idx, img in enumerate(images):
        plt.subplot(x, y, idx+1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(titles[idx])
    plt.show()


def get_color_map(class_num):
    def get_rgb(rgb_str):
        numbers = rgb_str.split('(')[1]
        numbers = numbers.split(')')[0]
        numbers = numbers.split(',')

        rgb = [int(i) for i in numbers]

        return rgb

    rand_color = randomcolor.RandomColor()
    color_map_str = rand_color.generate(count=class_num, format_='rgb')
    color_map = [get_rgb(i) for i in color_map_str]

    return color_map


def label2rgb(img, class_num, color_map):
    assert img.ndim == 2
    img_rgb = np.zeros(img.shape+(3,), dtype=np.uint8)

    for i in range(class_num):
        img_rgb[img == i] = color_map[i]

    return img_rgb


def preprocess_image_bgr(x):
    """
    batch_size,height,width,channel=x.shape
    scale x from [0,255] to [-1,1]
    """
    assert x.ndim == 4
    x = 2*(x.astype(np.float32)/255.0-0.5)

    return x


def preprocess_image_mask(y, num_classes):
    """
    batch_size,height,width=y.shape
    """
    assert y.ndim == 3

    b, h, w = y.shape
    y_category = keras.utils.to_categorical(y, num_classes)
    y_category = y_category.reshape([b, h, w, num_classes])
    return y_category


def get_centered_data(instance_image_list, target_size):
    """
    data={'shape':img.shape,
          'offset_image':offset_img,
          'center_points':center_points,
          'instances':instances}
    """

    # offset image and center point image
    offset_images = []
    cp_images = []

    for img_file in instance_image_list:
        task_dir = os.path.dirname(img_file)
        root_dir = os.path.dirname(task_dir)
        base_name = os.path.basename(img_file)
        pickle_name = base_name.split('.')[0]+".pkl"
        pickle_path = os.path.join(root_dir, 'centered_data', pickle_name)

        if not os.path.exists(pickle_path):
            print('unexist path for', pickle_path)
            assert False

        f = open(pickle_path, 'rb')
        data = pickle.load(f)
        offset_image = data['offset_image']
        h, w, c = offset_image.shape

        # range in [-1,1], use sigmoid
        x_offset = offset_image[:, :, 0]/(1.0*w)
        y_offset = offset_image[:, :, 1]/(1.0*h)

        resize_offset = np.zeros(target_size+(2,), dtype=np.float32)
        resize_offset[:, :, 0] = cv2.resize(x_offset, target_size)
        resize_offset[:, :, 1] = cv2.resize(y_offset, target_size)

        center_points = data['center_points']
        smooth_size = 2*10+1
        ksize = (smooth_size, smooth_size)
        smooth_sigma = 5
        center_point_image = np.zeros((h, w), dtype=np.float32)
        for x, y in center_points:
            center_point_image[int(y), int(x)] = 1

        center_point_image = cv2.GaussianBlur(
            center_point_image, ksize, smooth_sigma)

        # make center point to 1
        for x, y in center_points:
            center_point_image[int(y), int(x)] = 1

        # range in [0,1], use relu
        resize_cp_image = cv2.resize(center_point_image, target_size)

        offset_images.append(resize_offset)
        cp_images.append(resize_cp_image.reshape(target_size+(1,)))
    return np.asarray(offset_images, dtype=np.float32), np.asarray(cp_images, dtype=np.float32)


class dataset_rob2018():
    """
    [Robust Vision Challenge 2018](http://www.robustvision.net/)
    [devkit](http://cvlibs.net:3000/ageiger/rob_devkit.git)
    dataset=[cityscapes,kitti2015,scannet,wilddash]
    """

    def __init__(self, config):
        self.dataset_name = config['dataset_name']
        self.dataset_train_root = config['dataset_train_root']
        self.dataset_test_root = config['dataset_test_root']
        self.task = config['task']
        self.target_size = config['target_size']

        self.transfer_id_to_eval = False
        if 'transfer_id_to_eval' in config.keys():
            self.transfer_id_to_eval = config['transfer_id_to_eval']

        if self.transfer_id_to_eval:
            self.num_classes_dict = {'cityscapes': 20,
                                     'kitti2015': 20,
                                     'scannet': 40,
                                     'wilddash': 20}
        else:
            self.num_classes_dict = {'cityscapes': 34,
                                     'kitti2015': 34,
                                     'scannet': 74,
                                     'wilddash': 34}

        self.num_classes = self.num_classes_dict[self.dataset_name]

        self.dataset_size_dict = {'cityscapes': 3475,
                                  'kitti2015': 200,
                                  'scannet': 24353,
                                  'wilddash': 70}

        self.dataset_filter_dict = {'cityscapes': 'Cityscapes',
                                    'kitti2015': 'Kitti2015',
                                    'scannet': 'ScanNet',
                                    'wilddash': 'WildDash'}

        # in fact, we can use dataset_name for filter directly!!!
        self.dataset_filters = self.dataset_filter_dict[self.dataset_name]

        assert self.task in ['instance', 'semantic']

    def transfer_image_mask_id(self, y):
        if self.dataset_name == 'cityscapes':
            eval_y = np.zeros(y.shape, y.dtype)
            for l in labels:
                eval_y[y == l.id] = l.trainId

            eval_y[eval_y == 255] = 19

            return eval_y
        else:
            return y

    def get_uneval_img(self, label):
        assert label.ndim == 2
        if self.transfer_id_to_eval:
            return (label == 19).astype(np.uint8)

        uneval_img = np.zeros(label.shape, np.uint8)
        if self.dataset_name == 'cityscapes':
            for l in labels:
                if l.ignoreInEval:
                    uneval_img[label == l.id] = 1

        return uneval_img

    def batch_gen_images(self, image_2_list, batch_size):
        tasks = ['image_2', self.task]
        for image_2_batch, task_batch in self.batch_gen_paths(image_2_list, batch_size):
            images_batchs = []
            for _list, _task in zip([image_2_batch, task_batch], tasks):
                img_list = []
                for _f in _list:
                    assert os.path.exists(_f)
                    if _task == 'image_2':
                        img = cv2.imread(_f)
                    else:
                        img = cv2.imread(_f, cv2.IMREAD_GRAYSCALE)

                    img_list.append(img)

                if _task == 'image_2':
                    x = [cv2.resize(
                        img, self.target_size, interpolation=cv2.INTER_LINEAR) for img in img_list]
                    x = np.asarray(x, dtype=np.float32)
                    images_batchs.append(preprocess_image_bgr(x))
                elif _task == 'semantic':
                    y = [cv2.resize(
                        img, self.target_size, interpolation=cv2.INTER_NEAREST) for img in img_list]
                    y = np.asarray(y, dtype=np.float32)
                    if self.transfer_id_to_eval:
                        y = self.transfer_image_mask_id(y)

                    images_batchs.append(preprocess_image_mask(
                        y, self.num_classes))
                elif _task == 'instance':
                    y_category = [cv2.resize(
                        img, self.target_size, interpolation=cv2.INTER_NEAREST) for img in img_list]

                    y_category = np.asarray(y_category, dtype=np.float32)

                    y = []
                    y.append(preprocess_image_mask(
                        y_category, self.num_classes))

                    # center offset [-1,1] + center mask [0,1]
                    y_offset, y_center = get_centered_data(
                        _list, self.target_size)
                    assert y_offset.ndim == 4
                    assert y_center.ndim == 4

                    y.append(y_offset)
                    y.append(y_center)

                    images_batchs.append(y)
                else:
                    print('undefined precessing for task', _task)
                    assert False

#            if self.task=='semantic':
#                assert len(images_batchs) == 2
#            else:
#                assert len(images_batchs) == 4

            assert images_batchs[0].ndim == 4
#            assert images_batchs[1].ndim == 4
            yield images_batchs

    def batch_gen_inputs_for_val(self, image_2_list, task_list, batch_size, shuffle=False):
        """
        generate images with labels for validation
        """
        # todo support for instance label
        if shuffle:
            assert len(image_2_list) == len(task_list)
            seed = np.random.randint(10000)
            rand_state = np.random.RandomState(seed)
            rand_state.shuffle(image_2_list)
            rand_state.seed(seed)
            rand_state.shuffle(task_list)

        img2_origin_list = []
        task_origin_list = []
        for img2_f, task_f in zip(image_2_list, task_list):
            img2 = cv2.imread(img2_f, cv2.IMREAD_UNCHANGED)
            task = cv2.imread(task_f, cv2.IMREAD_UNCHANGED)
            img2_origin_list.append(img2)
            task_origin_list.append(task)

            if len(img2_origin_list) == batch_size:
                x = [cv2.resize(
                    img, self.target_size, interpolation=cv2.INTER_LINEAR) for img in img2_origin_list]
                x1 = np.asarray(x, dtype=np.float32)
                x2 = preprocess_image_bgr(x1)

                y = [cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)
                     for img in task_origin_list]
                y1 = np.asarray(y, dtype=np.float32)
                if self.transfer_id_to_eval:
                    y1 = self.transfer_image_mask_id(y1)
                y2 = preprocess_image_mask(
                    y1, self.num_classes)

                yield x, y, x2, y2

                img2_origin_list = []
                task_origin_list = []

        if len(img2_origin_list) > 0:
            x = [cv2.resize(
                img, self.target_size, interpolation=cv2.INTER_LINEAR) for img in img2_origin_list]
            x1 = np.asarray(x, dtype=np.float32)
            x2 = preprocess_image_bgr(x1)

            y = [cv2.resize(img, self.target_size, interpolation=cv2.INTER_NEAREST)
                 for img in task_origin_list]
            y1 = np.asarray(y, dtype=np.float32)
            if self.transfer_id_to_eval:
                y1 = self.transfer_image_mask_id(y1)
            y2 = preprocess_image_mask(
                y1, self.num_classes)

            yield x, y, x2, y2

            img2_origin_list = []
            task_origin_list = []

    def batch_gen_inputs_for_test(self, batch_size, shuffle=False):
        """
        generate images for test dataset
        """

        task_root = os.path.join(self.dataset_test_root, 'image_2')
        image_2_list = self.get_image_file_paths(task_root)
        paths = [os.path.join(task_root, f) for f in image_2_list]

        if shuffle:
            np.random.shuffle(paths)

        img_list = []
        img_path_list = []
        for _f in paths:
            assert os.path.exists(_f)
            img = cv2.imread(_f)

            img_list.append(img)
            img_path_list.append(_f)
            if len(img_list) == batch_size:
                x = [cv2.resize(
                    img, self.target_size, interpolation=cv2.INTER_LINEAR) for img in img_list]
                x1 = np.asarray(x, dtype=np.float32)
                yield x, img_path_list, preprocess_image_bgr(x1)
                img_list = []
                img_path_list = []

        if len(img_list) > 0:
            x = [cv2.resize(
                img, self.target_size, interpolation=cv2.INTER_LINEAR) for img in img_list]
            x1 = np.asarray(x, dtype=np.float32)
            yield x, img_path_list, preprocess_image_bgr(x1)
            img_list = []
            img_path_list = []

    def batch_gen_paths(self, image_2_list, batch_size, output=False):
        """
        input relative image_2 list
        output batch path
        """
        n = len(image_2_list)
        while True:
            random.shuffle(image_2_list)
            paths = []
            for task in ['image_2', self.task]:
                task_root = os.path.join(self.dataset_train_root, task)
                abs_list = [os.path.join(task_root, f) for f in image_2_list]
                paths.append(abs_list)

            for idx in range(0, n, batch_size):
                if idx+batch_size <= n:
                    if output == True:
                        print('image_2 is', paths[0][idx:idx+batch_size])
                        print('mask is', paths[1][idx:idx+batch_size])
                    else:
                        yield [paths[0][idx:idx+batch_size], paths[1][idx:idx+batch_size]]

    def get_image_file_paths(self, image_dir):
        #        image_dir = os.path.join(self.dataset_train_root, 'image_2')
        _list = os.listdir(image_dir)
#            print('_list size is',len(_list))
        img_suffix = ('png', 'jpg', 'jpeg', 'bmp')
        img_list = [f for f in _list if f.lower().endswith(
            img_suffix) and f.lower().startswith(self.dataset_filters.lower())]

        return img_list

    def get_train_val(self):
        train_txt = os.path.join(
            self.dataset_train_root, '%s_train.txt' % self.dataset_name)
        val_txt = os.path.join(self.dataset_train_root,
                               '%s_val.txt' % self.dataset_name)

        lists = []
        if os.path.exists(train_txt) and os.path.exists(val_txt):
            for txt in [train_txt, val_txt]:
                file = open(txt, 'r')
                lines = file.readlines()
                _list = [line.strip() for line in lines]
                lists.append(_list)
                file.close()
        else:
            print('warning: cannot find train txt and val txt', train_txt, val_txt)
            print('create them by random')
            image_dir = os.path.join(self.dataset_train_root, 'image_2')
            img_list = self.get_image_file_paths(image_dir)
            list_size = len(img_list)
            if list_size != self.dataset_size_dict[self.dataset_name]:
                print('except size is',
                      self.dataset_size_dict[self.dataset_name])
                print('infact size is', list_size)
                assert False

            random.shuffle(img_list)
            train_size = list_size*3//5
            lists.append(img_list[:train_size])
            lists.append(img_list[train_size:])
            for _list, _file in zip(lists, [train_txt, val_txt]):
                file = open(_file, 'w')
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

        f = None
        for _dir in os.listdir(self.dataset_train_root):
            sub_dir = os.path.join(self.dataset_train_root, _dir)
            if not os.path.isdir(sub_dir):
                continue

            f_list = os.listdir(sub_dir)
            if len(f_list) == 0:
                print('warning: empty dir', sub_dir)
                continue

            f = random.choice(f_list)
            break

        imgs = []
        titles = []
        if f is not None:
            for _dir in os.listdir(self.dataset_train_root):
                img_file = os.path.join(self.dataset_train_root, _dir, f)
                if os.path.exists(img_file):
                    img = cv2.imread(img_file)
                    imgs.append(img)
                    titles.append(_dir)
                else:
                    print('cannot find image file', img_file)

        show_images(imgs, titles)

    def get_class_range(self):
        """
        max class id is 33
        min class id is 0
        """
        sub_dir = os.path.join(self.dataset_train_root, self.task)
        f_list = os.listdir(sub_dir)

        img_suffix = ('png', 'jpg', 'jpeg', 'bmp')
        f_list = [f for f in f_list if f.lower().endswith(
            img_suffix) and f.lower().startswith(self.dataset_filters.lower())]
        list_size = len(f_list)
        assert list_size == self.dataset_size_dict[self.dataset_name]

        max_class = 0
        min_class = 100
        for idx, f in enumerate(f_list):
            img_file = os.path.join(sub_dir, f)
            img = cv2.imread(img_file)
            max_class = max(np.max(img), max_class)
            min_class = min(np.min(img), min_class)

#            if idx % 20 == 0:
#                print("%d/%d"%(idx,list_size),f)
#                print('sub_max class id is', max_class)
#                print('sub_min class id is', min_class)

        print('for dataset', self.dataset_name)
        print('max class id is', max_class)
        print('min class id is', min_class)


if __name__ == '__main__':
    app = 'label_test'
    if app == 'batch_test':
        dataset_names = get_rob2018_dataset_names()
        for name in dataset_names:
            if name is 'scannet':
                continue

            config = {}
            config['dataset_name'] = name
            config['dataset_train_root'] = '/media/sdb/CVDataset/ObjectSegmentation/datasets_kitti2015/training'
            config['task'] = 'semantic'
            config['target_size'] = (224, 224)

            print('config is', config)
            dataset = dataset_rob2018(config)
            dataset.get_class_range()
        #    dataset.view()

            train_list, val_list = dataset.get_train_val()
            print('train dataset size is', len(train_list))
            print('val dataset size is', len(val_list))
            dataset.batch_gen_paths(train_list, batch_size=2, output=True)
    elif app == 'label_test':
        for l in labels:
            print(l.name, l.id, l.trainId, l.ignoreInEval, l.color)
