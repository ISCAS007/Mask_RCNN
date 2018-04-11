# -*- coding: utf-8 -*-
import os
from keras.applications import MobileNet,VGG16,VGG19

def get_encoder(config):
    encoder=config['encoder'].lower()
    h, w, c =config['input_shape']
    model_path=os.path.join(os.getenv('HOME'),'.keras','models')
    
    if encoder=='mobilenet':
        base_model = MobileNet(include_top=False, weights=None,input_shape=(h,w,3))
        base_model.load_weights(filepath=os.path.join(model_path,'mobilenet_1_0_224_tf_no_top.h5'))

        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('conv_pw_1_relu')
        conv2 = base_model.get_layer('conv_pw_3_relu')
        conv3 = base_model.get_layer('conv_pw_5_relu')
        conv4 = base_model.get_layer('conv_pw_11_relu')
        # conv5 = base_model.get_layer('conv_pw_13_relu')
    elif encoder=='vgg16':
        base_model = VGG16(include_top=False, input_shape=(h, w, 3))
        
        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('block1_pool')
        conv2 = base_model.get_layer('block2_pool')
        conv3 = base_model.get_layer('block3_pool')
        conv4 = base_model.get_layer('block4_pool')
        #conv5 = base_model.get_layer('block5_pool')

    elif encoder=='vgg19':
        base_model = VGG19(include_top=False, input_shape=(h, w, 3))
        
        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('block1_pool')
        conv2 = base_model.get_layer('block2_pool')
        conv3 = base_model.get_layer('block3_pool')
        conv4 = base_model.get_layer('block4_pool')
        #conv5 = base_model.get_layer('block5_pool')
    else:
        print('unknown encoder')
        assert False
        
    input_layers = [input0,
                    conv1,
                    conv2,
                    conv3,
                    conv4,
                    base_model]

    return input_layers
