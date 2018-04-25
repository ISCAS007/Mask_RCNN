# -*- coding: utf-8 -*-
import os
from keras.applications import MobileNet, VGG16, VGG19, ResNet50, DenseNet121, DenseNet169, DenseNet201, NASNetMobile, InceptionV3, Xception, InceptionResNetV2
from keras.backend import tf as ktf
import keras

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

def get_encoder(config):
    encoder = config['encoder'].lower()
    h, w, c = config['input_shape']
    model_path = os.path.join(os.getenv('HOME'), '.keras', 'models')

    if encoder == 'mobilenet':
        base_model = MobileNet(
            include_top=False, weights=None, input_shape=(h, w, 3))
        base_model.load_weights(filepath=os.path.join(
            model_path, 'mobilenet_1_0_224_tf_no_top.h5'))

        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('conv_pw_1_relu')
        conv2 = base_model.get_layer('conv_pw_3_relu')
        conv3 = base_model.get_layer('conv_pw_5_relu')
        conv4 = base_model.get_layer('conv_pw_11_relu')
        conv5 = base_model.get_layer('conv_pw_13_relu')
    elif encoder == 'vgg16':
        base_model = VGG16(include_top=False, input_shape=(h, w, 3))

        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('block1_pool')
        conv2 = base_model.get_layer('block2_pool')
        conv3 = base_model.get_layer('block3_pool')
        conv4 = base_model.get_layer('block4_pool')
        conv5 = base_model.get_layer('block5_pool')

    elif encoder == 'vgg19':
        base_model = VGG19(include_top=False, input_shape=(h, w, 3))

        for layer in base_model.layers:
            layer.trainable = False

        input0 = base_model.get_layer(index=0)
        conv1 = base_model.get_layer('block1_pool')
        conv2 = base_model.get_layer('block2_pool')
        conv3 = base_model.get_layer('block3_pool')
        conv4 = base_model.get_layer('block4_pool')
        conv5 = base_model.get_layer('block5_pool')
    elif encoder == 'resnet50':
        base_model = ResNet50(
            include_top=False, weights='imagenet', input_shape=(h, w, 3))
        for layer in base_model.layers:
            layer.trainable = False
#            print(layer.name)
#        base_model.summary()

        input0 = base_model.get_layer(index=0)
        conv_names={'112':'activation_1',
                    '55':'activation_10',
                    '28':'activation_22',
                    '14':'activation_40',
                    '7':'activation_49'}
        
# =============================================================================
# layer_idx layer_name
# 3 activation_1
# 36 activation_10
# 78 activation_22
# 140 activation_40
# 172 activation_49
# =============================================================================

        
        conv1 = base_model.get_layer(index=3)  # 112
        _conv2 = base_model.get_layer(index=36)  # 55
#        conv2 = keras.layers.ZeroPadding2D(
#            padding=((0, 1), (0, 1)), data_format=config['data_format'])  # 56
        conv2 = Interp(new_size=(56,56))
        conv2(_conv2.output)
        conv2.trainable = False
#        print('_output shape is', conv2.output_shape)
        conv3 = base_model.get_layer(index=78)
        conv4 = base_model.get_layer(index=140)
        conv5 = base_model.get_layer(index=172)
        
        for layer in base_model.layers:
            layer.name="_".join([encoder,layer.name])
        
#        for idx,layer in enumerate([conv1,conv2,conv3,conv4,conv5]):
#            print('conv%d: %s'%(idx,layer.name))
    elif encoder.lower()=='NASNetMobile'.lower():
        print('unknown encoder')
        assert False
        base_model = NASNetMobile(
            include_top=False, weights='imagenet', input_shape=(h, w, 3))
        conv_names={'111':'reduction_bn_1_stem_1',
                    '56':'adjust_bn_stem_2',
                    '28':'adjust_bn_reduce_4',
                    '14':'adjust_bn_reduce_8',
                    '7':'normal_concat_12'}
    else:
        # simplify interface
        if encoder.lower()=='DenseNet121'.lower():
            base_model = DenseNet121(
                include_top=False, weights='imagenet', input_shape=(h, w, 3))
            conv_names=['conv1/relu','pool2_conv','pool3_conv','pool4_conv','conv5_block16_concat']
        elif encoder.lower()=='DenseNet169'.lower():
            base_model = DenseNet169(
                include_top=False, weights='imagenet', input_shape=(h, w, 3))
            conv_names=['conv1/relu','pool2_conv','pool3_conv','pool4_conv','conv5_block32_concat']
        elif encoder.lower()=='DenseNet201'.lower():
            base_model = DenseNet201(
                include_top=False, weights='imagenet', input_shape=(h, w, 3))
            conv_names=['conv1/relu','pool2_conv','pool3_conv','pool4_conv','conv5_block32_concat']
        elif encoder.lower()=='InceptionResNetV2'.lower():
            print('unknown encoder')
            assert False
            #input_shape=[(299,299,3)]
        elif encoder.lower()=='InceptionV3'.lower():
            print('unknown encoder')
            assert False
            #input_shape=(299,299,3)
            pass
        elif encoder.lower()=='Xception'.lower():
            print('unknown encoder')
            assert False
            #input_shape=(299,299,3)
            pass
        else:
            print('unknown encoder')
            assert False
            
        for layer in base_model.layers:
            layer.trainable = False
        input0 = base_model.get_layer(index=0)
        conv_layers=[input0]
        for layer_name in conv_names:
            layer=base_model.get_layer(layer_name)
            conv_layers.append(layer)
            
        return conv_layers
                
    input_layers = [input0,
                    conv1,
                    conv2,
                    conv3,
                    conv4,
                    conv5]

    return input_layers
