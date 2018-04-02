# -*- coding: utf-8 -*-
"""
motion net structure for semantic segmentation
"""

from semantic_segmentation_basic import semantic_segmentation_basic
from encoder.encoder_basic import encoder_basic
import keras
from keras.layers import Conv2D, Conv2DTranspose


class semantic_segmentation_motion_net(semantic_segmentation_basic):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)

    def train(self):
        self.model.compile(loss='mse',
                           optimizer='adam')

        dataset = self.get_dataset(self.config)
        train_main_input_paths, test_main_input_paths = dataset.get_main_input_list()

        print('train dataset size', len(train_main_input_paths))
        print('test dataset size', len(test_main_input_paths))

        self.model.fit_generator(generator=dataset.batch_gen(train_main_input_paths, 'trainval'),
                                 steps_per_epoch=dataset.get_steps_per_epoch(
                                     train_main_input_paths),
                                 epochs=self.config['epoches'],
                                 verbose=1,
                                 callbacks=self.get_callbacks(self.config),
                                 validation_data=dataset.batch_gen(
                                     test_main_input_paths, 'trainval'),
                                 validation_steps=dataset.get_steps_per_epoch(test_main_input_paths))

    def get_model(self):
        # h, w, c = self.config['input_shape']
        input_layers = encoder_basic.get_encoder(self.config)
        merge_type = self.config['merge_type'].lower()
        drop_ratio = self.config['dropout_ratio']
        data_format = self.config['data_format']

        if data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        layer_depth = len(input_layers)
        deconv_output = None
        model = None
        for i in range(layer_depth):
            if i == 0:
                merge_output = input_layers[-i - 1].output
                merge_output = keras.layers.Dropout(
                    rate=drop_ratio)(merge_output)
            else:
                assert deconv_output is not None
                merge_output = self.merge(inputs=[input_layers[-i - 1].output,
                                                  deconv_output],
                                          mode=merge_type)

                merge_output = keras.layers.Dropout(
                    rate=drop_ratio)(merge_output)

                if merge_type == 'concat':
                    merge_output = Conv2D(filters=input_layers[-i - 1].output_shape[channel_axis],
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same',
                                          data_format=data_format)(merge_output)
                    merge_output = keras.layers.BatchNormalization()(merge_output)
                    merge_output = keras.layers.Dropout(
                        rate=drop_ratio)(merge_output)

            if i < layer_depth - 1:
                # input_filters=merge_output.shape[channel_axis].value
                # target_filters=input_layers[-i-2].output_shape[channel_axis]

                deconv_output = Conv2DTranspose(filters=input_layers[-i - 2].output_shape[channel_axis],
                                                kernel_size=(2, 2),
                                                strides=(2, 2),
                                                activation='relu',
                                                padding='same',
                                                data_format=data_format)(merge_output)
                deconv_output = keras.layers.BatchNormalization()(deconv_output)
                deconv_output = keras.layers.Dropout(
                    rate=drop_ratio)(deconv_output)

            else:
                outputs = Conv2D(filters=self.config['class_number'],
                                 kernel_size=(3, 3),
                                 activation='softmax',
                                 padding='same',
                                 data_format=data_format)(merge_output)

                model = keras.models.Model(inputs=input_layers[0].input,
                                           outputs=outputs)

        return model
