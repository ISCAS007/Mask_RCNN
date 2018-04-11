# -*- coding: utf-8 -*-
"""
usage:
    model = Model(inputs=[inputs], outputs=[outputs])
    tf_iou_metric=slow_mean_iou(34)
    metrics=[tf_iou_metric]
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=metrics )
    model.summary()
reference:
    https://www.kaggle.com/aglotero/another-iou-metric (error in our case)
    https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras (okay in our case)
"""
import tensorflow as tf
import keras.backend as K

def slow_mean_iou(NUM_CLASSES):
    def tf_mean_iou(y_true, y_pred):
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred, NUM_CLASSES)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score
    
    return tf_mean_iou
