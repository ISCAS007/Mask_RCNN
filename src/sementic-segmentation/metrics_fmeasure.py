# -*- coding: utf-8 -*-

import keras.backend as K

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    arg_y_true = K.cast(K.argmax(y_true),K.floatx())
    arg_y_pred = K.cast(K.argmax(y_pred),K.floatx())
    true_positives = K.sum(K.cast(K.equal(arg_y_true,arg_y_pred),K.floatx()))
    predicted_positives = K.sum(K.cast(K.greater(arg_y_pred,0),K.floatx()))
    precision = true_positives / (predicted_positives+K.constant(0.1,K.floatx()))
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    arg_y_true = K.cast(K.argmax(y_true),K.floatx())
    arg_y_pred = K.cast(K.argmax(y_pred),K.floatx())
    true_positives = K.sum(K.cast(K.equal(arg_y_true,arg_y_pred),K.floatx()))
    possible_positives = K.sum(K.cast(K.greater(arg_y_true,0),K.floatx()))
    recall = true_positives / (possible_positives+K.constant(0.1,K.floatx()))
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.argmax(y_true)) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r+K.constant(0.1,K.floatx()))
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)

#TODO https://github.com/JihongJu/keras-fcn/blob/master/keras_fcn/metrics.py
#also refer to https://github.com/david-vazquez/keras_zoo/blob/master/metrics/metrics.py
# =============================================================================
# import keras.backend as K
# import tensorflow as tf
# from tensorflow.python.ops import control_flow_ops
# 
# def Mean_IoU(classes):
#     def mean_iou(y_true, y_pred):
#         mean_iou, op = tf.metrics.mean_iou(y_true, y_pred, classes)
#         return mean_iou
#     _initialize_variables()
#     return mean_iou
# 
# 
# def _initialize_variables():
#     """Utility to initialize uninitialized variables on the fly.
#     """
#     variables = tf.local_variables()
#     uninitialized_variables = []
#     for v in variables:
#         if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
#             uninitialized_variables.append(v)
#             v._keras_initialized = True
#     if uninitialized_variables:
#         sess = K.get_session()
#         sess.run(tf.variables_initializer(uninitialized_variables))
# =============================================================================
