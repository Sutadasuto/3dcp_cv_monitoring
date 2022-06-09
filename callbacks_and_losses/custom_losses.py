import numpy as np

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError, binary_crossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.python.ops import math_ops


# Smooth factor for dice coefficient. DC = (2 * GT n Pred + 1) / (GT u Pred + 1)
smooth = 1


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)


def bce_dsc_loss(alpha=0.5):
    def hybrid_loss(y_true, y_pred):
        dice = dice_coef_loss(y_true, y_pred)
        BCE = BinaryCrossentropy()
        bce = BCE(y_true, y_pred)
        return K.sum(bce + alpha * dice)

    return hybrid_loss


def corrected_dice_coef(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    return dice_coef(gt*weights, y_pred*weights)


def corrected_bce(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    BCE = BinaryCrossentropy()
    return BCE(gt, y_pred, sample_weight=weights)


def corrected_bce_dsc_loss(alpha=0.5):
    def hybrid_loss(y_true, y_pred):
        dice = 1.0 - corrected_dice_coef(y_true, y_pred)
        bce = corrected_bce(y_true, y_pred)
        return K.sum(bce + alpha * dice)

    return hybrid_loss


def corrected_precision(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    PRE = Precision()
    return PRE(gt, y_pred, sample_weight=weights)


def corrected_recall(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    RE = Recall()
    return RE(gt, y_pred, sample_weight=weights)


def test_dice_coef(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    return K.sum(K.flatten(weights))
    y_true = gt
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def test_bce(y_true, y_pred):
    gt, weights = tf.split(y_true, 2, -1)
    BCE = BinaryCrossentropy()
    return BCE(gt, y_pred)


def test_bce_dsc_loss(alpha=0.5):
    def hybrid_loss(y_true, y_pred):
        dice = 1.0 - test_dice_coef(y_true, y_pred)
        bce = test_bce(y_true, y_pred)
        return K.sum(bce + alpha * dice)

    return hybrid_loss

