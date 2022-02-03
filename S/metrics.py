import tensorflow as tf

def multiclass_dice(channel=None):
    def dice(y_true, y_pred):
        y_true = y_true[..., :-1]
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        y_true_vol = tf.reduce_sum(y_true, axis=(1, 2, 3))
        y_pred_vol = tf.reduce_sum(y_pred, axis=(1, 2, 3))
        dsc = 2 * intersection / (y_true_vol + y_pred_vol)
        if channel != None:
            return dsc[..., channel]
        return tf.reduce_mean(dsc, axis=-1)
    setattr(dice, 'name', 'dice_{channel}'.format(channel=channel if channel != None else 'mc'))
    return dice

## Evaluation metrics

import numpy as np
from scipy import ndimage

def volume_metrics(y_true, y_pred, channel=None):
    if channel != None:
        y_true = y_true[..., channel]
        y_pred = y_pred[..., channel]
        
    intersection = np.sum(y_true * y_pred)
    y_true_vol = np.sum(y_true)
    y_pred_vol = np.sum(y_pred)
    sum_vol = y_true_vol + y_pred_vol

    dsc = (2 * intersection) / sum_vol
    voe = 1 - intersection / (sum_vol - intersection)
    vd = (y_pred_vol - y_true_vol) / y_true_vol
    return dsc, voe, vd

def surface_distance(y_true, y_pred, channel=None, sampling=(1, 1, 1)):
    if channel != None:
        y_true = y_true[..., channel]
        y_pred = y_pred[..., channel]
        
    conn = ndimage.generate_binary_structure(3, 1)
    S_true = y_true - ndimage.binary_erosion(y_true, conn)
    S_pred = y_pred - ndimage.binary_erosion(y_pred, conn)
    
    dt_true = ndimage.distance_transform_edt((1 - S_true), sampling)
    dt_pred = ndimage.distance_transform_edt((1 - S_pred), sampling)

    sd_true = dt_true[S_pred==1]
    sd_pred = dt_pred[S_true==1]
    sds = np.concatenate([sd_true, sd_pred])

    asd = np.mean(sds)
    rsd = np.sqrt(np.mean(sds ** 2))
    msd = np.amax(sds)
    return asd, rsd, msd
