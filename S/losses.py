import tensorflow as tf
import metrics
 
def dice_loss(y_true, y_pred):
    return 1.0 - metrics.multiclass_dice()(y_true, y_pred)

def ce_loss(y_true, y_pred, beta=99.0):
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return (1.0 + beta * y_true[..., -1]) * ce(y_true[..., :-1], y_pred)

def S_loss(y_true, y_pred, alpha=10.0):
    return dice_loss(y_true, y_pred) + alpha * ce_loss(y_true, y_pred)
