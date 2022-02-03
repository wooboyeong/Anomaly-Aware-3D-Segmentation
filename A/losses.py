import tensorflow as tf

def A_loss(y_true, y_pred):
    output_diff = tf.square(y_true[..., 0] - y_pred[..., 0])
    mse_output = tf.reduce_mean(output_diff, axis=(1, 2, 3))
    
    err_true = tf.square(y_true[..., 1] - y_true[..., 0])
    err_pred = tf.square(y_true[..., 1] - y_pred[..., 0])
    
    error_diff = tf.square(err_true - err_pred)
    mse_error = tf.reduce_mean(error_diff, axis=(1, 2, 3))
    
    return mse_output + mse_error
