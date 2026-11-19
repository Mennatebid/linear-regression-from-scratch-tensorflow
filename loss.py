import tensorflow as tf

def sqr_loss(y,y_hat):
    diff = y-y_hat
    sqrt_error = tf.square(diff)
    loss = tf.reduce_mean(sqrt_error)/2
    return loss
