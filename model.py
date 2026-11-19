from synthetic_data import train_loader
import tensorflow as tf
from loss import sqr_loss
from optimizer import sgd

num_inputs = 2
sigma = 0.01

w = tf.Variable(
    tf.random.normal((num_inputs,1),stddev= sigma), trainable=True
)

b= tf.Variable(tf.zeros(1),trainable=True)

def forward_prop(X,y,b):
    return tf.matmul(X,y)+b



