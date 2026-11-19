import tensorflow as tf 

def sgd(params, grads, lr):
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad)