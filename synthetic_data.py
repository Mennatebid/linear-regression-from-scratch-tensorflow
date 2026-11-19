import tensorflow as tf
import random
"""// Generating data // """

#setting parameters 
true_weights = tf.constant([2.0, -3.4])
true_bias = 4.2

#setting number of datapoints of training and validation sets and normally distributed noise
num_train = 1000
num_val = 1000
noise_std = 0.01

n = num_train + num_val # total examples of the dataset

X = tf.random.normal((n,true_weights.shape[0]))

print("Weights:", true_weights)

w_column = tf.reshape(true_weights,(-1,1))
noise = tf.random.normal((n,1))*noise_std

y = tf.matmul(X, w_column) + true_bias + noise

x_train = X[:num_train]
x_val = X[num_train:]

y_train = y[:num_train]
y_val = y[num_train:]

"""// Building data loader //
def generate_loader(X,y, batch_size=32, train=True):
    indices = list(range(len(X)))
    if train:
        random.shuffle(indices)
    for i in range(0,len(indices),batch_size):
        batch_indices = indices[i:i+batch_size]
        j = tf.constant(batch_indices)
        x_batch = tf.gather(X,j)
        y_batch = tf.gather(y,j)
        yield x_batch,y_batch

# Create a training generator
loader = generate_loader(
    x_train,
    y_train,
    batch_size=32,
    train=True
)

# Retrieve the first minibatch
x_batch, y_batch = next(loader)

print("Features shape:", x_batch.shape)
print("Labels shape:", y_batch.shape)

# Retrieve the second minibatch
x_batch2, y_batch2 = next(loader)

print("Second batch features:", x_batch2.shape)
print("Second batch labels:", y_batch2.shape)      """

"""// Data loader using tensorflow //"""

def tf_data_loader(X,y,batch_size=32,train=True):
    dataset= tf.data.Dataset.from_tensor_slices((X,y))
    
    if train:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size)
    
    return dataset

train_loader = tf_data_loader(x_train,y_train,batch_size=32,train=True)
val_loader = tf_data_loader(x_val,y_val,batch_size=32,train=False)

iterator = iter(train_loader)

x_batch,y_batch = next(iterator)

print(x_batch.shape)
print(y_batch.shape)

        