import tensorflow as tf
from synthetic_data import train_loader,val_loader

lambd = 0.01

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(
            units=1,
            kernel_regularizer= tf.keras.regularizers.L2(lambd/2))
    ]
)
model.compile(
    optimizer = tf.keras.optimizers.SGD(
        learning_rate= 0.01
    ),
    loss = "mse",
    metrics = ["mse"]
)
model.fit(
    train_loader, epochs = 10, validation_data = val_loader
)
model.summary()