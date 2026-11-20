import tensorflow as tf
from synthetic_data import train_loader,val_loader

lambd = 0.01

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.01
            ),
            kernel_regularizer=tf.keras.regularizers.L2(lambd/2),
            bias_initializer="zeros"
        )
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
    loss = tf.keras.losses.MeanSquaredError()
)

model.fit(
    train_loader, epochs= 10, validation_data=val_loader
)

model.summary()