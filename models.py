import tensorflow as tf

def create_model(num_characters):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(100, 200, 3)),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(num_characters * 10, activation="softmax"),
        tf.keras.layers.Reshape((10, num_characters))
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam())
    
    return model