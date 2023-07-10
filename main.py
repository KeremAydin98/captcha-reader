from preprocessing import *
import tensorflow as tf
from models import *
import config
import logging

preprocess = Preprocessing()

batch_size = 16
root_path = '/content/generated_captchas/'

all_images = os.listdir(root_path)
random.shuffle(all_images)
all_images = all_images[:100000]
train_images = all_images[:int(len(all_images) * 0.8)]
val_images = all_images[int(len(all_images) * 0.8):]


train_gen = get_data_generator(root_path, train_images, batch_size=batch_size)
val_gen = get_data_generator(root_path, val_images,batch_size=batch_size)

model = build_ViT(config.transformer_layers, config.patch_size, config.hidden_size, config.num_heads, config.mlp_dim, preprocess.n_characters)
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

model.fit(train_gen,
          epochs=20,
          validation_data=val_gen,
          steps_per_epoch=len(train_images) // batch_size,
          validation_steps=len(val_images) // batch_size,
          callbacks=[early_stop])




