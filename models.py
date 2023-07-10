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

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = tf.keras.layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x

class AddPositionEmbs(tf.keras.layers.Layer):
  """inputs are image patches
  Custom layer to add positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

def mlp_block_f(mlp_dim, inputs):
  x = tf.keras.layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = tf.keras.layers.Dropout(rate=0.1)(x) # dropout rate is from original paper,
  x = tf.keras.layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x) # check GELU paper
  x = tf.keras.layers.Dropout(rate=0.1)(x)
  return x

def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
  x = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
  # self attention multi-head, dropout_rate is from original implementation
  x = tf.keras.layers.Add()([x, inputs]) # 1st residual part

  y = tf.keras.layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = tf.keras.layers.Add()([y, x]) #2nd residual part
  return y_1

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = tf.keras.layers.Dropout(rate=0.2)(x)
  for _ in range(num_layers):
    x = Encoder1Dblock_f(num_heads, mlp_dim, x)

  encoded = tf.keras.layers.LayerNormalization(name='encoder_norm')(x)
  return encoded

def build_ViT(num_characters):
  inputs = tf.keras.layers.Input(shape=(100, 200, 1))
  # generate patches with conv layer
  patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs)

  ######################################
  # ready for the transformer blocks
  ######################################
  encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)

  #####################################
  #  final part (mlp to classification)
  #####################################
  #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
  # similar to the GAP, this is from original Google GitHub

  logits = tf.keras.layers.Dense(units=num_characters * 6, activation = 'softmax', name='head')(im_representation)
  # !!! important !!! activation is linear

  outputs = tf.keras.layers.Reshape((6, num_characters))(logits)
  # class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
  #                 'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website

  final_model = tf.keras.Model(inputs = inputs, outputs = outputs)

  return final_model