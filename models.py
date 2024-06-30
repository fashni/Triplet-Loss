import tensorflow as tf
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Concatenate, Conv2D,
                                     Dense, Dropout, Flatten,
                                     GlobalAveragePooling2D, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.regularizers import l2

from triplet_loss import (batch_all_triplet_loss, batch_hard_triplet_loss,
                          siamese_triplet_loss)


class SiameseModel(tf.keras.Model):
  def __init__(self, model, margin, squared):
    super(SiameseModel, self).__init__()
    self.model = model
    self.base_model = model.layers[-1]
    self.margin = margin
    self.squared = squared

  def save_weights(self, *args, **kwargs):
    self.base_model.save_weights(*args, **kwargs)

  def load_weights(self, *args, **kwargs):
    self.base_model.load_weights(*args, **kwargs)

  def summary(self, *args, **kwargs):
    self.model.summary(*args, **kwargs)

  def predict(self, *args, **kwargs):
    return self.base_model.predict(*args, **kwargs)

  def compile(self, optimizer, weighted_metrics=[]):
    super(SiameseModel, self).compile(optimizer=optimizer, weighted_metrics=weighted_metrics)
    self.optimizer = optimizer
    self.weighted_metrics = weighted_metrics if weighted_metrics else []

  def train_step(self, data):
    anchors, positives, negatives = data
    with tf.GradientTape() as tape:
      a_embeddings, p_embeddings, n_embeddings = self.model([anchors, positives, negatives], training=True)
      loss = siamese_triplet_loss(a_embeddings, p_embeddings, n_embeddings, self.margin, self.squared)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return {"loss": loss}

  def test_step(self, data):
    anchors, positives, negatives = data
    a_embeddings, p_embeddings, n_embeddings = self.model([anchors, positives, negatives], training=False)
    loss = siamese_triplet_loss(a_embeddings, p_embeddings, n_embeddings, self.margin, self.squared)
    return {"loss": loss}


class OnlineTripletModel(tf.keras.Model):
  def __init__(self, strategy, margin, squared, *args, **kwargs):
    super().__init__(*args, **kwargs)
    assert strategy in ["batch_all", "batch_hard"]
    self.strategy = strategy
    self.margin = margin
    self.squared = squared

  def _loss_fn(self, y_true, y_pred):
    if self.strategy == "batch_all":
      loss, _ = batch_all_triplet_loss(y_true, y_pred, self.margin, self.squared)
    else:
      loss = batch_hard_triplet_loss(y_true, y_pred, self.margin, self.squared)
    return loss

  def train_step(self, data):
    images, labels = data
    with tf.GradientTape() as tape:
      embeddings = self(images, training=True)
      loss = self._loss_fn(labels, embeddings)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return {"loss": loss}

  def test_step(self, data):
    images, labels = data
    embeddings = self(images, training=False)
    loss = self._loss_fn(labels, embeddings)
    return {"loss": loss}


# Helper functions
def _validate(strategy):
  valid_strategies = ["siamese", "batch_all", "batch_hard"]
  assert strategy is not None, f"Strategy must provided, valid strategies: {valid_strategies}."
  assert strategy in valid_strategies, f"Invalid strategy, valid strategies: {valid_strategies}."


def _build_model(input_layer, output_layer, strategy, margin, squared=False, name="model"):
  if strategy.startswith("batch"):
    return OnlineTripletModel(strategy, margin, squared,
                              inputs=input_layer, outputs=output_layer, name=name)

  base_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name=name)
  model = _build_siamese_model(input_layer.shape[1:], base_model, margin, squared)
  return model


def _build_siamese_model(input_shape, base_model, margin, squared):
  input_a = Input(shape=(input_shape), name="input_a")
  input_p = Input(shape=(input_shape), name="input_p")
  input_n = Input(shape=(input_shape), name="input_n")

  output_a = base_model(input_a)
  output_p = base_model(input_p)
  output_n = base_model(input_n)

  model = tf.keras.models.Model(inputs=[input_a, input_p, input_n], outputs=[output_a, output_p, output_n])
  return SiameseModel(model, margin=margin, squared=squared)


def _conv2d_bn(x, filters, kernel_size, strides=1, padding='same', name=None):
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
  x = BatchNormalization(scale=False)(x)
  x = Activation('relu')(x)
  return x


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
  if block_type == 'block35':
    branch_0 = _conv2d_bn(x, 32, 1)
    branch_1 = _conv2d_bn(x, 32, 1)
    branch_1 = _conv2d_bn(branch_1, 32, 3)
    branch_2 = _conv2d_bn(x, 32, 1)
    branch_2 = _conv2d_bn(branch_2, 48, 3)
    branch_2 = _conv2d_bn(branch_2, 64, 3)
    branches = [branch_0, branch_1, branch_2]
  elif block_type == 'block17':
    branch_0 = _conv2d_bn(x, 192, 1)
    branch_1 = _conv2d_bn(x, 128, 1)
    branch_1 = _conv2d_bn(branch_1, 160, [1, 7])
    branch_1 = _conv2d_bn(branch_1, 192, [7, 1])
    branches = [branch_0, branch_1]
  elif block_type == 'block8':
    branch_0 = _conv2d_bn(x, 192, 1)
    branch_1 = _conv2d_bn(x, 192, 1)
    branch_1 = _conv2d_bn(branch_1, 224, [1, 3])
    branch_1 = _conv2d_bn(branch_1, 256, [3, 1])
    branches = [branch_0, branch_1]
  else:
    raise ValueError('Unknown block type: {}'.format(block_type))

  mixed = Concatenate(axis=-1)(branches)
  up = Conv2D(tf.keras.backend.int_shape(x)[-1], 1, activation=None, use_bias=True)(mixed)
  up = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, arguments={'scale': scale})([x, up])
  if activation is not None:
    up = Activation(activation)(up)
  return up


def _inception_module(x, filters, name=None):
  f_1x1, f_3x3, f_5x5, f_pool_proj = filters
  conv_1x1 = Conv2D(f_1x1, (1, 1), padding='same', activation='relu')(x)

  conv_3x3 = Conv2D(f_3x3[0], (1, 1), padding='same', activation='relu')(x)
  conv_3x3 = Conv2D(f_3x3[1], (3, 3), padding='same', activation='relu')(conv_3x3)

  conv_5x5 = Conv2D(f_5x5[0], (1, 1), padding='same', activation='relu')(x)
  conv_5x5 = Conv2D(f_5x5[1], (5, 5), padding='same', activation='relu')(conv_5x5)

  pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
  pool_proj = Conv2D(f_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

  output = Concatenate(axis=-1, name=name)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
  return output


# Network definitions
def siamnet(input_shape=(105, 105, 3), embedding_size=128, margin=0.5, squared=False, strategy=None, weight_decay=0.0005):
  """
  https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
  """
  _validate(strategy)
  input_layer = Input(shape=input_shape)
  x = Conv2D(64, (10, 10), kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), name='conv1')(input_layer)
  x = BatchNormalization(name='bn1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(name='maxpool1')(x)

  x = Conv2D(128, (7, 7), kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), name='conv2')(x)
  x = BatchNormalization(name='bn2')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(name='maxpool2')(x)

  x = Conv2D(128, (4, 4), kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), name='conv3')(x)
  x = BatchNormalization(name='bn3')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(name='maxpool3')(x)

  x = Conv2D(256, (4, 4), kernel_initializer="he_uniform", kernel_regularizer=l2(weight_decay), name='conv4')(x)
  x = BatchNormalization(name='bn4')(x)
  x = Activation('relu')(x)

  x = Flatten()(x)
  x = Dense(embedding_size, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay), name="fc1")(x)
  # x = Dropout(0.5)(x)
  # x = Dense(embedding_size, activation=None, kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay), name="fc2")(x)
  x = BatchNormalization(name='bn5')(x)

  # Final L2 normalization
  output = Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x)
  name = "siamnet"
  return _build_model(input_layer, output, strategy, margin, squared, name)


def facenet(input_shape=(160, 160, 3), embedding_size=128, margin=0.5, squared=False, strategy=None, weight_decay=0.0005):
  """
  https://arxiv.org/abs/1503.03832
  """
  _validate(strategy)
  input_layer = Input(shape=input_shape)

  # Layer 1
  x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(weight_decay), name='conv1')(input_layer)
  x = BatchNormalization(name='bn1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpool1')(x)

  # Layer 2
  x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv2')(x)
  x = BatchNormalization(name='bn2')(x)
  x = Activation('relu')(x)

  # Layer 3
  x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv3')(x)
  x = BatchNormalization(name='bn3')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpool2')(x)

  # Layer 4
  x = Conv2D(192, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv4')(x)
  x = BatchNormalization(name='bn4')(x)
  x = Activation('relu')(x)

  # Layer 5
  x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv5')(x)
  x = BatchNormalization(name='bn5')(x)
  x = Activation('relu')(x)

  # Layer 6
  x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv6')(x)
  x = BatchNormalization(name='bn6')(x)
  x = Activation('relu')(x)

  # Layer 7
  x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv7')(x)
  x = BatchNormalization(name='bn7')(x)
  x = Activation('relu')(x)

  # Layer 8
  x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv8')(x)
  x = BatchNormalization(name='bn8')(x)
  x = Activation('relu')(x)

  # Layer 9
  x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv9')(x)
  x = BatchNormalization(name='bn9')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='maxpool3')(x)

  # Layer 10
  x = Conv2D(4096, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay), name='conv10')(x)
  x = BatchNormalization(name='bn10')(x)
  x = Activation('relu')(x)
  x = GlobalAveragePooling2D(name='avg_pool')(x)

  # Fully connected layer
  x = Dense(embedding_size, kernel_regularizer=l2(weight_decay), name='fc1')(x)
  x = BatchNormalization(name='bn11')(x)
  x = Dropout(0.5)(x)

  # Final L2 normalization
  output = Lambda(lambda y: tf.math.l2_normalize(y, axis=1))(x)
  name = "facenet"
  return _build_model(input_layer, output, strategy, margin, squared, name)


def inception_resnet(input_shape=(160, 160, 3), embedding_size=128, margin=0.5, squared=False, strategy=None):
  _validate(strategy)
  input_layer = Input(shape=input_shape)
  x = _conv2d_bn(input_layer, 32, 3, strides=2, padding='valid')
  x = _conv2d_bn(x, 32, 3, padding='valid')
  x = _conv2d_bn(x, 64, 3)
  x = MaxPooling2D(3, strides=2)(x)
  x = _conv2d_bn(x, 80, 1, padding='valid')
  x = _conv2d_bn(x, 192, 3, padding='valid')
  x = MaxPooling2D(3, strides=2)(x)

  # Mixed 5b (Inception-A block)
  branch_0 = _conv2d_bn(x, 96, 1)
  branch_1 = _conv2d_bn(x, 48, 1)
  branch_1 = _conv2d_bn(branch_1, 64, 5)
  branch_2 = _conv2d_bn(x, 64, 1)
  branch_2 = _conv2d_bn(branch_2, 96, 3)
  branch_2 = _conv2d_bn(branch_2, 96, 3)
  branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
  branch_pool = _conv2d_bn(branch_pool, 64, 1)
  branches = [branch_0, branch_1, branch_2, branch_pool]
  x = Concatenate(axis=-1, name='mixed_5b')(branches)

  # 10x block35 (Inception-ResNet-A block)
  for block_idx in range(1, 11):
    x = _inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=block_idx)

  # Reduction-A block
  branch_0 = _conv2d_bn(x, 384, 3, strides=2, padding='valid')
  branch_1 = _conv2d_bn(x, 256, 1)
  branch_1 = _conv2d_bn(branch_1, 256, 3)
  branch_1 = _conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
  branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
  branches = [branch_0, branch_1, branch_pool]
  x = Concatenate(axis=-1, name='mixed_6a')(branches)

  # 20x block17 (Inception-ResNet-B block)
  for block_idx in range(1, 21):
    x = _inception_resnet_block(x, scale=0.1, block_type='block17', block_idx=block_idx)

  # Reduction-B block
  branch_0 = _conv2d_bn(x, 256, 1)
  branch_0 = _conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
  branch_1 = _conv2d_bn(x, 256, 1)
  branch_1 = _conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
  branch_2 = _conv2d_bn(x, 256, 1)
  branch_2 = _conv2d_bn(branch_2, 288, 3)
  branch_2 = _conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
  branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
  branches = [branch_0, branch_1, branch_2, branch_pool]
  x = Concatenate(axis=-1, name='mixed_7a')(branches)

  # 10x block8 (Inception-ResNet-C block)
  for block_idx in range(1, 10):
    x = _inception_resnet_block(x, scale=0.2, block_type='block8', block_idx=block_idx)
  x = _inception_resnet_block(x, scale=1.0, activation=None, block_type='block8', block_idx=10)

  x = _conv2d_bn(x, 1536, 1)
  x = GlobalAveragePooling2D(name='avg_pool')(x)

  # Final fully connected layer
  x = Dense(embedding_size, name='fc')(x)
  x = BatchNormalization(name='bn')(x)

  # Final L2 normalization
  output = Lambda(lambda y: tf.math.l2_normalize(y, axis=1), name='l2_norm')(x)
  name = "inception_resnet"
  return _build_model(input_layer, output, strategy, margin, squared, name)


def inception(input_shape=(160, 160, 3), embedding_size=128, margin=0.5, squared=False, strategy=None):
  _validate(strategy)
  input_layer = Input(shape=input_shape)

  # Initial layers
  x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input_layer)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)
  x = BatchNormalization(name='bn1')(x)
  x = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2')(x)
  x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv3')(x)
  x = BatchNormalization(name='bn2')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool2')(x)

  # Inception modules
  x = _inception_module(x, [64, (96, 128), (16, 32), 32], name='inception3a')
  x = _inception_module(x, [128, (128, 192), (32, 96), 64], name='inception3b')
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool3')(x)

  x = _inception_module(x, [192, (96, 208), (16, 48), 64], name='inception4a')
  x = _inception_module(x, [160, (112, 224), (24, 64), 64], name='inception4b')
  x = _inception_module(x, [128, (128, 256), (24, 64), 64], name='inception4c')
  x = _inception_module(x, [112, (144, 288), (32, 64), 64], name='inception4d')
  x = _inception_module(x, [256, (160, 320), (32, 128), 128], name='inception4e')
  x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool4')(x)

  x = _inception_module(x, [256, (160, 320), (32, 128), 128], name='inception5a')
  x = _inception_module(x, [384, (192, 384), (48, 128), 128], name='inception5b')

  # Final layers
  x = GlobalAveragePooling2D(name='avgpool')(x)
  x = Dropout(0.4)(x)
  x = Flatten()(x)
  x = Dense(embedding_size, name='fc1')(x)
  x = BatchNormalization(name='bn3')(x)
  x = Activation('relu')(x)

  # Final L2 normalization
  output = Lambda(lambda y: tf.math.l2_normalize(y, axis=1), name='l2_norm')(x)
  name = "inception_v2"
  return _build_model(input_layer, output, strategy, margin, squared, name)


if __name__ == "__main__":
  model = facenet((160,160,3), strategy="siamese")
  model.summary()
