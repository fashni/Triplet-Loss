import json
import random
from pathlib import Path

import tensorflow as tf


def augment_image(image):
  h, w, c = image.shape
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_brightness(image, max_delta=0.1)
  image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
  image = tf.image.random_hue(image, max_delta=0.1)
  image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
  image = tf.image.random_jpeg_quality(image, 65, 95)
  # image = tf.image.random_crop(image, size=[round(0.9*h), round(0.9*w), c])
  # image = tf.image.resize(image, size=[h, w])
  return image


class DatasetGenerator:
  def __init__(self, base_path, batch_size=32, input_shape=(160, 160, 3), augment=True, augment_fn=None, seed=None, shuffle=True):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.shuffle = shuffle
    self.augment = augment
    self.augment_fn = augment_fn or augment_image
    self.images, self.classes = self._load_paths(base_path)
    self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
    self.seed = seed
    if seed is not None:
      random.seed(seed)

  def _load_paths(self, base_path):
    base_path = Path(base_path)
    if base_path.is_dir():
      classes = [c.name for c in base_path.iterdir() if c.is_dir()]
      images = {c: [str(img) for img in (base_path / c).iterdir() if img.is_file()] for c in classes}
    elif base_path.is_file() and base_path.suffix.casefold() == ".json":
      with base_path.open("r") as f:
        images = json.load(f)
      classes = list(images.keys())
    else:
      raise ValueError(f"Invalid path: {base_path}")
    return images, classes

  def _load_image(self, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=self.input_shape[-1], dtype=tf.float32)
    return img

  def _preprocess_image(self, img):
    img = tf.image.resize(img, (self.input_shape[:2]))
    return img

  def _augment_image(self, img):
    return self.augment_fn(img)

  def _load_and_preprocess(self, image_path):
    img = self._load_image(image_path)
    img = self._preprocess_image(img)
    if self.augment:
      img = self._augment_image(img)
    return img

  def get_dataset(self):
    dataset = tf.data.Dataset.from_generator(
      self._generator,
      output_signature=self._get_output_signature()
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def _generator(self):
    raise NotImplementedError("Subclasses should implement this method.")

  def _get_output_signature(self):
    raise NotImplementedError("Subclasses should implement this method.")


class TripletGenerator(DatasetGenerator):
  def _get_triplet(self):
    name = random.choice(self.classes)
    while len(self.images[name]) < 2:
      name = random.choice(self.classes)
    anchor, positive = random.sample(self.images[name], 2)
    negative_name = random.choice([c for c in self.classes if c != name])
    negative = random.choice(self.images[negative_name])
    return anchor, positive, negative

  def _get_next_batch(self):
    paths = [self._get_triplet() for _ in range(self.batch_size)]
    anchors = [self._load_and_preprocess(path[0]) for path in paths]
    positives = [self._load_and_preprocess(path[1]) for path in paths]
    negatives = [self._load_and_preprocess(path[2]) for path in paths]
    return anchors, positives, negatives

  def _generator(self):
    while True:
      anchors, positives, negatives = self._get_next_batch()
      yield tf.stack(anchors), tf.stack(positives), tf.stack(negatives)

  def _get_output_signature(self):
    return (
      tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
      tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
      tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
    )


class BatchGenerator(DatasetGenerator):
  def __init__(self, base_path, **kwargs):
    super().__init__(base_path, **kwargs)
    self.current_index = 0
    self.image_pool, self.n_images = self._get_image_pool()
    self.n_batches = -(-self.n_images // self.batch_size)

  def _get_image_pool(self):
    image_pool = [(img, label) for label in self.classes for img in self.images[label]]
    if self.shuffle:
      random.shuffle(image_pool)
    return image_pool, len(image_pool)

  def _get_next_batch(self):
    batch_paths = self.image_pool[self.current_index:self.current_index + self.batch_size]
    batch_images = [self._load_and_preprocess(img_path) for img_path, _ in batch_paths]
    batch_labels = [self.class_to_index[label] for _, label in batch_paths]
    self.current_index += self.batch_size
    if self.current_index > len(self.image_pool):
      self.image_pool, _ = self._get_image_pool()
      self.current_index = 0
    return batch_images, batch_labels

  def _generator(self):
    while True:
      batch_images, batch_labels = self._get_next_batch()
      yield tf.stack(batch_images), tf.convert_to_tensor(batch_labels, dtype=tf.int32)

  def _get_output_signature(self):
    return (
      tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32),
      tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

  def get_dataset(self):
    dataset = tf.data.Dataset.from_generator(
      self._generator,
      output_signature=self._get_output_signature()
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).take(self.n_batches)
    return dataset
