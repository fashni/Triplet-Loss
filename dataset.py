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


class BaseImageGenerator:
  def __init__(self, base_path, batch_size=32, input_shape=(160, 160, 3), augment=True, augment_fn=None, name="dataset"):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.height, self.width, self.channels = input_shape
    self.augment = augment
    self.augment_fn = augment_fn or augment_image
    self.images, self.names = self._load_paths(base_path)
    self.name = name

  def _load_paths(self, base_path):
    base_path = Path(base_path)
    if base_path.is_dir():
      names = [c.name for c in base_path.iterdir() if c.is_dir()]
      images = {c: [str(img) for img in (base_path / c).iterdir() if img.is_file()] for c in names}
    elif base_path.is_file() and base_path.suffix.casefold() == ".json":
      with base_path.open("r") as f:
        images = json.load(f)
      names = list(images.keys())
    else:
      raise ValueError(f"Invalid path: {base_path}")
    return images, names

  def _load_image(self, img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=self.input_shape[-1], dtype=tf.float32)
    return img

  def _preprocess_image(self, img):
    img = tf.image.resize(img, (self.input_shape[:2]))
    # img = tf.image.resize_with_pad(img, target_height, target_width)
    if self.augment:
      return self.augment_fn(img)
    return img

  def _generator(self):
    raise NotImplementedError("Subclasses should implement this method.")

  def get_dataset(self):
    dataset = tf.data.Dataset.from_generator(
      self._generator,
      output_signature=self._get_output_signature()
    )
    dataset = dataset.batch(self.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  def _get_output_signature(self):
    raise NotImplementedError("Subclasses should implement this method.")


class TripletGenerator(BaseImageGenerator):
  def _get_triplet(self):
    name = random.choice(self.names)
    while len(self.images[name]) < 2:
      name = random.choice(self.names)
    anchor, positive = random.sample(self.images[name], 2)
    negative_name = random.choice([c for c in self.names if c != name])
    negative = random.choice(self.images[negative_name])
    return anchor, positive, negative

  def _load_and_preprocess_triplet(self, triplet):
    imgs = [self._load_image(img_path) for img_path in triplet]
    imgs = [self._preprocess_image(img) for img in imgs]
    anchor_img, positive_img, negative_img = imgs
    return anchor_img, positive_img, negative_img

  def _generator(self):
    while True:
      triplets = [self._get_triplet() for _ in range(self.batch_size)]
      for triplet in triplets:
        yield self._load_and_preprocess_triplet(triplet)

  def _get_output_signature(self):
    return (
      tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
      tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
      tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
    )


class ImageGenerator(BaseImageGenerator):
  def __init__(self, base_path, batch_size=32, input_shape=(160, 160, 3), augment=True, augment_fn=None, name="dataset"):
    super().__init__(base_path, batch_size, input_shape, augment, augment_fn, name)
    self.image_pool = self._initialize_image_pool()
    self.current_index = 0

  def _initialize_image_pool(self):
    image_pool = [(img, name) for name in self.names for img in self.images[name]]
    random.shuffle(image_pool)
    return image_pool

  def _get_next_batch(self):
    batch = []
    while len(batch) < self.batch_size and self.current_index < len(self.image_pool):
      img, name = self.image_pool[self.current_index]
      batch.append((img, name))
      self.current_index += 1
    return batch

  def _load_and_preprocess_image(self, image_path):
    img = self._load_image(image_path)
    img = self._preprocess_image(img)
    return img

  def _generator(self):
    while self.current_index < len(self.image_pool):
      batch = self._get_next_batch()
      if not batch:
        break
      for img, name in batch:
        yield self._load_and_preprocess_image(img), name

  def _get_output_signature(self):
    return (
      tf.TensorSpec(shape=self.input_shape, dtype=tf.float32),
      tf.TensorSpec(shape=(), dtype=tf.string)
    )
