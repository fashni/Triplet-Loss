import tensorflow as tf

def load_image(image_path, target_shape):
  h, w, c = target_shape
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=c, dtype=tf.float32)
  # img = tf.image.resize(img, (h, w))
  img = tf.image.resize_with_pad(img, h, w)
  return img

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
