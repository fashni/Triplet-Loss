import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm


def load_image(image_path, target_shape):
  h, w, c = target_shape
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=c, dtype=tf.float32)
  img = tf.image.resize(img, (h, w))

  # Padding causes error during training, can't figure out why
  # img = tf.image.resize_with_pad(img, h, w)
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


def compute_preds(model, imgs, labels, bs=32, verbose=0):
  triu_idx = np.triu_indices(imgs.shape[0], k=1)
  y_true = (labels[:, None] == labels[None, :])[triu_idx].astype(int)
  embeddings = model.predict(imgs, batch_size=bs, verbose=verbose)
  preds_matrix = -compute_dists(embeddings[:, np.newaxis, :], embeddings[np.newaxis, :, :], axis=2)
  return y_true, preds_matrix[triu_idx], embeddings


def compute_metrics(y_true, y_pred):
  auc = roc_auc_score(y_true, y_pred)
  fpr, tpr, thres = roc_curve(y_true, y_pred)
  j = (tpr-fpr).argmax()
  cm = confusion_matrix(y_true, y_pred>thres[j])
  acc = cm.diagonal().sum() / cm.sum()
  return fpr, tpr, thres, j, cm, acc, auc


def compute_dists(a, b, axis=None):
  return ((a-b)**2).sum(axis=axis)
