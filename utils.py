import numpy as np
from tqdm import tqdm


def batched(iterable, n):
  for i in range(0, len(iterable), n):
    yield iterable[i:i + n]


def get_embeddings(model, images, batch_size=32, verbose=0):
  return model.predict(images, batch_size=batch_size, verbose=verbose)


def get_pairwise_similarity(embeddings, labels, squared=False, norm=False):
  assert labels.shape[0] == embeddings.shape[0]
  n = labels.shape[0]
  triu_idx = np.triu_indices(n, k=1)

  preds_matrix = -pairwise_distances(embeddings, squared)
  labls_matrix = (labels[:, None] == labels[None, :]).astype(int)
  y_true = labls_matrix[triu_idx]
  y_pred = preds_matrix[triu_idx]

  if norm:
    y_pred = normalise_value(y_pred)
  return y_true, y_pred


def get_images_and_labels(dataset, n_batches=None):
  if n_batches is not None:
    dataset = dataset.take(n_batches)
  images, labels = [], []
  for image, label in tqdm(dataset):
    images.append(image)
    labels.append(label)
  return np.concatenate(images), np.concatenate(labels)


def get_metrics(y_true, y_pred):
  sort_idx = y_pred.argsort()
  preds = y_pred[sort_idx]
  labls = y_true[sort_idx]

  tp = labls[::-1].cumsum()[::-1]
  fp = (1 - labls[::-1]).cumsum()[::-1]
  fn = labls.sum() - tp
  tn = (1 - labls).sum() - fp

  prc = tp / (tp + fp)
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  acc = (tp + tn) / (tp + tn + fp + fn)
  f1 = 2 * prc * tpr / (prc + tpr)

  roc_auc = np.trapz(tpr[::-1], fpr[::-1])

  thres, idx = np.unique(preds, return_index=True)
  return fpr[idx], tpr[idx], prc[idx], acc[idx], f1[idx], thres, roc_auc


def distances(embedding1, embedding2, axis=None, squared=False):
  distances = np.sum(np.square(embedding1 - embedding2), axis=axis)
  if not squared:
    distances = np.sqrt(distances)
  return distances


def pairwise_distances(embeddings, squared=False):
  dot_product = np.matmul(embeddings, embeddings.T)
  square_norm = np.diagonal(dot_product)
  distances = square_norm[None, :] - 2.0 * dot_product + square_norm[:, None]
  if not squared:
    distances = np.sqrt(distances)
  return distances


def normalise_value(array): # [min(array), 0] -> [0, 1]
  return 1 - (array / array.min())
