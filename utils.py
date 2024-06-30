import numpy as np
from tqdm import tqdm


def batched(iterable, n):
  for i in range(0, len(iterable), n):
    yield iterable[i:i + n]


def compute_preds(model, images, labels, bs=32, verbose=0):
  embeddings = model.predict(images, batch_size=bs, verbose=verbose)
  preds_matrix = -pairwise_distances(embeddings)
  triu_idx = np.triu_indices(labels.shape[0], k=1)
  y_true = (labels[:, None] == labels[None, :])[triu_idx].astype(int)
  return y_true, preds_matrix[triu_idx], embeddings


def get_images_and_labels(dataset, n_batches=None):
  if n_batches is not None:
    dataset = dataset.take(n_batches)
  images, labels = [], []
  for image, label in tqdm(dataset):
    images.append(image)
    labels.append(label)
  return np.concatenate(images), np.concatenate(labels)


def compute_metrics(y_true, y_pred):
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


def compute_distances(embedding1, embedding2, axis=None):
  return ((embedding1 - embedding2)**2).sum(axis=axis)


def pairwise_distances(embeddings):
  dot_product = np.matmul(embeddings, embeddings.T)
  square_norm = np.diagonal(dot_product)
  distances = square_norm[None, :] - 2.0 * dot_product + square_norm[:, None]
  return distances


def normalise_value(array): # [min(array), 0] -> [0, 1]
  return 1 - (array / array.min())
