import tensorflow as tf

class TripletLossModel(tf.keras.Model):
  def __init__(self, model, margin=0.2):
    super(TripletLossModel, self).__init__()
    self.model = model
    self.margin = margin

  def compile(self, optimizer, weighted_metrics=[]):
    super(TripletLossModel, self).compile(optimizer=optimizer, weighted_metrics=weighted_metrics)
    self.optimizer = optimizer
    self.weighted_metrics = weighted_metrics if weighted_metrics else []

  def train_step(self, data):
    anchors, positives, negatives = data
    with tf.GradientTape() as tape:
      anchor_embeddings, positive_embeddings, negative_embeddings = self.model([anchors, positives, negatives], training=True)
      loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return {"loss": loss}

  def test_step(self, data):
    anchors, positives, negatives = data
    anchor_embeddings, positive_embeddings, negative_embeddings = self.model([anchors, positives, negatives], training=False)
    loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings, self.margin)
    return {"loss": loss}

  @staticmethod
  def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)
    return loss


def build_triplet_model(input_shape, base_model, margin=0.2):
  input_a = tf.keras.layers.Input(shape=(input_shape), name="input_a")
  input_p = tf.keras.layers.Input(shape=(input_shape), name="input_p")
  input_n = tf.keras.layers.Input(shape=(input_shape), name="input_n")

  output_a = base_model(input_a)
  output_p = base_model(input_p)
  output_n = base_model(input_n)

  model = tf.keras.models.Model(inputs=[input_a, input_p, input_n], outputs=[output_a, output_p, output_n])
  triplet_model = TripletLossModel(model, margin=margin)

  return triplet_model
