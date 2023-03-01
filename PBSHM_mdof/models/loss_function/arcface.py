import tensorflow as tf

class ArcFaceLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, margin=0.5, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.mm = self.sin_m * margin  # minimum value to guarantee cos(theta+m) > 0
        self.threshold = tf.math.cos(tf.constant(1.0) - margin)

    def call(self, y_true, y_pred):
        # Normalize the feature vectors
        y_pred = tf.nn.l2_normalize(y_pred, axis=1)

        # Get the ground truth labels
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype=tf.int32)

        # Get the number of samples and the number of dimensions in the feature vectors
        n = tf.shape(y_pred)[0]
        d = tf.shape(y_pred)[1]

        # Compute the cosine similarity between the feature vectors and the weights
        w = tf.Variable(tf.random.normal((d, self.num_classes), stddev=0.01, dtype=tf.float32), trainable=True)
        w = tf.nn.l2_normalize(w, axis=0)
        logits = tf.matmul(y_pred, w)
        cos_t = tf.gather_nd(logits, tf.stack([tf.range(n), y_true[:, tf.newaxis]], axis=1))
        sin_t = tf.sqrt(1.0 - tf.square(cos_t))

        # Compute the final logit values using the ArcFace formula
        cos_mt = cos_t * self.cos_m - sin_t * self.sin_m
        cos_mt = tf.where(cos_t > self.threshold, cos_mt, cos_t - self.mm)
        logits = self.scale * cos_mt

        # Compute the cross-entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

        return tf.reduce_mean(loss)
