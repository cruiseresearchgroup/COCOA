import tensorflow as tf
import numpy as np


# ------------------- Helper utils ---------------------- #
def get_ts_modality_encoder(input_shape,
                            signal_channel,
                            modality_name,
                            filters,
                            code_size,
                            l2_rate):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    input = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv1D(filters=2 * filters,
                               kernel_size=10,
                               activation="linear",
                               padding="same",
                               strides=4,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(input)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=filters,
                               kernel_size=8,
                               activation="linear",
                               padding="same",
                               strides=2,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1])(x)

    x = tf.keras.layers.Conv1D(filters=code_size,
                               kernel_size=4,
                               activation="linear",
                               padding="same",
                               strides=1,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_rate),
                               kernel_initializer=initializer)(x)

    x = tf.keras.layers.PReLU(shared_axes=[1])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(code_size, activation="linear")(x)

    #output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    return tf.keras.models.Model(input, output, name=modality_name)


# ------------------------------------------------------------------------- #
def create_self_supervised_network(win_size, embedding_dim, modality_name, modality_dim, modality_filters=24,
                                   temperature=0.1, mode="ssl",loss_fn="cocoa"):
    # modality specific encoders
    mod_encoder = []
    mod_input = []
    for m in range(len(modality_name)):
        channels = modality_dim[m]
        input_shape = (win_size, channels)
        encoder = get_ts_modality_encoder(input_shape,
                                          channels,
                                          modality_name=modality_name[m],
                                          filters=modality_filters,
                                          code_size=embedding_dim,
                                          l2_rate=1e-4)

        mod_input.append(tf.keras.layers.Input(shape=input_shape))

        x_a = encoder(mod_input[-1])

        #x_a = tf.keras.layers.GlobalMaxPooling1D()(x_a)

        mod_encoder.append(x_a)

    embedding_model = tf.keras.Model(mod_input, mod_encoder)
    if mode in ["ssl", "fine"]:
        return ContrastiveModel(embedding_model, loss_fn, temperature)
    else:
        return embedding_model


# ------------------------------------------------------------------------- #
class DotProduct(tf.keras.layers.Layer):
    def call(self, x, y):
        x = tf.nn.l2_normalize(x, axis=-1)
        y = tf.nn.l2_normalize(y, axis=-1)
        return tf.linalg.matmul(x, y, transpose_b=True)


# ------------------------------------------------------------------------- #
class ContrastiveModel(tf.keras.Model):
    def __init__(self, embedding_model, loss_fn, temperature=1.0, **kwargs):
        super().__init__()
        self.embedding_model = embedding_model
        self._temperature = temperature
        self._similarity_layer = DotProduct()
        self._lossfn = loss_fn

    def train_step(self, data):
        with tf.GradientTape() as tape:
            modality_embeddings = self.embedding_model(data, training=True)

            sparse_labels = tf.range(tf.shape(modality_embeddings[0])[0])

            pred = modality_embeddings
            pred = tf.nn.l2_normalize(tf.stack(pred), axis=-1)

            loss = self.compiled_loss(sparse_labels, pred)
            loss += sum(self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {m.name: m.result() for m in self.metrics}

    def call(self, input):
        return self.embedding_model(input)
