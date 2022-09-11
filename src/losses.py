import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class CustomLoss:
    # pos: exponential for positive example
    # neg: sum of exponentials for negative examples
    # N : number of negative examples
    # temperature : temperature scaling
    # tau : class probability
    def __init__(
            self,
            temperature=0.5,
            tau=0.01,
            beta=2,
            elimination_th=0,
            elimination_topk=0.1,
            lambd = 3.9e-3,
            scale_loss= 1/32,
            attraction=False
    ):
        self.temperature = temperature,
        self.tau = tau,
        self.beta = beta,
        self.elimination_th = elimination_th,
        self.elimination_topk = elimination_topk,
        self.attraction = attraction
        self.lambd = lambd
        self.scale_loss = scale_loss
        # Please double-check `reduction` parameter
        self.criterion = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.SUM)



    def get_loss_fn(self, loss_type):
        loss = None
        # Info-NCE
        if loss_type == "nce":
            def loss(ytrue, ypred):
                all_sim = K.exp(ypred / self.temperature)
                logits = tf.divide(
                    tf.linalg.tensor_diag_part(all_sim), K.sum(all_sim, axis=1))
                print(logits)
                lbl = np.ones(ypred.shape[0])
                error = self.criterion(y_pred=logits, y_true=lbl)
                return error

        # Debiased Contrastive Learning
        elif loss_type in ["dcl", "harddcl"]:
            def loss(ytrue, ypred):
                # dcl: from Debiased Contrastive Learning paper: https://github.com/chingyaoc/DCL/
                # harddcl: from ICLR2021 paper: Contrastive LEarning with Hard Negative Samples
                # https://www.groundai.com/project/contrastive-learning-with-hard-negative-samples
                # reweight = (beta * neg) / neg.mean()
                # Neg = max((-N * tau_plus * pos + reweight * neg).sum() / (1 - tau_plus), e ** (-1 / t))
                # hard_loss = -log(pos.sum() / (pos.sum() + Neg))
                N = ypred.shape[0]
                all_sim = K.exp(ypred / self.temperature)
                pos_sim = tf.linalg.tensor_diag_part(all_sim)

                tri_mask = np.ones(N ** 2, dtype=np.bool).reshape(N, N)
                tri_mask[np.diag_indices(N)] = False
                neg_sim = tf.reshape(tf.boolean_mask(all_sim, tri_mask), [N, N - 1])

                reweight = 1.0
                if loss_type == "harddcl":
                    reweight = (self.beta * neg_sim) / tf.reshape(tf.reduce_mean(neg_sim, axis=1), [-1, 1])
                if self.beta == 0:
                    reweight = 1.0

                Ng = tf.divide(
                    tf.multiply(self.tau[0] * (1 - N), pos_sim) + K.sum(tf.multiply(reweight, neg_sim), axis=-1),
                    (1 - self.tau[0]))
                print(Ng)
                # constrain (optional)
                Ng = tf.clip_by_value(Ng, clip_value_min=(N - 1) * np.e ** (-1 / self.temperature[0]),
                                      clip_value_max=tf.float32.max)
                error = K.mean(- tf.math.log(pos_sim / (pos_sim + Ng)))
                return error


        # Contrasting More than two dimenstions
        elif loss_type == "cocoa":
            def loss(ytrue, ypred):
                batch_size, dim_size = ypred.shape[1], ypred.shape[0]

                # Positive Pairs
                pos_error=[]
                for i in range(batch_size):
                    sim = tf.exp(tf.linalg.matmul(ypred[:,i,:], ypred[:,i,:], transpose_b=True)/self.temperature)
                    tri_mask = np.ones(dim_size ** 2, dtype=np.bool).reshape(dim_size, dim_size)
                    tri_mask[np.diag_indices(dim_size)] = False
                    off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [dim_size, dim_size - 1])
                    pos_error.append(tf.reduce_sum(off_diag_sim))
                # Negative pairs
                neg_error = 0
                for i in range(dim_size):
                    sim = tf.cast(tf.linalg.matmul(ypred[i], ypred[i], transpose_b=True), dtype=tf.dtypes.float32)
                    sim = tf.exp(sim/self.temperature)
                    tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
                    tri_mask[np.diag_indices(batch_size)] = False
                    off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
                    neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))

                logits = tf.divide(pos_error, pos_error+neg_error)
                lbl = np.ones(batch_size)
                error = self.criterion(y_pred=logits, y_true=lbl)
                return error

        elif loss_type == "cocoa2":
            def loss(ytrue, ypred):
                batch_size, dim_size = ypred.shape[1], ypred.shape[0]
                # Positive Pairs
                pos_error = []
                for i in range(batch_size):
                    sim = tf.linalg.matmul(ypred[:, i, :], ypred[:, i, :], transpose_b=True)
                    sim = tf.subtract(tf.ones([dim_size, dim_size], dtype=tf.dtypes.float32), sim)
                    sim = tf.exp(sim/self.temperature)
                    pos_error.append(tf.reduce_mean(sim))
                # Negative pairs
                neg_error = 0
                for i in range(dim_size):
                    sim = tf.cast(tf.linalg.matmul(ypred[i], ypred[i], transpose_b=True), dtype=tf.dtypes.float32)
                    sim = tf.exp(sim / self.temperature)
                    # sim = tf.add(sim, tf.ones([batch_size, batch_size]))
                    tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
                    tri_mask[np.diag_indices(batch_size)] = False
                    off_diag_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [batch_size, batch_size - 1])
                    neg_error += (tf.reduce_mean(off_diag_sim, axis=-1))

                error = tf.multiply(tf.reduce_sum(pos_error),self.scale_loss) + self.lambd * tf.reduce_sum(neg_error)

                return error

        elif loss_type == "mse":
            def loss(ytrue, ypred):
                reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(ytrue, ypred)))
                return reconstruction_error

        else:
            raise ValueError("Undefined loss function.")

        return loss


def mse_loss(model, original):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
  return reconstruction_error