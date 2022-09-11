import argparse
import os
from datetime import datetime
import pandas as pd
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from utils.visualisation import EmbeddingVisualisation
from classifier import Classifier
from losses import CustomLoss
from utils.DataHelper import load_dataset
import model.cocoa as cocoa
import tensorflow as tf
from utils.ds_config import read_config
from tensorflow.keras.models import Model

import numpy as np

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
import time

# ----------------------------------------------------------------------
# Argument Processing

parser = argparse.ArgumentParser(description='interface of running experiments for COCOA')
parser.add_argument('--datapath', type=str, required=True, help='[ ./data ] prefix path to data directory')
parser.add_argument('--output', type=str, required=True, help='[ ./output ] prefix path to output directory')
parser.add_argument('--dataset', type=str, default='HASC', help='dataset name ')
parser.add_argument('--loss', type=str, default='nce', help='loss function ')
parser.add_argument('--sim', type=str, default='cosine', help='similarity metric ')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--tsne', type=int, default=0, help='[0|1] Visualize the representations using t-SNE')
# hyperparameters for grid search
parser.add_argument('--window', type=int, default=100, help='window size')
parser.add_argument('--labeleff', type=float, default=1.0, help='Label efficiency ratio for fine tuning')
parser.add_argument('--code', type=int, default=40, help='size of encoded features')
parser.add_argument('--beta', type=float, default=2,
                    help='parameter for FN loss function or threshold for FC loss function')
parser.add_argument('--epoch', type=int, default=1, help='max iteration for training')
parser.add_argument('--batch', type=int, default=16, help='batch_size for training')
parser.add_argument('--eval_freq', type=int, default=25, help='evaluation frequency per batch updates')
parser.add_argument('--temp', type=float, default=.1, help='Temperature parameter for NCE loss function')
parser.add_argument('--tau', type=float, default=.01, help='parameter for Debiased contrastive loss function')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--mode', type=str, default='ssl', help='sbase:supervised baseline \n '
                                                            'ssl:self supervised training with classification \n '
                                                            'fine: fine tuning the ssl part')
# -------------------------------------------------------------------
args = parser.parse_args()
config = read_config(args.dataset)
WIN = config["WINDOW"]
CODE_SIZE = config["CODE"]
optimizer = tf.keras.optimizers.Adam(lr=args.lr)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


train_name = args.mode + "_" + args.dataset + "_T" + str(args.temp) + "_WIN" + str(WIN) + \
             "_BS" + str(args.batch) + "_CS" + str(CODE_SIZE) + "_lr" + str(args.lr) + \
             "_LOSS" + args.loss + "_SIM" + args.sim + "_TAU" + str(args.tau) + "_BETA" + str(args.beta)

print("Experiment Name >>> " + train_name)

# sanity check
# -------------------------------------------------------------------
if not os.path.exists(args.output):
    os.mkdir(args.output)
OUTPUT_PATH = os.path.join(args.output, args.dataset)
MODEL_PATH = os.path.join(OUTPUT_PATH, "model")
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
    os.mkdir(os.path.join(OUTPUT_PATH, "plots"))
    os.mkdir(os.path.join(OUTPUT_PATH, "model"))

log_dir = os.path.join("..", "logs")

earlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                          min_delta=0.00009,
                                                          patience=5,
                                                          verbose=1,
                                                          mode="min",
                                                          restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="loss", patience=1, verbose=1, factor=0.5)
time_callback = TimeHistory()
report_path = os.path.join(OUTPUT_PATH, "report.csv")
# Fetch Data-set specific configuration
config = read_config(args.dataset)
# Report file header
df_columns = ['MODE', 'DS_NAME', 'TEMP', 'BATCH', 'CODE', 'WIN', 'LR', 'LABEL_EFF',
              'LOSS', 'SIM', 'TAU', 'BETA', 'SSL_TIME', 'TRN_LOSS',
              'TRN_ACC', 'TST-LOSS', 'TST_ACC', 'TRN_FSCORE', 'TST_FSCORE', 'MODALITY']

if os.path.exists(report_path) == False:
    # Create the pandas DataFrame
    df = pd.DataFrame([], columns=df_columns)
    df.to_csv(report_path, header=True)

ssl_time = 0
base_model = None
if (args.mode in ["ssl", "fine"]):
    # -----------------------------------------------------------------
    # 1 PREPARE SSL DATASET and OBJ FN
    # ----------------------------------------------------------------
    ssl_train_ds = load_dataset(args.datapath,
                                                        args.dataset,
                                                        config['WINDOW'],
                                                        args.batch,
                                                        mode="train")
    custom_loss_obj = CustomLoss(temperature=args.temp)
    loss_fn = custom_loss_obj.get_loss_fn(args.loss)
    # -----------------------------------------------------------------
    # 2 SSL TRAINING
    # -----------------------------------------------------------------
    self_supervised_model = cocoa.create_self_supervised_network(win_size=config["WINDOW"],
                                                                 embedding_dim=args.code,
                                                                 modality_name=config["MODALITY_NAME"],
                                                                 modality_dim=config['MODALITY_CH'],
                                                                 loss_fn=args.loss)

    self_supervised_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                                  loss=loss_fn,
                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    result = self_supervised_model.fit(ssl_train_ds,
                                       batch_size=args.batch,
                                       epochs=args.epoch,
                                       verbose=1,
                                       callbacks=[earlyStopping_callback, time_callback])
    ssl_time = np.sum(time_callback.times)
    base_model = self_supervised_model

    if args.tsne == 1:
        print("t-SNE visualization of learnt representation:")
        vis = EmbeddingVisualisation(args.dataset, os.path.join(OUTPUT_PATH, "plots"), class_size=config["CLASS"])

        data, modalities = load_dataset(args.datapath,
                                        args.dataset,
                                        config['WINDOW'],
                                        args.batch,
                                        mode="test", state="all")
        lbl = list(tf.concat([y for x, y in data], axis=0))

        encoder = Model(base_model.embedding_model.input,
                        base_model.embedding_model.output)
        embeddings = np.array(tf.concat([tf.concat(item, axis=-1) for item in [encoder(x) for x, y in data]], axis=0))

        vis.plot_tSNE(embeddings, lbl, train_name + "_selfsupervised", config['LABEL_NAME'])
        embeddings = np.array(tf.concat([tf.concat(item, axis=-1) for item in [x for x, y in data]], axis=0))
        embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2]))

        vis.plot_tSNE(embeddings, lbl, train_name + "_rawdata", config['LABEL_NAME'])

# If it is E2E supervised or Random Supervised
if (args.mode in ["e2e", "rand"]):
    base_model = cocoa.create_self_supervised_network(win_size=config["WINDOW"],
                                                      embedding_dim=config["CODE"],
                                                      modality=config["MODALITY_NAME"],
                                                      mode="base")
    ssl_time = 0
# ------------------------------------------------------------------
# Downstream
# ------------------------------------------------------------------

classifier_model = Classifier(data_path=args.datapath,
                              ds_config=config,
                              batch_size=args.batch,
                              label_ratio=args.labeleff
                              )
trn_score, tst_score, trn_fscore, tst_fscore, metrics = classifier_model.train_and_evalute(args.epoch, args.mode,
                                                                                           base_model)

# TSNE Visualisation
data = load_dataset(args.datapath, args.dataset, config['WINDOW'], args.batch, mode="test", state="all")
if args.tsne == 1:
    print("t-SNE visualization of learnt representation:")
    vis = EmbeddingVisualisation(args.dataset, os.path.join(OUTPUT_PATH, "plots"), class_size=config["CLASS"])
    lbl = list(tf.concat([y for x, y in data], axis=0))

    encoder = Model(classifier_model.classifier_model.input,
                    classifier_model.classifier_model.layers[4].output)
    embeddings = np.array(tf.concat([tf.concat(item, axis=-1) for item in [encoder(x) for x, y in data]], axis=0))

    vis.plot_tSNE(embeddings, lbl, train_name + "_finetuned", config['LABEL_NAME'])

report = [[args.mode, args.dataset, args.temp, args.batch, config["CODE"], config["WINDOW"], args.lr, args.labeleff,
           args.loss,
           args.sim, args.tau, args.beta,
           ssl_time, trn_score[0], trn_score[1], tst_score[0], tst_score[1], trn_fscore, tst_fscore, config['MODALITY_NAME']]]

df = pd.DataFrame(report, columns=df_columns)
df.to_csv(report_path, mode='a', header=False)
