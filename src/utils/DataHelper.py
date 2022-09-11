import json
import random

import numpy as np
from pandas import read_csv
import glob
import os
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def one_hot_encoder(in_array):
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(in_array)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    cls_onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return cls_onehot_encoded


def load_pamap2_ds(path, label_efficiency=1.0, mode="train", state="ssl"):
    if mode == 'vis':
        state = 'vis'
        mode = 'train'
    ext = '.npy'
    all_data = np.load(os.path.join(path, 'X_' + mode + ext))
    all_label = np.load(os.path.join(path, 'y_' + mode + ext))

    if state =="vis":
        x_val = np.load(os.path.join(path, 'X_val' + ext))
        all_data = np.vstack((all_data, x_val))
        y_val = np.load(os.path.join(path, 'y_val' + ext))
        all_label = np.vstack((all_label, y_val))

    sample_size = int(all_label.shape[0] * label_efficiency)
    all_data.astype('float32')
    idx = random.sample(range(0, all_label.shape[0]), sample_size)
    all_data = all_data[idx]
    all_label = all_label[idx]

    return all_data, all_label


def get_data_bundle(name, data):
    if name == "SLEEPEDF":
        return tf.data.Dataset.from_tensor_slices((data[:, :, [0]], data[:, :, [1]],
                                                   data[:, :, [2]], data[:, :, [3]]))

    if name == "PAMAP2":
        return tf.data.Dataset.from_tensor_slices((data[:, :, 0:3], data[:, :, 3:6],
                                                   data[:, :, 6:9],))
"""
def get_data_bundle(data, modNo, split_ratio,lbl=None):
    if modNo==1:
        return tf.data.Dataset.from_tensor_slices((data[0][0:split_ratio])), \
                tf.data.Dataset.from_tensor_slices((data[0][split_ratio:-1]))
    if modNo == 2:
        return tf.data.Dataset.from_tensor_slices((data[0][0:split_ratio], data[1][0:split_ratio])),\
               tf.data.Dataset.from_tensor_slices((data[0][split_ratio:-1], data[1][split_ratio:-1]))
    if modNo == 3:
        return tf.data.Dataset.from_tensor_slices((data[0][0:split_ratio], data[1][0:split_ratio], data[2][0:split_ratio])),\
               tf.data.Dataset.from_tensor_slices((data[0][split_ratio:-1], data[1][split_ratio:-1], data[2][split_ratio:-1]))
    if modNo == 4:
        return tf.data.Dataset.from_tensor_slices((data[0][0:split_ratio], data[1][0:split_ratio], data[2][0:split_ratio], data[3][0:split_ratio])),\
               tf.data.Dataset.from_tensor_slices((data[0][split_ratio:-1], data[1][split_ratio:-1], data[2][split_ratio:-1], data[3][split_ratio:-1]))
    else:
        return tf.data.Dataset.from_tensor_slices([data[i][0:split_ratio] for i in range(len(data))]), \
               tf.data.Dataset.from_tensor_slices([data[i][split_ratio:-1] for i in range(len(data))])
"""
def load_dataset(path, ds_name, win_size, batch_size, mode="train", state="ssl",label_efficiency=1,combined=False):

    if ds_name in ["PAMAP2"]:
        dataset, lbl = load_pamap2_ds(os.path.join(path, "pamap2"), label_efficiency, mode=mode, state=state)


    if state in "ssl":
        trn_data = get_data_bundle(ds_name,dataset)
        trn_data = trn_data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return trn_data

    data = get_data_bundle(ds_name, dataset)
    data = tf.data.Dataset.zip((data, tf.data.Dataset.from_tensor_slices(lbl)))
    data = data.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return data

