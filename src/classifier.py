import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Input, GlobalMaxPool1D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd

from utils.DataHelper import load_dataset
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

class Classifier:

    def __init__(
            self,
            data_path,
            ds_config,
            batch_size=32,
            reg_dense=0.005,
            reg_out=0.005,
            lr=0.0001,
            save_path="model/COCOA",
            label_ratio=1,
            combined=False
    ):
        self.classifier_model = None
        self.data_path = data_path
        self.ds_name = ds_config["DS_NAME"]
        self.num_classes = ds_config["CLASS"]
        self.batch_size = batch_size
        self.win = ds_config["WINDOW"]
        if combined:
            self.modality = ["all"]
            self.channel = [np.sum(ds_config["MODALITY_CH"])]
        else:
            self.modality = ds_config["MODALITY_NAME"]
            self.channel = ds_config["MODALITY_CH"]
        self.combined = combined
        self.reg_dense = reg_dense
        self.reg_out = reg_out
        self.save_path = save_path
        self.lr = lr
        self.label_efficiency = label_ratio

    def ds_generator(self, mode):
        ds_gen = load_dataset(self.data_path, self.ds_name, 200, self.batch_size, mode=mode, state="cls")
        return ds_gen

    def build_classifier(self, mode, input_shape):
        """ Build classifier
        """
        # Freeze layers of base_model
        if mode in ['ssl', 'rand']:
            for layer in self.base_model.layers:
                layer.trainable = False
        input_x = self.base_model.input
        xi = self.base_model(input_x)
        x = Concatenate()(xi)
        x = Flatten()(x)

        x = Dense(128, activation="relu", kernel_regularizer=l1(self.reg_dense))(x)
        ## Even for fine-tuning, it is recommended to freez the encoder for the first few epochs during classification.
        ## Make sure to compile once you freez/unfreez the trainables before calling fit()
        classifier_model = Dense(self.num_classes, activation="softmax", kernel_regularizer=l1(self.reg_out))(x)

        opt = Adam(lr=self.lr, amsgrad=True)

        # Combine encoder and extra layers
        c_model = Model(input_x, classifier_model)
        c_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

        return c_model

    def get_callbacks(self, fraction=0.8):
        """ Returns callbacks used while training the classifier
        """
        earlyStopping = EarlyStopping(
            monitor="val_categorical_accuracy",
            min_delta=0.00009,
            patience=6,
            verbose=2,
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_categorical_accuracy",
            min_delta=0.0009,
            patience=3,
            verbose=2,
            factor=0.2
        )

        return earlyStopping, reduce_lr

    def train_and_evalute(self, epochs, mode, base_model):
        trn_data = load_dataset(self.data_path,
                                self.ds_name,
                                self.win,
                                self.batch_size,
                                mode="train",
                                state="all",
                                label_efficiency=self.label_efficiency)
        val_data = load_dataset(self.data_path,
                                self.ds_name,
                                self.win,
                                self.batch_size,
                                mode="val",
                                state="all",
                                label_efficiency=self.label_efficiency)
        self.base_model = base_model
        if mode in ["ssl", "fine"]:
            self.base_model = Model(base_model.embedding_model.input,
                                    base_model.embedding_model.output)

        self.classifier_model = self.build_classifier(mode, input_shape=(self.win, self.channel[0]))
        self.training(trn_data, val_data, epochs=epochs)
        print("Evaluation on Validation set:")
        trn_score, trn_report, trn_fscore = self.evaluate(val_data)
        print("Evaluation on Training set:{}".format(trn_score))
        data = load_dataset(self.data_path,
                                      self.ds_name,
                                      self.win,
                                      self.batch_size,
                                      mode="test",
                                      state="all",
                                      combined=self.combined)
        print("Evaluation on Test set:")
        tst_score, tst_report, tst_fscore = self.evaluate(data)
        print("Evaluation on Test set:{}".format(tst_score))
        return trn_score, tst_score, trn_fscore, tst_fscore, self.classifier_model.metrics_names

    def training(self, data, val_data, epochs):
        earlyStopping, reduce_lr = self.get_callbacks()
        self.classifier_model.fit(data,
                                  validation_data=val_data,
                                  epochs=epochs,
                                  verbose=2,
                                  callbacks=[earlyStopping, reduce_lr]
                                  )

    def evaluate(self, data):
        """ Evaluation of the trained fine-tuned classifier.
            Minimum accuracy of 0.3 is imposed to avoid evaluation on diverged model.
        """
        score = self.classifier_model.evaluate(data, verbose=0)
        pred = self.classifier_model.predict(data, verbose=0)
        pred_class = tf.argmax(pred, axis=1)
        true_class = tf.argmax(list(tf.concat([y for x, y in data], axis=0)), axis=1)
        conf_mat = confusion_matrix(pred_class, true_class)
        print(pd.crosstab(true_class, pred_class, rownames=['True'], colnames=['Predicted'], margins=True))
        report = classification_report(true_class, pred_class)
        fscore = f1_score(true_class, pred_class, average="macro")

        return score, report, fscore
