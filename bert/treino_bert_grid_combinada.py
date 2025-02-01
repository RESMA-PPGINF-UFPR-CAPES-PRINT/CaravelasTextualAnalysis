# -*- coding: utf-8 -*-
"""
Treina coma as bases com normalização combinadas

"""

import numpy as np
import os
import pandas as pd
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from tensorflow import keras
from os.path import exists
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from statistics import mean, stdev

from official.nlp import optimization  # to create AdamW optimizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"

# Early Stopping baseado na perda obtida na base treino
callbacks = []
earlystop = EarlyStopping(monitor="loss", patience=10, verbose=1,)
callbacks.append(earlystop)

filepath = 'modelos/bertbestf1cross'

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.best_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        current_precision = logs.get("val_precision")
        current_recall = logs.get("val_recall")
        current_f1 = 0
        if (current_precision+current_recall)>0:
            current_f1 = 2 * (current_precision*current_recall / (current_precision+current_recall))
        print("\nEnd epoch {} of training f1 {}".format(epoch, current_f1))
        if np.less(self.best_f1, current_f1):
            self.best_f1 = current_f1
            print("\nBest F1 now is {}".format(current_f1))
            self.model.save(filepath, save_format='tf')

#callbacks.append(CustomCallback())

# This model is cased: it does make a difference between english and English
# L-12 no nome significa que são 12 camadas ou seja é o base
preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"

tam_base = 1827
EPOCHS = 15
BATCH_SIZE = 16
def otimizador():
    steps_per_epoch = tam_base/BATCH_SIZE
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    return optimizer

def build_model():

  input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(preprocess_url, name='preprocessing')
  encoder_inputs = preprocessing_layer(input_layer)
  encoder = hub.KerasLayer(encoder_url, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  model = tf.keras.Model(input_layer, net)

  optimizer = otimizador()
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metrics=[BinaryAccuracy(), Precision(name='precision'), Recall(name='recall')]

  model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

  return model

# Calculo do Scikit
# Fonte: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

def get_class_weight(y):
  n_samples = len(y)
  n_classes = 2
  weight_for_0, weight_for_1 = n_samples / (n_classes * np.bincount(y))
  class_weight = {0: weight_for_0, 1: weight_for_1}
  return class_weight

def calcf1(precision, recall):
   recall = float(recall)
   f1 = 0
   if (precision+recall)>0:
      f1 = 2 * (precision*recall / (precision+recall))
   return f1

"""### Modelo"""

PATH = '../../bases/combinadas/'

#dfg = pd.read_csv('../../bases/combinadas/grid_stem.csv', sep=';')
dfg = pd.read_csv('historico/grid_historico_bert_stem.csv', sep=';')
dfg = dfg[ (dfg['mencao']!='tokenizar') & (dfg['numero']!='tokenizar') & (dfg['url']!='remover') & (dfg['emoji']!='traduzir') & (dfg['emoji']!='remover')]
print('QT',len(dfg))

coluna = 'normalizado'
col_rotulo = 'rotulo_adaptado2'
TEST_SIZE = 30
FOLDS = 3
RANDOM_STATE = 42


for param in dfg.itertuples():

  BASE = PATH + str(param.id) +'_' + param.nome+'.csv'

  if os.path.exists(BASE) is False:
    print('nao existe', BASE)
    break

  X = pd.read_csv(BASE, sep=';')

  if param.f1>0:
    print(param.f1)
    continue

  if len(X[X[coluna].str.strip()==''])>0 or len(X[X[coluna].str.strip().isna()])>0:
    X.drop(X[X[coluna].str.strip()==''].index, inplace=True)
    X.drop(X[X[coluna].str.strip().isna()].index, inplace=True)
    X.reset_index(inplace=True)

  X['rotulo'] = X[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)
  y = X['rotulo']

  class_weight = get_class_weight(y)

  scores = {'precision': [], 'recall': [], 'f1': [], 'loss': []}
  kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)

  for k, (train_index, val_index) in enumerate(kf.split(X, y)):
    print('FOLD',k)
    tf.keras.backend.clear_session()
    
    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_validation, y_validation = X.loc[val_index], y.loc[val_index]
    X_train = X_train[coluna]
    X_validation = X_validation[coluna]

    model = build_model()
    historico = model.fit(x=X_train, y=y_train, validation_data=(X_validation, y_validation), epochs=EPOCHS, callbacks=callbacks, class_weight=class_weight)

    dfh = pd.DataFrame.from_dict(historico.history)
    dfh['f1'] = dfh.apply(lambda x: calcf1(x.val_precision, x.val_recall), axis=1)

    result = dfh.sort_values(by=['f1'], ascending=False).head(1)
    precision = result['val_precision'].item()
    recall = result['val_recall'].item()
    f1 = result['f1'].item()

    scores['loss'].append(result['loss'].item())
    scores['precision'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)

  dfg.loc[dfg['id']==param.id, 'precision'] = mean(scores['precision'])
  dfg.loc[dfg['id']==param.id, 'precision_stdev'] = stdev(scores['precision'])
  dfg.loc[dfg['id']==param.id, 'recall'] = mean(scores['recall'])
  dfg.loc[dfg['id']==param.id, 'recall_stdev'] = stdev(scores['recall'])
  dfg.loc[dfg['id']==param.id, 'f1'] = mean(scores['f1'])
  dfg.loc[dfg['id']==param.id, 'f1_stdev'] = stdev(scores['f1'])
  dfg.loc[dfg['id']==param.id, 'tam_dev'] = len(X)
  dfg.loc[dfg['id']==param.id, 'tam_treino'] = len(X_train)

  dfg.to_csv('historico/grid_historico_bert_stem.csv', index=False, sep=';')
