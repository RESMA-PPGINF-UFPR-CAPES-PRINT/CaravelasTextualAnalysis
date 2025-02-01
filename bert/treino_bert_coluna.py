# -*- coding: utf-8 -*-
"""Caravelas - 2 - Treino BERT.ipynb

Treino modificando usando colunas normalizadas

# Setup
"""


import os
import pandas as pd
import numpy as np
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization  # to create AdamW optimizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.model_selection import StratifiedShuffleSplit
from statistics import mean, stdev

"""## Constroi Modelo"""

# Early Stopping baseado na perda obtida na base treino
callbacks = []
earlystop = EarlyStopping(monitor="loss", patience=10, verbose=1,)
callbacks.append(earlystop)

# This model is cased: it does make a difference between english and English
# L-12 no nome significa que são 12 camadas ou seja é o base
preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"

EPOCHS = 50
BATCH_SIZE = 16
BASE = '../../bases/base_limpa_exp_geral.csv'

FOLDS = 5
TEST_SIZE = 30
RANDOM_STATE = 42
col_rotulo = 'rotulo_adaptado2'
vazias = []
colunas = ['texto','del_user','token_user','del_number','token_number','del_url','token_url',
          'lemma','stem','del_stopwords',
           'del_stopwords_cp','del_stopwords_cn','del_stopwords_av','del_stopwords_pp',
           'del_cp','del_cn','del_av','del_pp',
           'del_stopwords_hashtags_1','del_stopwords_hashtags_2',
          'enelvo1','enelvo2',
           'del_emoji','token_emoji','tra_emoji']
SALVAR_MODELO = False
historico = []
#FILE_HISTORICO = 'historico/bert_exp_geral_weight_adamw_50e_batch16.csv'
FILE_HISTORICO = 'historico/bert_exp_underline_weight_adamw_50e_batch16_texto.csv'
#FILE_HISTORICO = 'historico/bert_exp_ponto_weight_adamw_50e_batch16_texto.csv'
columns={0:'accuracy', 1:'accuracy_stdev',
         2:'precision', 3:'precision_stdev',
         4:'recall', 5:'recall_stdev',
         6:'f1', 7:'f1_stdev',
         8:'loss', 9: 'coluna', 10: 'rotulo', 11: 'tam_dev', 12: 'tam_treino'}


df1 = pd.read_csv(BASE, sep=';')
#df1 = df0.head(20).copy()
df1['rotulo'] = df1[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)

def otimizador():
    steps_per_epoch = len(df1)/BATCH_SIZE
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

  #optimizer = tf.keras.optimizers.AdamW()
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

"""# Treino"""


# loop nas colunas
for i, coluna in enumerate(colunas):

  X = df1.copy()

  # Exclui amostras vazias
  if coluna in vazias:
    X.drop(X[X[coluna].str.strip()==''].index, inplace=True)
    X.drop(X[X[coluna].str.strip().isna()].index, inplace=True)
    X.reset_index(inplace=True)

  y = X['rotulo']
  class_weight = get_class_weight(y)

  scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'loss': []}
  kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)

  for k, (train_index, val_index) in enumerate(kf.split(X, y)):

    print('fold',k)

    tf.keras.backend.clear_session()

    X_train, y_train = X.loc[train_index], y.loc[train_index]
    X_validation, y_validation = X.loc[val_index], y.loc[val_index]

    X_train = X_train[coluna]
    X_validation = X_validation[coluna]

    model = build_model()
    model.fit(x=X_train, y=y_train, epochs=EPOCHS, callbacks=callbacks, class_weight=class_weight)
    result = model.evaluate(X_validation, y_validation, verbose=0, return_dict=True)

    #print(result)
    precision = result['precision']
    recall = result['recall']
    f1 = 0
    if (precision+recall)>0:
      f1 = 2 * (precision*recall / (precision+recall))

    scores['accuracy'].append(result['binary_accuracy'])
    scores['loss'].append(result['loss'])
    scores['precision'].append(precision)
    scores['recall'].append(recall)
    scores['f1'].append(f1)

    # salva o modelo e vetorizador
    if SALVAR_MODELO is True:
      pass

  historico.append([
                    mean(scores['accuracy']), stdev(scores['accuracy']),
                    mean(scores['precision']), stdev(scores['precision']),
                    mean(scores['recall']), stdev(scores['recall']),
                    mean(scores['f1']), stdev(scores['f1']),
                    mean(scores['loss']), stdev(scores['loss']),
                    coluna, col_rotulo, len(X), len(X_train)])
  print('média', mean(scores['f1']))

  """### Salva Histórico"""
  df = pd.DataFrame.from_dict(historico)
  df.rename(columns=columns, inplace=True)
  df.to_csv(FILE_HISTORICO, index=False)


