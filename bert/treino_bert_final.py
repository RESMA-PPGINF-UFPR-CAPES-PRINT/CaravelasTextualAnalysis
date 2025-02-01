# -*- coding: utf-8 -*-
"""Caravelas - FINAL com tamanho sentença.ipynb


# Setup
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import random

SEED = 88
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf

tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

import tensorflow_hub as hub
import tensorflow_text as text
import timeit

from official.nlp import optimization  # to create AdamW optimizer
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.model_selection import StratifiedShuffleSplit
from statistics import mean, stdev

inicio = timeit.default_timer()
ID = 'bruta'
IDX = '471'
#IDX = '213'
#ID = '5169'
RODADA = IDX+'_R205'
#col_rotulo = 'rotulo_adaptado2'
col_rotulo = 'rotulo_adaptado1'
if ID=='bruta':
  #BASE = '../../bases/base_limpa_exp_ponto.csv'
  BASE = '../../bases/base_exp_bruta.csv'
  COLUNA = 'texto'
else:  
    BASE = '../../bases/combinadas/5169_userno_urlno_numno_emojno_unino_stopno_enelno_ascno_defle_lowtru.csv'
    COLUNA = 'normalizado'
#BASE_TESTE = '../../bases/base_teste_'+ID+'.csv'    
BASE_TESTE = '../../bases/base_teste_bruta_filename.csv'

df_train = pd.read_csv(BASE, sep=';')
df_train[COLUNA] = df_train[COLUNA].str.lower()
df_train['rotulo'] = df_train[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)

df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test[COLUNA] = df_test[COLUNA].str.lower()
df_test['rotulo'] = df_test[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)

# This model is cased: it does make a difference between english and English
# L-12 no nome significa que são 12 camadas ou seja é o base
preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4"

"""# Carrega os Dados"""

def make_bert_preprocessing_model(sentence_features, seq_length):

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(preprocess_url)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name="packer",
    )
    model_inputs = packer(segments)
    return keras.Model(input_segments, model_inputs)


bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]
def preprocess_text(text, seq_length):
  # print(seq_length)
  text = tf.convert_to_tensor([text])
  output = bert_preprocess_model([text], seq_length)
  output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
  return output

def preprocess_text_and_image(sample, seq_length):
  text = preprocess_text(sample[COLUNA], seq_length)
  return text

def dataframe_to_dataset(dataframe):
    columns = [COLUNA, "rotulo"]
    dataframe = dataframe[columns].copy()
    labels = dataframe.pop("rotulo")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    return ds

auto = tf.data.AUTOTUNE

def prepare_dataset(dataframe, bsize, seq_length):
  ds = dataframe_to_dataset(dataframe)
  ds = ds.map(lambda x, y: (preprocess_text_and_image(x, seq_length), y)).cache()
  ds = ds.batch(bsize).prefetch(auto)
  return ds

"""# Constroi Modelo"""

def otimizador(bsize, epochs, init_lr, warmup):
    steps_per_epoch = len(df_train)/bsize
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(warmup*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    return optimizer

def build_model(seq_length, optimizer):

  # Load the pre-trained BERT model to be used as the base encoder.
  bert = hub.KerasLayer(encoder_url, trainable=True, name="bert",)

  # Receive the text as inputs.
  bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
  text_inputs = {
      feature: keras.Input(shape=(seq_length,), dtype=tf.int32, name=feature)
      for feature in bert_input_features
  }

  # Generate embeddings for the preprocessed text using the BERT model.
  embeddings = bert(text_inputs)["pooled_output"]

  outputs2 = tf.keras.layers.Dropout(0.1)(embeddings)
  outputs2 = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(outputs2)

  model = tf.keras.Model(text_inputs, outputs2, name="text_encoder")

  loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  metrics=[BinaryAccuracy(), Precision(name='precision'), Recall(name='recall')]

  model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

  return model

# Early Stopping baseado na perda obtida na base treino
# o melhor é fazer monitoramento com dados de validacao, por causa do overfitting
callbacks = []
earlystop = EarlyStopping(monitor="val_loss", patience=25, verbose=1,)
callbacks.append(earlystop)

MDLBESTPRE = 'modelos/bestprecision'+ID+'_'+RODADA
checkpoint1 = ModelCheckpoint(MDLBESTPRE, monitor='val_precision', verbose=0, save_best_only=True, mode='max')
callbacks.append(checkpoint1)

#MDLBESTLOSS = "modelos/bestloss"+ID+RODADA
#checkpoint2 = ModelCheckpoint(MDLBESTLOSS, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
#callbacks.append(checkpoint2)

#MDLEPOCH = 'modelos/modelo{epoch:02d}'+ID+RODADA
#checkpoint4 = ModelCheckpoint(MDLEPOCH, save_freq=int(len(df_train)/16))
#callbacks.append(checkpoint4)

MDLBESTF1 = 'modelos/bestf1'+ID+'_'+RODADA
class CustomCallback(keras.callbacks.Callback):
    
    def on_train_begin(self, logs=None):
        self.best_f1 = 0
    def on_epoch_end(self, epoch, logs=None):
        current_precision = logs.get("val_precision")
        current_recall = logs.get("val_recall")
        current_f1 = 0
        if (current_precision+current_recall)>0:
            current_f1 = 2 * (current_precision*current_recall / (current_precision+current_recall))
        if np.greater(current_f1, self.best_f1):
            self.best_f1 = current_f1
            self.model.save(MDLBESTF1, save_format='tf')
            print("BEST F1",current_f1,'EPOCH',epoch)

callbacks.append(CustomCallback())

def get_class_weight(y):
  n_samples = len(y)
  n_classes = 2
  weight_for_0, weight_for_1 = n_samples / (n_classes * np.bincount(y))
  class_weight = {0: weight_for_0, 1: weight_for_1}
  return class_weight

class_weight = get_class_weight(df_train['rotulo'])

def calcf1(precision, recall):
   recall = float(recall)
   precision = float(precision)
   f1 = 0
   if (precision+recall)>0:
      f1 = 2 * (precision*recall / (precision+recall))
   return f1

"""# Treino"""

grid = [
        {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 344, 'EPOCHS':15, 'LR':3e-5, 'IDX':'471'}, # bruta 471
#        {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 384, 'EPOCHS':15, 'LR':5e-5, 'IDX':'544'}, # bruta 544
#        {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 164, 'EPOCHS':20, 'LR':3e-5,'IDX':'213'}, # norm 213
#        {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 512, 'EPOCHS':50, 'LR':3e-5,'IDX':'227'}, # norm 227
]

WARMUP = 0.1

y = df_train['rotulo']
class_weight = get_class_weight(y)

for param in grid:

      if param['IDX']!=IDX:
        continue

      EPOCHS = param['EPOCHS']
      LR = param['LR']
      BATCH_SIZE = param['BATCH_SIZE']
      SENTENCE_LENGTH = param['SENTENCE_LENGTH']
      print('BSIZE',BATCH_SIZE,'SENTENCE',SENTENCE_LENGTH)

      scores = {'precision': [], 'recall': [], 'f1': [], 'loss': []}

      tf.keras.backend.clear_session()
    
      bert_preprocess_model = make_bert_preprocessing_model(["text_1"], SENTENCE_LENGTH)
    
      train_ds = prepare_dataset(df_train, BATCH_SIZE, SENTENCE_LENGTH)
      val_ds = prepare_dataset(df_test, BATCH_SIZE, SENTENCE_LENGTH)

      optimizer = otimizador(BATCH_SIZE, EPOCHS, LR, WARMUP)
      model = build_model(SENTENCE_LENGTH, optimizer)
      history = model.fit(train_ds, validation_data=val_ds,
                          epochs=EPOCHS, 
                          callbacks=callbacks,
                          shuffle=False,
                          class_weight=class_weight)

      dfh = pd.DataFrame.from_dict(history.history)
      dfh['f1'] = dfh.apply(lambda x: calcf1(x.val_precision, x.val_recall), axis=1)
      FILE_HISTORICO = 'historico/bert_final_'+ID+RODADA+'.csv'
      FILE_TEMPO = 'tempo/bert_final_'+ID+RODADA+'.csv'
      dfh.to_csv(FILE_HISTORICO, index=False)
      result = dfh.sort_values(by=['f1'], ascending=False).head(1)

      print(result)

fim = timeit.default_timer()
duracao = fim - inicio
tempo = pd.DataFrame.from_dict({'inicio':[inicio],'fim':[fim],'duracao_seg':[duracao]})
tempo.to_csv(FILE_TEMPO, index=False)

