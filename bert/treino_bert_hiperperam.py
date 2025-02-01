# -*- coding: utf-8 -*-
# Otimizacao de hiperparametros
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization  # to create AdamW optimizer
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from sklearn.model_selection import StratifiedShuffleSplit
from statistics import mean, stdev

FOLDS = 5
TEST_SIZE = 30
RANDOM_STATE = 42
col_rotulo = 'rotulo_adaptado2'
#BASE = '../../bases/base_limpa_exp_ponto.csv'
BASE = '../../bases/combinadas/5169_userno_urlno_numno_emojno_unino_stopno_enelno_ascno_defle_lowtru.csv'
#COLUNA = 'texto'
COLUNA = 'normalizado'
historico = []
#FILE_HISTORICO = 'historico/bert_hiperparam_p18.csv'
FILE_HISTORICO = 'historico/bert_hiperparam_norm_r9.csv'
columns={0:'precision', 1:'precision_stdev',
         2:'recall', 3:'recall_stdev',
         4:'f1', 5:'f1_stdev',
         6:'loss', 7: 'loss_stdev',
         8:'sentence_len', 9:'batch_size',
         10:'lr',11:'warmup',
         12:'epoch',13:'paciencia'}

df_train = pd.read_csv(BASE, sep=';')
df_train['texto'] = df_train['texto'].str.lower()
df_train['rotulo'] = df_train[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)

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
earlystop = EarlyStopping(monitor="val_loss", patience=5, verbose=1,)
callbacks.append(earlystop)

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

grid_sl = [
#    {'BATCH_SIZE': 64, 'SENTENCE_LENGTH': 64},
#    {'BATCH_SIZE': 64, 'SENTENCE_LENGTH': 95},
     
#    {'BATCH_SIZE': 32, 'SENTENCE_LENGTH': 64},
#    {'BATCH_SIZE': 32, 'SENTENCE_LENGTH': 96},
#    {'BATCH_SIZE': 32, 'SENTENCE_LENGTH': 128},
#    {'BATCH_SIZE': 32, 'SENTENCE_LENGTH': 164},
#    {'BATCH_SIZE': 32, 'SENTENCE_LENGTH': 200},

#    {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 64},
#    {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 96},
{'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 128},
#   {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 164},
#   {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 200},
#   {'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 343},
#{'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 128, 'EPOCHS':15},
#{'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 164, 'EPOCHS':10},
#{'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 200, 'EPOCHS':20},
#{'BATCH_SIZE': 16, 'SENTENCE_LENGTH': 343, 'EPOCHS':15},

#    {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 64},
#    {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 96},
#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 128},
#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 164},

#     {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 128, 'EPOCHS':20},
#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 164, 'EPOCHS':15},
#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 164, 'EPOCHS':20},
#    {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 200},
#    {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 384},

#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 384, 'EPOCHS':15},
#{'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 384, 'EPOCHS':20i},
#    {'BATCH_SIZE': 8, 'SENTENCE_LENGTH': 512}
]

grid_epoch = [50]
#grid_lr = [3e-5]
LR=3e-5
WARMUP = 0.1
PACIENCIA = 15

X = df_train
y = df_train['rotulo']

for EPOCHS in grid_epoch:
#for LR in grid_lr:

    for param in grid_sl:
      #EPOCHS = 10
      #EPOCHS = param['EPOCHS']
      BATCH_SIZE = param['BATCH_SIZE']
      SENTENCE_LENGTH = param['SENTENCE_LENGTH']
      print('BSIZE',BATCH_SIZE,'SENTENCE',SENTENCE_LENGTH)

      scores = {'precision': [], 'recall': [], 'f1': [], 'loss': []}
      kf = StratifiedShuffleSplit(n_splits=FOLDS, test_size=(TEST_SIZE/100), random_state=RANDOM_STATE)

      for k, (train_index, val_index) in enumerate(kf.split(X, y)):
    
        print('fold', k)
    
        tf.keras.backend.clear_session()
    
        X_train = X.loc[train_index]
        X_validation = X.loc[val_index]
    
        bert_preprocess_model = make_bert_preprocessing_model(["text_1"], SENTENCE_LENGTH)
    
        train_ds = prepare_dataset(X_train, BATCH_SIZE, SENTENCE_LENGTH)
        val_ds = prepare_dataset(X_validation, BATCH_SIZE, SENTENCE_LENGTH)

        #optimizer = tf.keras.optimizers.AdamW(3e-5, weight_decay=0.01)
        earlystop = EarlyStopping(monitor="val_loss", patience=PACIENCIA, verbose=1,)
        optimizer = otimizador(BATCH_SIZE, EPOCHS, LR, WARMUP)
        model = build_model(SENTENCE_LENGTH, optimizer)
        history = model.fit(train_ds, validation_data=val_ds,
                          epochs=EPOCHS, 
                          callbacks=[earlystop],
                          class_weight=class_weight)

        dfh = pd.DataFrame.from_dict(history.history)
        dfh['f1'] = dfh.apply(lambda x: calcf1(x.val_precision, x.val_recall), axis=1)
        result = dfh.sort_values(by=['f1'], ascending=False).head(1)

        scores['loss'].append(result['val_loss'].item()) # tava errado aqui
        scores['precision'].append(result['val_precision'].item())
        scores['recall'].append(result['val_recall'].item())
        scores['f1'].append(result['f1'].item())

      historico.append([
                    mean(scores['precision']), stdev(scores['precision']),
                    mean(scores['recall']), stdev(scores['recall']),
                    mean(scores['f1']), stdev(scores['f1']),
                    mean(scores['loss']), stdev(scores['loss']),
                    SENTENCE_LENGTH, BATCH_SIZE, LR, WARMUP, EPOCHS, PACIENCIA])

      df = pd.DataFrame.from_dict(historico)
      df.rename(columns=columns, inplace=True)
      df.to_csv(FILE_HISTORICO, index=False)

