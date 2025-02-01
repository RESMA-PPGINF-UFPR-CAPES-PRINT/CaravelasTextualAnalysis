# -*- coding: utf-8 -*-
"""
AVALIA UM MODELO BERT
APLICA FILTRO AOS DADOS DE TESTE PARA SIMULAR PRODUCAO
# Setup
"""

import os
import pandas as pd
import numpy as np
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from official.nlp import optimization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow import keras

"""# Avalia o Modelo
"""
col_rotulo = 'rotulo_adaptado2'
#col_rotulo = 'rotulo_adaptado1'
#METRIC='f1'
METRIC='pre'
RODADA = '_471_R7'
BATCH_SIZE = 16
EPOCHS = 15
SENTENCE_LENGTH = 344
TAM_BASE = 1827
LR=3e-5
ID = 'bruta'
#ID = '5169'
#ID = '433'

COLUNA = 'normalizado'
if ID=='bruta':
    COLUNA = 'texto'

#BASE_TESTE = '../../bases/base_teste_'+ID+'.csv'
BASE_TESTE = '../../bases/base_teste_bruta_filename_motivo.csv'

preprocess_url = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"


def otimizador():
        steps_per_epoch = TAM_BASE/BATCH_SIZE
        num_train_steps = steps_per_epoch * EPOCHS
        num_warmup_steps = int(0.1*num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=LR,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type='adamw')
        return optimizer
optimizer = otimizador()
#loss = tf.keras.losses.BinaryCrossentropy()
#metrics=[Precision(name='precision'), Recall(name='recall')]
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metrics=[BinaryAccuracy(), Precision(name='precision'), Recall(name='recall')]

df_test = pd.read_csv(BASE_TESTE, sep=';')
df_test['rotulo'] = df_test[col_rotulo].apply(lambda r: 1 if r=='ACEITA' else 0)
df_test[COLUNA] = df_test[COLUNA].str.lower()
df_test = df_test[df_test['motivo'].isin(['MIDIA','ATENDE OS CRITÉRIOS'])]
print(len(df_test))

def make_bert_preprocessing_model(sentence_features, seq_length):

    input_segments = [
                         tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
                         for ft in sentence_features
                                     ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(preprocess_url)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    packer = hub.KerasLayer(
                                                    bert_preprocess.bert_pack_inputs,
                                                            arguments=dict(seq_length=seq_length),
                                                                    name="packer",
                                                                        )
    model_inputs = packer(segments)
    return keras.Model(input_segments, model_inputs)

bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]
def preprocess_text(text, seq_length):
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

bert_preprocess_model = make_bert_preprocessing_model(["text_1"], SENTENCE_LENGTH)

val_ds = prepare_dataset(df_test, BATCH_SIZE, SENTENCE_LENGTH)
if METRIC=='f1':
    path_to_model = '/home/hfrocha/envRLBERT/treino/modelos/bestf1'+ID+RODADA

if METRIC=='pre':
    path_to_model = '/home/hfrocha/envRLBERT/treino/modelos/bestprecision'+ID+RODADA

modelf1 = tf.keras.models.load_model(path_to_model, compile=False)
modelf1.compile(optimizer=optimizer, loss=loss, metrics=metrics)
result = modelf1.evaluate(val_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(ID, 'MODEL',METRIC,result, 'f1',f1)
print(result['precision'],',',result['recall'],',',f1)

"""
model1.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model2.compile(optimizer=optimizer, loss=loss, metrics=metrics)
modelv1.compile(optimizer=optimizer, loss=loss, metrics=metrics)
modelv2.compile(optimizer=optimizer, loss=loss, metrics=metrics)
"""

"""
result = model1.evaluate(val_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(ID, 'MODEL 1',result, 'f1', f1)

result = model2.evaluate(val_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(ID, 'MODEL 2',result, 'f1', f1)

result = modelv1.evaluate(val_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(ID, 'MODEL v1',result, 'f1', f1)

result = modelv2.evaluate(val_ds, verbose=0, return_dict=True)
f1 = 2 * (result['precision']*result['recall'] / (result['precision']+result['recall']))
print(ID, 'MODEL v2',result, 'f1', f1)
"""

